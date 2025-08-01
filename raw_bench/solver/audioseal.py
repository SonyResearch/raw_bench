import os
import json
from pathlib import Path
from typing import Union

import pandas as pd
import torch

from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from qqdm import qqdm
from torchmetrics.audio.snr import ScaleInvariantSignalNoiseRatio


from audiocraft.models.builders import get_watermark_model

from .base import Solver
from ..utils import compute_bit_acc

class SolverAudioSeal(Solver):
    def __init__(
        self, 
        config: Union[Path, DictConfig]
    ):
        """
        Initialize the SolverAudioSeal class. This is a wrapper class for the AudioSeal.

        Args:
            config (DictConfig): Configuration containing all the settings for the solver.
        """        
        Solver.__init__(self, config)
        self.rng:  torch.Generator  # set at each epoch
        self.model: None
        if hasattr(config, "fsdp"):
            assert not getattr(config.fsdp, "use", False
            ), "FSDP not supported by WatermarkSolver."

        self.path_specs = os.path.join(self.run_dir, "spectrograms")
        os.makedirs(self.path_specs, exist_ok=True)
        self.build_model()
        
        self.load_models(config.checkpoint)
        try:
            hparam_path = os.path.join(config.checkpoint, "../../hparams.yaml")
            with open(hparam_path, 'r') as f:
                hparams = OmegaConf.load(f)
            logger.info(f"Loaded hyperparameters from {hparam_path}")
        except Exception as e:
            hparams = None

        logger.info(f"Loaded generator and detector from {config.checkpoint}")
                        
        self.exp_logger.log_hparams(config, hparams)

    def eval(
        self,
        epoch_num: int = None,
        write_to_disk: bool = True):
        """
        Evaluate the model on the test set.

        Args:
            epoch_num (int, optional): Epoch number for naming output files.
            write_to_disk (bool): Whether to write results to disk.

        Returns:
            Tuple[pd.DataFrame, dict]: DataFrame of results and dictionary of current loss logs.
        """        

        if self.config.test_suffix is not None:
            csv_suffix = '_' + self.config.test_suffix 
        self.eval_mode()
        logger.info("Start evaluation.")
        
        if self.audio_attack is not None:
            self.audio_attack.set_mode('test')

        res_list = []
        column_names = ['audio_filepath', 
                        'dataset', 
                        'attack_type', 
                        'attack_params', 
                        'chunk_index']

        sisnr_f = ScaleInvariantSignalNoiseRatio().to(self.device)
        
        self.seed_everything(self.config.random_seed_for_eval)
        with torch.inference_mode():
            num_items = 0  # Keep track of the number of batches proczessed
            for ret in qqdm(self.test_loader):
                audio_chunks, audio_filepaths, datasets, att_types, attack_params, chunk_indices, start_times = ret
                y = audio_chunks.to(self.device)
                message = self.random_message(self.model.nbits, batch_size=y.shape[0]).to(self.device)
                watermark = self.model.get_watermark(y, message)
                y_wm = y + watermark
                
                y_wm_dirty = torch.zeros_like(y_wm)
                y_dirty = torch.zeros_like(y)

                # currently we only support single batch for evaluation: b must be always 0
                for b in range(y_wm_dirty.shape[0]):
                    cur_losses_log = {}
                    args = {} if attack_params[b] is None else json.loads(attack_params[b])
                    
                    if att_types[b] == 'phase_shift':
                        # During test and validation, the phase shift parameter is in seconds.
                        args['shift'] = int(args['shift'] * self.sample_rate)   
                    
                    length = y[b, ...].shape[-1]
                    y_wm_dirty[b, ...] = self.audio_attack(y_wm[b, ...], attack_type=att_types[b], **args)[..., :length]
                    y_dirty[b, ...] = self.audio_attack(y[b, ...], attack_type=att_types[b], **args)[..., :length]

                    # now the each attack can handle device by itself
                    # we do not need this the line below          
                    y_wm_decoded = self.model.detect_watermark(y_wm[b, ...].unsqueeze(0))
                    y_wm_dirty_decoded = self.model.detect_watermark(y_wm_dirty[b, ...].unsqueeze(0))
                    y_decoded = self.model.detect_watermark(y[b, ...].unsqueeze(0))
                    y_dirty_decoded = self.model.detect_watermark(y_dirty[b, ...].unsqueeze(0))

                    cur_losses_log['bitwise/clean'] = compute_bit_acc(y_wm_decoded, message[b]).item()
                    cur_losses_log['bitwise/distorted'] = compute_bit_acc(y_wm_dirty_decoded, message[b]).item()
                    cur_losses_log['bitwise/no_watermark_clean'] = compute_bit_acc(y_decoded, message[b]).item()
                    cur_losses_log['bitwise/no_watermark_distorted'] = compute_bit_acc(y_dirty_decoded, message[b]).item()
                        
                    hard_metics = {key.replace('bitwise/', 'hard/'): int(value == 1.0) for key, value in cur_losses_log.items()}
                    cur_losses_log.update(hard_metics)

                    if self.config.full_perceptual:
                        perceptual_metrics = self.compute_perceptual_metrics(audio_filepath=audio_filepaths[b],
                                                                            start_time=start_times[b],
                                                                            audio_duration=self.config.dataset.eval_seg_duration,
                                                                            watermarked_audio=y_wm[b, ...],
                                                                            distorted_audio=y_dirty[b, ...])

                        cur_losses_log.update(perceptual_metrics)

                    cur_losses_log['sisnr_wm'] =  sisnr_f(y_wm[b], y[b]).item()
                    cur_losses_log['sisnr_attack'] =  sisnr_f(y_dirty[b], y[b]).item()

                    log_dir = {f"{att_types[b]}/{key}": val for key, val in cur_losses_log.items()}
                    self.exp_logger.log_metric(log_dir, step=num_items)
                        
                    num_items += 1
                    res_list.append(
                        [audio_filepaths[b], 
                         datasets[b], 
                         att_types[b], 
                         attack_params[b], 
                         chunk_indices[b].item()] + [val for val in cur_losses_log.values()]
                    )
            column_names += list(cur_losses_log.keys())    
            df_result = pd.DataFrame(res_list, columns=column_names)
        
        key_columns = ['bitwise/clean', 'bitwise/distorted', 'hard/clean', 'hard/distorted', 'sisnr_wm', 'sisnr_attack']

        # per chunk aggregation
        self.compute_agg(df_result, cur_losses_log.keys(), csv_suffix, key_columns, prefix='chunklv')

        # write raw results to disk
        if write_to_disk:
            os.makedirs(self.test_results_dir, exist_ok=True)
            if epoch_num is not None:
                output_csv_filename = f'test_results_epoch{csv_suffix}{epoch_num}.csv'
            else:
                output_csv_filename = f'test_results{csv_suffix}.csv'
        
            df_result.to_csv(os.path.join(self.test_results_dir, output_csv_filename), 
                             sep=self.csv_delimiter,
                             index=False)

        return df_result, cur_losses_log

    def build_model(self):
        """
        Instantiate the watermark model and optimizer as specified in the configuration.
        """
        # Model and optimizer
        self.model = get_watermark_model(self.config)

    def load_models(
        self, 
        checkpoint: Union[Path, str]):
        """
        Load generator and detector model weights (and optimizer state if in training mode) from checkpoint directory.

        Args:
            checkpoint (Union[Path, str]): Path to the checkpoint directory.
        """
        gen_ckpt = torch.load(os.path.join(checkpoint, "generator_base.pth"), map_location=self.device)
        det_ckpt = torch.load(os.path.join(checkpoint, "detector_base.pth"), map_location=self.device)
        if 'model' in gen_ckpt:
            gen_ckpt = gen_ckpt['model']
        if 'model' in det_ckpt:
            det_ckpt = det_ckpt['model']                
        self.model.generator.load_state_dict(gen_ckpt)
        self.model.detector.load_state_dict(det_ckpt)

    def eval_mode(self):
        """
        Set the model to evaluation mode.
        """        
        self.model.eval()
        super(SolverAudioSeal, self).eval_mode()

    # copied from https://github.com/facebookresearch/audiocraft/blob/896ec7c47f5e5d1e5aa1e4b260c4405328bf009d/audiocraft/solvers/watermark.py#L654
    def evaluate_audio_watermark(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
    ) -> dict:
        """Audio reconstruction evaluation method that can be conveniently pickled."""
        metrics = {}
        if self.config.full_perceptual:
            metrics["visqol"] = self.visqol(y_pred, y, self.sample_rate)
        metrics["sisnr"] = self.sisnr_f(y_pred, y)
        metrics["stoi"] = self.stoi_f(y_pred, y)
        # metrics["pesq"] = tensor_pesq(y_pred, y, sr=self.sample_rate)
        return metrics

    # copied from https://github.com/facebookresearch/audiocraft/blob/896ec7c47f5e5d1e5aa1e4b260c4405328bf009d/audiocraft/solvers/watermark.py#L69
    @staticmethod
    def random_message(nbits: int,
                       batch_size: int) -> torch.Tensor:
        """Return random message as 0/1 tensor."""
        if nbits == 0:
            return torch.tensor([])
        
        return torch.randint(0, 2, (batch_size, nbits))
