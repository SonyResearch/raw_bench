import json
from loguru import logger
import os
from omegaconf import DictConfig
import pandas as pd
from pathlib import Path
from qqdm import qqdm
import torch
from torchmetrics.audio.snr import ScaleInvariantSignalNoiseRatio
from typing import Dict, Tuple, Union

from .base import Solver
from ..model.timbre import Decoder, Encoder


class SolverTimbre(Solver):
    def __init__(
        self,
        config: DictConfig
    ):
        """
        Initialize the SolverTimbre class.

        Args:
            config: DictConfig
                Configuration object.
        """
        super(SolverTimbre, self).__init__(config)
        self.msg_len = config.message.len
        self.win_len = config.process.audio.win_len
        self.embedding_dim = config.model.dim.embedding

        self.build_model(vocoder_config=config.vocoder_config,
                         vocoder_checkpoint=config.vocoder_checkpoint)

        self.load_models(config.checkpoint)
        logger.info("models loaded")

        self.exp_logger.log_hparams(config)
                
    def build_model(
        self,
        vocoder_checkpoint,
        vocoder_config
    ):
        """
        Build and initialize the encoder and decoder models.
        Args:
            vocoder_checkpoint: str
                Path to the pretrained vocoder checkpoint file.
            vocoder_config: str
                Path to configuration file for the vocoder.
        """
        self.encoder = Encoder(self.config.process, 
                               self.config.model, 
                               self.msg_len, 
                               self.win_len,
                               self.embedding_dim, 
                               nlayers_encoder=self.config.model.layer.nlayers_encoder, 
                               attention_heads=self.config.model.layer.attention_heads_encoder).to(self.device)

        self.decoder = Decoder(self.config.process,
                               self.config.model, 
                               self.msg_len, 
                               self.win_len, 
                               self.embedding_dim, 
                               nlayers_decoder=self.config.model.layer.nlayers_decoder, 
                               attention_heads=self.config.model.layer.attention_heads_decoder,
                               vocoder_config=vocoder_config,
                               vocoder_checkpoint= vocoder_checkpoint).to(self.device)
                               
    def eval_mode(
        self
    ):
        """
        Set encoder and decoder models to evaluation mode.
        """
        self.encoder.eval()
        self.decoder.eval()
        self.decoder.robust = False # because we directly manipulate the audio

    def load_models(
        self,
        checkpoint: Union[Path, str]
    ):
        """
        Load encoder and decoder model weights from a checkpoint.

        Args:
            checkpoint: Union[Path, str]
                Path to the checkpoint directory or file.
        """
        ckpt=torch.load(checkpoint, map_location=self.device)
        logger.info("model <<{}>> loaded".format(checkpoint))

        self.encoder.load_state_dict(ckpt["encoder"])
        self.decoder.load_state_dict(ckpt["decoder"], strict=False)

    def eval(
        self,
        epoch_num: int = None,
        write_to_disk: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Evaluate the model on the test set and compute metrics.

        Args:
            epoch_num: int, optional
                Epoch number for saving results.
            write_to_disk: bool, optional
                Whether to write results to disk.

        Returns:
            Tuple[pd.DataFrame, Dict[str, float]]:
                DataFrame of results and dictionary of last computed metrics.
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
                message = self.random_message(self.msg_len, audio_chunks.shape[0])
                y = audio_chunks.to(self.device)
                message = message.to(self.device).to(torch.float32)
                message = message.unsqueeze(1)

                y_wm, _ = self.encoder.test_forward(y, message)

                y_wm_dirty = torch.zeros_like(y_wm)
                y_dirty = torch.zeros_like(y)

                for b in range(y_wm_dirty.shape[0]):
                    cur_losses_log = {}
                    args = {} if attack_params[b] is None else json.loads(attack_params[b])
                    
                    if att_types[b] == 'phase_shift':
                        # During test and validation, the phase shift parameter is in seconds.
                        args['shift'] = int(args['shift'] * self.sample_rate)   
                                        
                    y_wm_dirty[b, ...] = self.audio_attack(y_wm[b, ...],
                                                          attack_type=att_types[b],
                                                          **args)
                    y_dirty[b, ...] = self.audio_attack(y[b, ...],
                                                        attack_type=att_types[b],
                                                        **args)
                
                    y_wm_decoded = self.decoder.test_forward(y_wm[b, ...].unsqueeze(0))
                    y_wm_dirty_decoded = self.decoder.test_forward(y_wm_dirty[b, ...].unsqueeze(0))
                    y_decoded = self.decoder.test_forward(y[b, ...].unsqueeze(0))
                    y_dirty_decoded = self.decoder.test_forward(y_dirty[b, ...].unsqueeze(0))

                    cur_losses_log['bitwise/clean'] = self.__bitwise_acc(y_wm_decoded, message[b]).item()
                    cur_losses_log['bitwise/distorted'] = self.__bitwise_acc(y_wm_dirty_decoded, message[b]).item()
                    cur_losses_log['bitwise/no_watermark_clean'] = self.__bitwise_acc(y_decoded, message[b]).item()
                    cur_losses_log['bitwise/no_watermark_distorted'] = self.__bitwise_acc(y_dirty_decoded, message[b]).item()
                    hard_metics = {key.replace('bitwise/', 'hard/'): int(value == 1.0) for key, value in cur_losses_log.items()}
                    cur_losses_log.update(hard_metics)    

                    if self.config.full_perceptual:
                        perceptual_metrics = self.compute_perceptual_metrics(audio_filepath=audio_filepaths[0],
                                                                            start_time=start_times[0],
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
                        [audio_filepaths[0], 
                         datasets[0], 
                         att_types[0], 
                         attack_params[0], 
                         chunk_indices[0].item()] + [val for val in cur_losses_log.values()]
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

    @staticmethod
    def random_message(
        nbits: int,
        batch_size: int
    ) -> torch.Tensor:
        """
        Generate a random message as a 0/1 tensor.

        Args:
            nbits: int
                Number of bits in the message.
            batch_size: int
                Number of messages to generate.

        Returns:
            torch.Tensor: Random message tensor.
        """
        if nbits == 0:
            return torch.tensor([])
        
        return torch.randint(0, 2, (batch_size, nbits)) * 2 - 1

    def __bitwise_acc(
        self,
        decoded: torch.Tensor,
        message: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute bitwise accuracy between decoded and original message tensors.

        Args:
            decoded: torch.Tensor
                Decoded message tensor.
            message: torch.Tensor
                Original message tensor.

        Returns:
            torch.Tensor: Bitwise accuracy.
        """
        return (decoded >= 0).eq(message >= 0).sum().float() / self.msg_len
