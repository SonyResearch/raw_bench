import json
import math
import os
import time
from collections import defaultdict, OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from qqdm import qqdm
from ..model.silentcipher import CarrierDecoder, Encoder, MsgDecoder
from torchmetrics.audio.snr import ScaleInvariantSignalNoiseRatio

from .base import Solver
from ..utils import AVERAGE_ENERGY_VCTK

DEC_CFG = SimpleNamespace(
    ensure_negative_message=True,
    no_normalization=False
)

class SolverSilentCipher(Solver):
    def __init__(
        self, 
        config: Union[Path, DictConfig]
        ):
        """        Initialize the SolverSilentCipher class.

        Args:
            config: DictConfig
                Configuration object.
        """

        Solver.__init__(self, config)

        self.sample_rate = config.sample_rate

        # Message parameters
        msg_config = config.message
        self.msg_len = msg_config.len
        self.msg_dim = msg_config.dim
        self.msg_band_size = msg_config.band_size
        self.msg_sdr = msg_config.sdr
        self.n_messages = msg_config.n_messages

        self.build_model()
        self.eval_mode()
        
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
        
        logger.info(f"Initialized SolverSilentCipher with device: {self.device}")

    def build_model(self):
        model_cfg = self.config.model
        self.enc_c = Encoder(n_layers=model_cfg.encoder.n_layers,
                             out_dim=model_cfg.encoder.out_dim,
                             message_dim=self.msg_dim,
                             message_band_size=self.msg_band_size,
                             n_fft=self.config.stft.n_fft)
        
        self.dec_c = CarrierDecoder(config=DEC_CFG,
                                    conv_dim=model_cfg.decoder.c_conv_dim,
                                    n_layers=model_cfg.decoder.c_n_layers,
                                    message_band_size=self.msg_band_size)

        self.dec_m = [MsgDecoder(message_dim=self.msg_dim,
                                 message_band_size=self.msg_band_size) for _ in range(self.n_messages)]
    
        # Move models to device (just in case)
        self.enc_c.to(self.device)
        self.dec_c.to(self.device)
        for model in self.dec_m:
            model.to(self.device)
    
    def eval_mode(self):
        super(SolverSilentCipher, self).eval_mode()
        self.enc_c.eval()
        self.dec_c.eval()
        for model in self.dec_m:
            model.eval()
            
    @staticmethod
    def __strip_module_prefix(state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("module.", "", 1)
            new_state_dict[new_key] = v
        return new_state_dict
    
    def load_models(
        self, 
        checkpoint: Union[Path, str]):
        """
        Load generator and detector model weights (and optimizer state if in training mode) from checkpoint directory.

        Args:
            checkpoint (Union[Path, str]): Path to the checkpoint directory.
        """
        if checkpoint is not None:
            hparam_path = os.path.join(checkpoint, "hparams.yaml")
            with open(hparam_path, 'r') as f:
                hparams = OmegaConf.load(f)
                assert self.msg_band_size == hparams.message_band_size, "Message band size mismatch"
                assert self.msg_dim == hparams.message_dim, "Message dimension mismatch"
                assert self.msg_len == hparams.message_len, "Message length mismatch"
                assert self.msg_sdr == hparams.message_sdr, "Message SDR mismatch"
                assert self.n_messages == hparams.n_messages, "Number of messages mismatch"
                assert self.sample_rate == hparams.SR, "Sample rate mismatch"
                assert self.config.stft.n_fft == hparams.N_FFT,   "STFT N_FFT mismatch"
                assert self.config.stft.hop_len == hparams.HOP_LENGTH, "STFT hop length mismatch"            
                logger.info(f"Loaded hyperparameters from {hparam_path}")
            try:
                # Load encoder
                enc_path = os.path.join(checkpoint, "enc_c.ckpt")
                enc_state_dict = torch.load(enc_path, map_location=self.device)
                self.enc_c.load_state_dict(self.__strip_module_prefix(enc_state_dict))
                logger.info(f"Loaded encoder from {enc_path}")
        
                # Load carrier decoder
                dec_c_path = os.path.join(checkpoint, "dec_c.ckpt")
                dec_c_state_dict = torch.load(dec_c_path, map_location=self.device)
                self.dec_c.load_state_dict(self.__strip_module_prefix(dec_c_state_dict))
                logger.info(f"Loaded carrier decoder from {dec_c_path}")
        
                # Load message decoders
                for i, m in enumerate(self.dec_m):
                    dec_m_path = os.path.join(checkpoint, f"dec_m_{i}.ckpt")
                    try:
                        dec_m_i_state_dict = torch.load(dec_m_path, map_location=self.device)
                        m.load_state_dict(self.__strip_module_prefix(dec_m_i_state_dict))
                        logger.info(f"Loaded message decoder {i} from {dec_m_path}")
                    except Exception as e:
                        logger.error(f"Failed to load message decoder {i} from {dec_m_path}: {e}")
                        raise
        
                logger.info("All model components successfully loaded.")
        
            except Exception as e:
                logger.exception(f"Error occurred during model loading: {e}")
                raise
        else:
            raise ValueError("Checkpoint directory is None. Please provide a valid path to the checkpoint directory.")
    
    def incur_loss_test(self, 
                        mag_carrier: torch.Tensor, 
                        msg_gt: List[torch.Tensor], 
                        msg_reconst: Dict,
                        msg_compact: torch.Tensor):
        
        n_msg = len(msg_gt)
        loss_log = defaultdict(int)
        mag_carrier, msg_gt = mag_carrier.to(self.device), [msg_i.to(self.device) for msg_i in msg_gt]
        total_msg_loss = 0

        for i in range(n_msg):
            if msg_reconst['clean'][0][i].shape[3] == msg_gt[i].shape[3]:
                msg_loss = torch.nn.functional.cross_entropy(
                    msg_reconst['clean'][0][i].transpose(2, 3).reshape([-1, self.msg_dim]),
                    torch.argmax(msg_gt[i].transpose(2, 3), dim=3).reshape([-1]), reduction='none'
                )
            else:
                msg_loss = torch.zeros([1]).to(self.device)
                
            msg_loss = torch.mean(msg_loss)
            total_msg_loss += msg_loss
   
        loss_log['msg_l'] = total_msg_loss.item() / self.n_messages
        
        with torch.no_grad():
            for msg_reconst_key in msg_reconst:
                # Calculate the accuracy of the message reconstruction
                msg_pred = msg_reconst[msg_reconst_key][1][0]
                msg_target = torch.argmax(msg_compact, dim=3).squeeze(1)
                loss_log['hard/' + msg_reconst_key] = torch.mean(torch.all(msg_pred == msg_target, dim=1).float()).item()
                loss_log['bitwise/' + msg_reconst_key] = torch.mean((msg_pred == msg_target).float()).item()
           
        return loss_log

    def random_message(self, 
                       batch_size: int, 
                       sample_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        message = []
        message_compact = []

        if sample_len is not None:
            message_patch_len = math.ceil(sample_len / self.config.stft.hop_len) + 1
        else:
            message_patch_len = self.message_patch_len

        for _ in range(batch_size):
            batch_message = []
            batch_message_compact = []

            for _ in range(self.n_messages):
                # Generate random indices
                index = torch.cat((torch.randint(1, self.msg_dim, (self.msg_len - 1,)), torch.tensor([0])))

                # Create one-hot encoding
                one_hot = torch.eye(self.msg_dim)[index]
                batch_message_compact.append(one_hot)

                # Generate tiled or concatenated message based on self.message_patch_len
                if message_patch_len % self.msg_len == 0:
                    batch_message.append(one_hot.T.repeat(1, message_patch_len // self.msg_len))
                else:
                    tiled = one_hot.T.repeat(1, message_patch_len // self.msg_len)
                    concatenated = torch.cat([tiled, one_hot.T[:, :message_patch_len % self.msg_len]], dim=1)
                    batch_message.append(concatenated)

            # Stack messages for this batch
            batch_message = torch.stack(batch_message)
            batch_message_compact = torch.stack(batch_message_compact)

            message.append(batch_message)
            message_compact.append(batch_message_compact)

        # Stack across batches
        message = torch.stack(message).to(self.device)
        message_compact = torch.stack(message_compact).to(self.device)

        return message, message_compact

    def encode(self, 
               audio: torch.Tensor, 
               msg: List[float], 
               attack_types: List[str],
               attack_params: List[str]):     
           
        def decoder(encoded: torch.Tensor, 
                    carrier: torch.Tensor = None):
            """
            Parameters
            ----------
            encoded: torch.Tensor [shape=(B, 1, T)]
                Watermarked audio signal

            carrier: torch.Tensor [shape=(B, 1, M, T)]
                Magnitude spectrogram of the carrier signal
            """           
            carrier_reconst_tag, _ = self.stft.transform(encoded.squeeze(1))
            carrier_reconst_tag = carrier_reconst_tag.unsqueeze(1)

            # decode messages from carrier
            msg_reconst_list = []
            pred_msg_list = []
            
            for i in range(self.n_messages):  # decode each msg_i using decoder_m_i
                args = {'x': carrier_reconst_tag}
                msg_reconst = self.dec_m[i](**args)
                pred_message = self.calculate_aligned_message(msg_reconst)
                msg_reconst_list.append(msg_reconst.to(self.device))
                pred_msg_list.append(pred_message.to(self.device))

            return msg_reconst_list, pred_msg_list
        
        assert type(msg) == list  # type(carrier) == torch.Tensor and 

        start = time.time()
        audio, msg = audio.to(self.device), [msg_i.to(self.device) for msg_i in msg]

        mag_carrier, carrier_phase = self.stft.transform(audio.squeeze(1))
        mag_carrier = mag_carrier[:, None]
        carrier_phase = carrier_phase[:, None]

        # create embedded carrier
        carrier_enc = self.enc_c(mag_carrier)  # encode the carrier
        msg_enc = torch.cat(msg, dim=1)  # concat all msg_i into single tensor
        msg_enc = self.enc_c.transform_message(msg_enc)
        merged_enc = torch.cat((carrier_enc, 
                                mag_carrier.repeat(1, 32, 1, 1), 
                                msg_enc.repeat(1, 32, 1, 1)), dim=1)  # concat encodings on features axis

        message_info = self.dec_c(merged_enc, self.msg_sdr)
        
        # Utterance level normalization
        message_info = message_info*(torch.mean((mag_carrier**2), dim=(2,3), keepdim=True)**0.5)  # *time_weighing
        
        # Ensure negative message
        message_info = -message_info
        carrier_reconst = torch.nn.functional.relu(message_info + mag_carrier)  # decode carrier, output in stft domain

        y_wm = self.stft.inverse(carrier_reconst.squeeze(1), 
                                              carrier_phase.squeeze(1),
                                              audio.shape[-1])
        y_wm = y_wm[..., :audio.shape[-1]]

        mag_wm, _ = self.stft.transform(y_wm.squeeze(1))
        mag_wm = mag_wm[:, None]
        y_wm_cpu = y_wm.clone().detach()

        # Decode without distortion
        all_msg_reconst = {'clean': decoder(y_wm_cpu, carrier=mag_wm)}

        # Apply audio attacks, if available
        if attack_types is not None:
            y_wm_dirty = torch.zeros_like(y_wm)
            y_dirty = torch.zeros_like(y_wm)

            for b in range(y_wm_dirty.shape[0]):
                args = {} if attack_params[b] is None else json.loads(attack_params[b])
                
                if attack_types[b] == 'phase_shift':
                    # During test and validation, the phase shift parameter is in seconds.
                    args['shift'] = int(args['shift'] * self.sample_rate)   
            
                    # The computation here is in seconds, so we divide by the hop length to get the number of hops
                    if args['shift'] > self.stft.hop_len / 2:
                        msg_shift = int(args['shift'] / self.stft.hop_len + 0.5) # what is this?
                                     
                y_wm_dirty[b, ...] = self.audio_attack(y_wm_cpu[b, ...],
                                                       attack_type=attack_types[b],
                                                       **args)
                y_dirty[b, ...] = self.audio_attack(audio[b, ...],
                                                    attack_type=attack_types[b],
                                                    **args)

                                                            
            # now the each attack can handle device by itself
            y_wm_dirty = y_wm_dirty * ((AVERAGE_ENERGY_VCTK / torch.mean(y_wm_dirty**2, dim=2, keepdim=True))**0.5)
            all_msg_reconst['distorted'] = decoder(y_wm_dirty, carrier=mag_wm)
            all_msg_reconst['no_watermark_clean'] = decoder(audio, carrier=mag_carrier)
            all_msg_reconst['no_watermark_distorted'] = decoder(y_dirty, carrier=mag_carrier)

        return mag_carrier, all_msg_reconst, y_wm_cpu, y_dirty


    def eval(self, 
             epoch_num: int = None,
             write_to_disk: bool = True):
        
        if self.config.test_suffix is not None:
            csv_suffix = '_' + self.config.test_suffix 

        if self.audio_attack is not None:
            self.audio_attack.set_mode('test')

        self.eval_mode()
        res_list = []
        column_names = ['audio_filepath', 
                        'dataset', 
                        'attack_type', 
                        'attack_params', 
                        'chunk_index']
        
        sisnr_f = ScaleInvariantSignalNoiseRatio().to(self.device)        

        self.seed_everything(self.config.random_seed_for_eval)
        with torch.inference_mode():
            num_items = 0
            data = self.test_loader
            logger.info("Start evaluation.")
            for ret in qqdm(data):
                audio_chunks, audio_filepaths, datasets, att_types, attack_params, chunk_indices, start_times = ret
                audio_chunks *= torch.sqrt(AVERAGE_ENERGY_VCTK / torch.mean(audio_chunks**2))
                assert audio_chunks.shape[0]==1, 'batch size should be 1'
                msg, msg_compact = self.random_message(audio_chunks.shape[0], sample_len=audio_chunks.shape[-1])
                
                # feedforward and incur loss
                msg = [msg]

                # feedforward and suffer loss
                mag_carrier, all_msg_reconst, y_wm, y_dirty = self.encode(audio=audio_chunks,
                                                                          msg=msg, 
                                                                          attack_types=att_types,
                                                                          attack_params=attack_params)

                cur_losses_log = self.incur_loss_test(mag_carrier=mag_carrier, 
                                                      msg_gt=msg, 
                                                      msg_reconst=all_msg_reconst,
                                                      msg_compact=msg_compact)                   

                if self.config.full_perceptual:
                    perceptual_metrics = self.compute_perceptual_metrics(audio_filepath=audio_filepaths[0],
                                                                         start_time=start_times[0],
                                                                         audio_duration=self.config.dataset.eval_seg_duration,
                                                                         watermarked_audio=y_wm,
                                                                         distorted_audio=y_dirty)
                    cur_losses_log.update(perceptual_metrics)
                audio_chunks = audio_chunks.to(self.device)
                cur_losses_log['sisnr_wm'] =  sisnr_f(y_wm, audio_chunks).item()
                cur_losses_log['sisnr_attack'] =  sisnr_f(y_dirty, audio_chunks).item()

                log_dir = {f"{att_types[0]}/{key}": val for key, val in cur_losses_log.items()}
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
        df_result = pd.DataFrame(res_list, 
                                 columns=column_names)
        
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

        # TODO: Take average of cur_losses_log

        return df_result, cur_losses_log
    
    def calculate_aligned_message_weighted(self,
                                           msg_pred_rpt: torch.Tensor,
                                           weight: torch.Tensor, 
                                           end_token: int = 0) -> torch.Tensor:
        """_summary_
        
        Args:
            pred_repeat_msg (tensor): [B, T]
            message_len (int): Number of characters after which message is repeat
            end_token (int, optional): Defaults to 0.
        """
        msg_pred_rpt = msg_pred_rpt.cpu().squeeze(1)
        weight = weight.cpu().reshape([msg_pred_rpt.shape[0], -1])[:, None]
        remove_preds = int(msg_pred_rpt.shape[2] / self.msg_len) * self.msg_len
        msg_pred_rpt = msg_pred_rpt[:, :, 0:remove_preds]
        weight = weight[:, :, 0:remove_preds]
        pred_message = torch.argmax(torch.sum((msg_pred_rpt * weight).reshape([msg_pred_rpt.shape[0], msg_pred_rpt.shape[1], -1, msg_len]), dim=2), dim=1)
        for idx in range(len(pred_message)):
            end_token_pos = torch.where(pred_message[idx] == end_token)[0]
            if len(end_token_pos) == 0:
                continue
            end_token_pos = end_token_pos[0].item()
            if end_token_pos == self.msg_len - 1:
                continue
            pred_message[idx] = torch.cat([pred_message[idx][end_token_pos+1:], pred_message[idx][:end_token_pos+1]])

        return pred_message
    
    def calculate_aligned_message(self,
                                  msg_pred_rpt: torch.Tensor,
                                  end_token: int = 0) -> torch.Tensor:
        """_summary_
        
        Args:
            pred_repeat_msg (tensor): [B, T]
            message_len (int): Number of characters after which message is repeat
            end_token (int, optional): Defaults to 0.
        """
        msg_pred_rpt = msg_pred_rpt.cpu().squeeze(1)
        remove_preds = int(msg_pred_rpt.shape[2] / self.msg_len) * self.msg_len
        msg_pred_rpt = msg_pred_rpt[:, :, 0:remove_preds]
        pred_message = torch.argmax(torch.mean(msg_pred_rpt.reshape([msg_pred_rpt.shape[0], 
                                                                        msg_pred_rpt.shape[1], -1, self.msg_len]), dim=2), dim=1)
        
        for idx in range(len(pred_message)):
            end_token_pos = torch.where(pred_message[idx] == end_token)[0]
            if len(end_token_pos) == 0:
                continue
            end_token_pos = end_token_pos[0].item()
            
            if end_token_pos == self.msg_len - 1:
                continue
            pred_message[idx] = torch.cat([pred_message[idx][end_token_pos+1:], pred_message[idx][:end_token_pos+1]])
        
        return pred_message