import json
import os
import time
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig
from qqdm import qqdm
from torchmetrics.audio.snr import ScaleInvariantSignalNoiseRatio
from tqdm import tqdm
from wavmark import wm_add_util
from wavmark.models.hinet import Hinet

from .base import Solver
from ..utils import snr as compute_snr

class SolverWavMark(Solver):
    def __init__(
        self, 
        config: Union[Path, DictConfig]
    ):
        """
        Initialize the SolverWavMark class.

        Args:
            config: DictConfig
                Configuration object.
        """
        super(SolverWavMark, self).__init__(config)
        self.num_point = config.sample_rate
        self.n_fft = self.config.stft.n_fft
        self.hop_len = self.config.stft.hop_len
        self.window = torch.hann_window(config.stft.n_fft).to(self.device)
        self.message_len = config.message.len
        self.payload_bits = config.message.payload_bits
        self.decode_batch_size = config.decode_batch_size
        self.build_model()
        
        self.load_models(config.checkpoint)
        logger.info("models loaded")

        self.exp_logger.log_hparams(config)

        if self.audio_attack is not None:
            self.audio_attack.set_mode('test')
            
    def eval_mode(
        self
    ):
        """
        Set all the compotnets to evaluation mode.
        """
        self.hinet.eval()
        self.watermark_fc.eval()
        self.watermark_fc_back.eval()

    def build_model(
        self
    ):
        """
        Build and initialize the model components.
        """
        self.hinet = Hinet(num_layers=self.config.model.num_layers).to(self.device)
        self.watermark_fc = torch.nn.Linear(self.config.model.num_bit, 
                                            self.num_point).to(self.device)
        self.watermark_fc_back = torch.nn.Linear(self.num_point, 
                                                 self.config.model.num_bit).to(self.device)

    def eval(
        self,
        epoch_num: int = None,
        write_to_disk: bool = True
    ):
        """
        Evaluate the model on the test set.

        Args:
            epoch_num: int, optional
                Epoch number for saving results.
            write_to_disk: bool, optional
                Whether to write results to disk.

        Returns:
            Tuple[pd.DataFrame, dict]: Results DataFrame and last loss log.
        """   
        if self.config.test_suffix is not None:
            csv_suffix = '_' + self.config.test_suffix 
        self.eval_mode()
        logger.info("Start evaluation.")

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
                message = self.random_message(self.payload_bits, audio_chunks.shape[0])
                y = audio_chunks.to(self.device)
                message = message.to(self.device).to(torch.float32)

                for b in range(y.shape[0]):
                    cur_losses_log = {}
                    audio_chunk = y[b]
                    msg = message[b]
                    y_wm, _ = self.encode_watermark(audio_chunk.view(-1), 
                                                    msg, 
                                                    show_progress=False)
                    y_wm = y_wm.unsqueeze(0)

                    args = {} if attack_params[b] is None else json.loads(attack_params[b])
                    
                    if att_types[b] == 'phase_shift':
                        # During test and validation, the phase shift parameter is in seconds.
                        args['shift'] = int(args['shift'] * self.config.sample_rate)   
                    
                    
                    length = audio_chunk.shape[-1]
                    y_wm_dirty = self.audio_attack(y_wm, attack_type=att_types[b], **args)[..., :length].squeeze(0)
                    y_dirty = self.audio_attack(audio_chunk, attack_type=att_types[b], **args)[..., :length].squeeze(0)
                    
                    y_wm_dirty_decoded, _ = self.decode_watermark(y_wm_dirty.view(-1))
                    y_wm_decoded, _  = self.decode_watermark(y_wm.view(-1))
                    y_decoded, _ = self.decode_watermark(audio_chunk.view(-1))
                    y_dirty_dicoded, _ = self.decode_watermark(y_dirty.view(-1))

                    if y_wm_decoded is None:
                        cur_losses_log['bitwise/clean'] = 0.0 
                    else:
                        cur_losses_log['bitwise/clean'] = torch.mean(((y_wm_decoded == message[b]).float())).item()
                    if y_wm_dirty_decoded is None:
                        cur_losses_log['bitwise/distorted'] = 0.0
                    else:
                        cur_losses_log['bitwise/distorted'] = torch.mean(((y_wm_dirty_decoded == message[b]).float())).item()
                    if y_decoded is None:
                        cur_losses_log['bitwise/no_watermark_clean'] = 0.0
                    else:
                        cur_losses_log['bitwise/no_watermark_clean'] = torch.mean(((y_decoded == message[b]).float())).item()
                    if y_dirty_dicoded is None:
                        cur_losses_log['bitwise/no_watermark_distorted'] = 0.0
                    else:
                        cur_losses_log['bitwise/no_watermark_distorted'] = torch.mean(((y_dirty_dicoded == message[b]).float())).item()

                    hard_metics = {key.replace('bitwise/', 'hard/'): int(value == 1.0) for key, value in cur_losses_log.items()}
                    cur_losses_log.update(hard_metics)
                    
                    if self.config.full_perceptual:
                        perceptual_metrics = self.compute_perceptual_metrics(audio_filepath=audio_filepaths[b],
                                                                            start_time=start_times[b],
                                                                            audio_duration=self.config.dataset.eval_seg_duration,
                                                                            watermarked_audio=y_wm[b],
                                                                            distorted_audio=y_dirty[b])
                        cur_losses_log.update(perceptual_metrics)

                    cur_losses_log['sisnr_wm'] =  sisnr_f(y_wm, audio_chunk).item()
                    cur_losses_log['sisnr_attack'] =  sisnr_f(y_dirty, audio_chunk).item()

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

    def custom_stft(
        self,
        data: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the Short-Time Fourier Transform (STFT) of the input tensor.

        Args:
            data: torch.Tensor
                Input audio tensor.

        Returns:
            torch.Tensor: STFT result as real and imaginary parts.
        """
        # torch: return_complex=False is deprecDeprecated since version 2.0: return_complex=False is deprecated,
        # instead use return_complex=True Note that calling torch.view_as_real() on the output will recover the deprecated output format.
        while data.dim() > 2:
            data = data.squeeze(0)
        tmp = torch.stft(data, 
                         n_fft=self.n_fft, 
                         hop_length=self.hop_len, 
                         window=self.window, 
                         return_complex=True)
        tmp = torch.view_as_real(tmp)

        return tmp

    def custom_istft(
        self,
        signal_wmd_fft: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the inverse Short-Time Fourier Transform (ISTFT).

        Args:
            signal_wmd_fft: torch.Tensor
                STFT tensor (real and imaginary parts).

        Returns:
            torch.Tensor: Reconstructed audio signal.
        """
        # torch: return_complex=False is deprecDeprecated since version 2.0: return_complex=False is deprecated,
        # instead use return_complex=True Note that calling torch.view_as_real() on the output will recover the deprecated output format.
        return torch.istft(torch.view_as_complex(signal_wmd_fft), 
                           n_fft=self.n_fft, 
                           hop_length=self.hop_len, 
                           window=self.window, 
                           return_complex=False)

    def load_models(
        self,
        checkpoint: Union[Path, str]
    ):
        """
        Load model weights from a checkpoint directory.

        Args:
            checkpoint: Union[Path, str]
                Path to the checkpoint directory.
        """
        ckpt = torch.load(os.path.join(checkpoint), map_location=self.device)
        hinet_ckpt = {key[6:]: val for key, val in ckpt.items() if key.startswith('hinet.')}
        watermark_fc_ckpt = {key[13:]: val for key, val in ckpt.items() if key.startswith('watermark_fc.')}
        watermark_fc_back_ckpt = {key[18:]: val for key, val in ckpt.items() if key.startswith('watermark_fc_back.')}
        self.hinet.load_state_dict(hinet_ckpt)
        self.watermark_fc.load_state_dict(watermark_fc_ckpt)
        self.watermark_fc_back.load_state_dict(watermark_fc_back_ckpt)

    # based on https://github.com/wavmark/wavmark/blob/6ab3bf7ce0679e5b5cfeff3a62e8df9cd2024b37/src/wavmark/__init__.py#L23
    def encode_watermark(self,
                         signal, 
                         payload, 
                         pattern_bit_length: int = 16,
                         min_snr: float = 20.0, 
                         max_snr: float = 38.0, 
                         show_progress: bool = False):
        pattern_bit = wm_add_util.fix_pattern[0:pattern_bit_length]
        pattern_bit = torch.tensor(pattern_bit, dtype=int).to(self.device)
        watermark = torch.cat([pattern_bit, payload], dim=0).to(self.device)

        assert len(watermark) == self.message_len

        signal_wmd, info = self.add_watermark(watermark, 
                                              signal, 
                                              0.1,
                                              min_snr, 
                                              max_snr,
                                              show_progress=show_progress)
        
        return signal_wmd, info
    
    def encode(
        self,
        signal: torch.Tensor,
        message: torch.Tensor
    ):
        """
        Encode a message into the signal.

        Args:
            signal: torch.Tensor
                Input audio tensor.
            message: torch.Tensor
                Message tensor to encode.

        Returns:
            torch.Tensor: Watermarked audio tensor.
        """
        signal_fft = self.custom_stft(signal)
        # (batch,freq_bins,time_frames,2)

        message_expand = self.watermark_fc(message)
        message_fft = self.custom_stft(message_expand)
        signal_wmd_fft, _ = self.enc_dec(signal_fft, message_fft, rev=False)
        # (batch,freq_bins,time_frames,2)
        signal_wmd = self.custom_istft(signal_wmd_fft)

        return signal_wmd

    # based on https://github.com/wavmark/wavmark/blob/6ab3bf7ce0679e5b5cfeff3a62e8df9cd2024b37/src/wavmark/models/my_model.py#L44
    def decode(self, 
               signal: torch.Tensor):
        signal_fft = self.custom_stft(signal)
        watermark_fft = signal_fft
        _, message_restored_fft = self.enc_dec(signal_fft, watermark_fft, rev=True)
        message_restored_expanded = self.custom_istft(message_restored_fft)
        message_restored_float = self.watermark_fc_back(message_restored_expanded).clamp(-1, 1)
        
        return message_restored_float

    # based on https://github.com/wavmark/wavmark/blob/6ab3bf7ce0679e5b5cfeff3a62e8df9cd2024b37/src/wavmark/__init__.py#L37
    def decode_watermark(self, 
                         signal: torch.Tensor, 
                         len_start_bit: int = 16, 
                         show_progress: bool = False):
        start_bit = wm_add_util.fix_pattern[0:len_start_bit]
        mean_result, info = self.extract_watermark_v3_batch(signal, start_bit, 0.1, show_progress=show_progress)

        if mean_result is None:
            return None, info

        payload = mean_result[len_start_bit:]
        return payload, info

    # based on https://github.com/wavmark/wavmark/blob/6ab3bf7ce0679e5b5cfeff3a62e8df9cd2024b37/src/wavmark/models/my_model.py#L52
    def enc_dec(self, 
                signal: torch.Tensor, 
                watermark: torch.Tensor, 
                rev: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        signal = signal.permute(0, 3, 2, 1)
        watermark = watermark.permute(0, 3, 2, 1)
        signal2, watermark2 = self.hinet(signal, watermark, rev)

        return signal2.permute(0, 3, 2, 1), watermark2.permute(0, 3, 2, 1)

    # based on https://github.com/wavmark/wavmark/blob/6ab3bf7ce0679e5b5cfeff3a62e8df9cd2024b37/src/wavmark/utils/wm_add_util.py#L16
    def add_watermark(self,
                      bit_arr: torch.Tensor, 
                      data: torch.Tensor, 
                      shift_range: float, 
                      min_snr: float = None, 
                      max_snr: float = None, 
                      show_progress: bool = True):
        t1 = time.time()

        chunk_size = self.num_point + int(self.num_point * shift_range)
        num_segments = int(data.size(-1) / chunk_size)
        len_remain = data.size(-1) - num_segments * chunk_size

        output_chunks = []
        encoded_sections = 0
        skip_sections = 0

        the_iter = range(num_segments)
        if show_progress:
            the_iter = tqdm(the_iter, desc="Processing")

        for i in the_iter:
            start_point = i * chunk_size
            # current_chunk = data[start_point:start_point + chunk_size].copy()
            current_chunk = data[start_point:start_point + chunk_size].clone()
            # [watermark_segment | shift_area ]
            current_chunk_cover_area = current_chunk[0:self.num_point]
            current_chunk_shift_area = current_chunk[self.num_point:]
            current_chunk_cover_area_wmd, state = self.encode_trunck_with_snr_check(i, 
                                                                                    current_chunk_cover_area,
                                                                                    bit_arr,
                                                                                    min_snr, 
                                                                                    max_snr)

            if state == "skip":
                skip_sections += 1
            else:
                encoded_sections += 1

            output = torch.cat([current_chunk_cover_area_wmd.squeeze(0), current_chunk_shift_area], dim=0)
            assert output.shape == current_chunk.shape
            output_chunks.append(output)

        assert len(output_chunks) > 0
        if len_remain > 0:
            output_chunks.append(data[len(data) - len_remain:])

        if isinstance(output_chunks[0], list) or isinstance(output_chunks[0], torch.Tensor):
            # Convert each chunk into a tensor and concatenate
            # output_chunks_tensors = [torch.tensor(chunk, dtype=torch.float32) for chunk in output_chunks]
            reconstructed_array = torch.cat(output_chunks, dim=0).to(self.device)
        else:
            # If output_chunks is already a flat list
            reconstructed_array = torch.tensor(output_chunks, dtype=torch.float32).to(self.device)

        time_cost = time.time() - t1

        info = {
            "time_cost": time_cost,
            "encoded_sections": encoded_sections,
            "skip_sections": skip_sections,
        }
        return reconstructed_array, info

    # based on https://github.com/wavmark/wavmark/blob/6ab3bf7ce0679e5b5cfeff3a62e8df9cd2024b37/src/wavmark/utils/wm_add_util.py#L66
    def encode_trunck_with_snr_check(self,
                                     idx_trunck, 
                                     signal, 
                                     wm, 
                                     min_snr,
                                     max_snr: float) -> Tuple[torch.Tensor, int]:
        signal_for_encode = signal
        encode_times = 0
        while True:
            encode_times += 1
            signal_wmd = self.encode_trunck(signal_for_encode, wm)
            snr = compute_snr(signal.unsqueeze(0), signal_wmd.unsqueeze(0)).item()
            if encode_times == 1 and snr < min_snr:
                print("skip section:%d, snr too low:%.1f" % (idx_trunck, min_snr))
                return signal, "skip"

            if snr < max_snr:
                return signal_wmd, encode_times
            # snr is too hugh
            signal_for_encode = signal_wmd

            if encode_times > 10:
                return signal_wmd, encode_times

    # based on https://github.com/wavmark/wavmark/blob/6ab3bf7ce0679e5b5cfeff3a62e8df9cd2024b37/src/wavmark/utils/wm_add_util.py#L86
    def encode_trunck(self,
                      signal: torch.Tensor, 
                      wm: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            signal_wmd = self.encode(signal[None], wm[None])
            
            return signal_wmd
    
    # based on https://github.com/wavmark/wavmark/blob/6ab3bf7ce0679e5b5cfeff3a62e8df9cd2024b37/src/wavmark/utils/wm_decode_util.py#L17
    def extract_watermark_v3_batch(self,
                                   data, 
                                   start_bit, 
                                   shift_range,
                                   shift_range_p=0.5,
                                   show_progress=False):
        assert type(show_progress) == bool
        start_time = time.time()
        # 1.determine the shift step length:
        shift_step = int(shift_range * self.num_point * shift_range_p)

        # 2.determine where to perform detection
        # pdb.set_trace()
        total_detections = (len(data) - self.num_point) // shift_step
        total_detect_points = [i * shift_step for i in range(total_detections)]

        # 3.construct batch for detection
        total_batch_counts = len(total_detect_points) // self.decode_batch_size + 1
        results = []

        the_iter = range(total_batch_counts)
        if show_progress:
            the_iter = tqdm(range(total_batch_counts))

        for i in the_iter:
            detect_points = total_detect_points[i * self.decode_batch_size:i * self.decode_batch_size + self.decode_batch_size]
            if len(detect_points) == 0:
                break
            # current_batch = np.array([data[p:p + self.num_point] for p in detect_points])
            current_batch = torch.stack([data[p:p + self.num_point] for p in detect_points])
            with torch.no_grad():
                # signal = torch.FloatTensor(current_batch).to(self.device)
                batch_message = (self.decode(current_batch) >= 0.5).int().detach().cpu().numpy()
                for p, bit_array in zip(detect_points, batch_message):
                    decoded_start_bit = bit_array[0:len(start_bit)]
                    ber_start_bit = 1 - np.mean(start_bit == decoded_start_bit)
                    num_equal_bits = np.sum(start_bit == decoded_start_bit)
                    if ber_start_bit > 0:  # exact match
                        continue
                    results.append({
                        "sim": 1 - ber_start_bit,
                        "num_equal_bits": num_equal_bits,
                        "msg": bit_array,
                        "start_position": p,
                        "start_time_position": p / self.config.sample_rate
                    })

        end_time = time.time()
        time_cost = end_time - start_time

        info = {
            "time_cost": time_cost,
            "results": results,
        }

        if len(results) == 0:
            return None, info

        results_1 = [i["msg"] for i in results if np.isclose(i["sim"], 1.0)]
        mean_result = (torch.tensor(results_1, dtype=torch.float32).mean(dim=0) >= 0.5).int().to(self.device)

        return mean_result, info

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
        
        return torch.randint(0, 2, (batch_size, nbits))
