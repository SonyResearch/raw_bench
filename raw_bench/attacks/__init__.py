import os
from omegaconf import DictConfig
from loguru import logger
import pandas as pd
import torch
from torch import nn
from typing import List, Optional, Tuple, Union
import torchaudio
import torchaudio.functional as F

from grafx.processors import GraphicEqualizer

from .compression import aac_wrapper, mp3_wrapper, vorbis_wrapper
from .dynamics import dynamic_range_compression, dynamic_range_expansion
from .filtering import highpass_filter, lowpass_filter
from .low_level import inverse_polarity, phase_shift, quantize, time_jitter, time_stretch_wrapper
from .neural_codecs import DACWrapper, EncodecWrapper
from .noise import gaussian_noise
from .specaugment import freq_mask, time_mask
from .utils import ste, choose_random_uniform_val, sample_from_intervals


class AudioAttack(nn.Module):
    """
    Implements a variety of audio attacks and augmentations for robustness evaluation.

    Args:
        sr: int
            Sample rate of the audio.
        datapath: dict
            Dictionary of dataset paths.
        stft: nn.Module
            STFT module for spectrogram-based attacks.
        mode: str
            Mode of operation ('train', 'test', or 'val').
        config: DictConf
            Configuration for attack parameters and settings.
        ffmpeg4codecs: str
            Path to ffmpeg binary for codecs.
        mixing_train_filepath: str
            Path to CSV file for mixing noises.
        reverb_train_filepath: str
            Path to CSV file for reverb IRs.
        delimiter: str
            Delimiter for CSV files.
        single_attack: bool
            Whether to apply a single attack or composite attacks. currently only single attacks are supported.
        device: str
            Device to run computations on.
    """
    def __init__(
        self,
        sr: int = 44100,
        datapath: dict = None,
        stft: nn.Module = None,
        mode: str = 'test',
        config: DictConfig = None,
        ffmpeg4codecs: Optional[str] = None,
        mixing_train_filepath: Optional[str] = None,
        reverb_train_filepath: Optional[str] = None,
        delimiter: str = '|',
        single_attack: bool = True,
        device: str = 'cuda'
    ):
        """
        Initialize AudioAttack with configuration and resources.
        """
        super().__init__()
        self.config = config
        self.datapath = datapath
        self.sr = sr
        self.stft = stft
        self.mode = mode
        self.single_attack = single_attack
        self.device = device
        self.ffmpeg4codecs = ffmpeg4codecs
        if self.ffmpeg4codecs is None:
            logger.warning(
                "ffmpeg4codecs is not provided. "
                "Codec attacks may not work properly if your default ffmpeg "
                "does not support all of them."
            )
        else:
            if not os.path.exists(self.ffmpeg4codecs):
                raise FileNotFoundError(f"ffmpeg binary not found at {self.ffmpeg4codecs}.")

        # Initialize the mixing file for training
        if self.mode == 'train':
            self.df_train_mixing = pd.read_csv(mixing_train_filepath, 
                                               delimiter=delimiter)
        else:
            self.df_train_mixing = None

        # Initialize the reverb impulse response (RIR) file for training
        if self.mode == 'train':
            self.df_train_rir = pd.read_csv(reverb_train_filepath, 
                                            delimiter=delimiter)
        else:
            self.df_train_rir = None

        # Initialize the graphic equalizer (GEQ) based on second-order peaking filters
        self._init_graphic_eq()

        # Initialize neural codecs (Encodec and DAC)
        self._init_neural_codecs()

        # Initialize the dictionary of attacks
        self._init_dict_attacks()
        self.num_attacks = len(self.dict_attacks)

        # Initialize the attack probabilities for each attack
        self._init_attack_probabilities()

    def _init_graphic_eq(self):
        """
        Initialize the graphic equalizer and resamplers for equalization attacks.
        """
        self.eq = GraphicEqualizer(sr=self.config.eq.sr,
                                   scale=self.config.eq.scale).to(self.device)
        
        if self.sr != self.config.eq.sr:
            self.model_sr2eq_sr= torchaudio.transforms.Resample(orig_freq=self.sr, 
                                                                new_freq=self.config.eq.sr).to(self.device)
            
            self.eq_sr2model_sr = torchaudio.transforms.Resample(orig_freq=self.config.eq.sr,
                                                                 new_freq=self.sr).to(self.device)                                                
        else:
            self.eq_resampler = None

        if self.config.eq.scale == 'bark':
            self.eq.num_bands = 24
        elif self.config.eq.scale == 'third_oct':
            self.eq.num_bands = 31
        else:
            raise ValueError('scale should be one of bark or third_oct.')

    def _init_neural_codecs(self):
        """
        Initialize neural audio codecs (Encodec and DAC) for compression attacks.
        """
        self.encodec = EncodecWrapper(
            codec_type=self.config.encodec.type,
            model_sr=self.sr,
            device=self.device
        )
        self.dac = DACWrapper(
            codec_type=self.config.dac.type,
            device=self.device
        )

    def _init_dict_attacks(self):
        """
        Build a dictionary mapping attack names to their corresponding methods.
        """
        self.dict_attacks = dict()
        for attack_dict in self.config.attacks:
            attack_type = list(attack_dict.keys())[0]
            func_attack = getattr(self, f"apply_{attack_type}")
            self.dict_attacks[attack_type] = func_attack

    def _init_attack_probabilities(self):
        """
        Initialize the probability for selecting each attack.
        """
        attack_weights = {key: value for d in self.config.attacks for key, value in d.items()}
        total_weight = sum(attack_weights.values())
        self.attack_probs = {k: v / total_weight for k, v in attack_weights.items()}

    def set_mode(
        self,
        mode: str
    ):
        """
        Set the mode of operation for the attack module.

        Args:
            mode: str
                Mode to set ('train', 'test', or 'val').
        """
        self.mode = mode

    def forward(
        self,
        audio: torch.Tensor,
        attack_type: Optional[str] = None,
        return_attack_params: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, str, dict]]:
        """
        Apply a randomly selected or specified attack to the input audio.

        Args:
            audio: torch.Tensor
                Input audio tensor.
            attack_type: str. optional
                Specific attack to apply.
            return_attack_params: bool
                Whether to return attack parameters.

        Returns:
            torch.Tensor or (torch.Tensor, str, dict): Distorted audio, and optionally attack type and parameters.
        """
        # Check if the shape of the audio tensor is valid [B, C, T]
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)

        if audio.shape[1] > 1:
            raise NotImplementedError('Only mono audio is supported.')

        if self.single_attack:
            if attack_type is None:
                list_attack_names = list(self.attack_probs.keys())
                list_attack_probs = [self.attack_probs[attack] for attack in list_attack_names]      
                attack_type = list_attack_names[torch.multinomial(
                    torch.tensor(list_attack_probs, dtype=torch.float), num_samples=1).item()]
                attack_fn = self.dict_attacks[attack_type]
            else:
                attack_fn = self.dict_attacks[attack_type]
        else:
            raise NotImplementedError('Composite attacks are not yet implemented.')

        # Spectrogram-based attacks
        if attack_type in ['freq_mask', 'time_mask']:
            distorted_audio, attack_type, attack_params = self.apply_specaugment(audio=audio, 
                                                                                 attack_type=attack_type, 
                                                                                 attack_fn=attack_fn, 
                                                                                 **kwargs)            
        else:
            distorted_audio, attack_type, attack_params = attack_fn(audio, **kwargs)
    
        # Normalize the audio, if needed, except for gain attacks
        if distorted_audio.abs().max() > 1 and attack_type != 'gain':
            distorted_audio = distorted_audio / distorted_audio.abs().max()

        if return_attack_params:
            return distorted_audio, attack_type, attack_params
        else:
            return distorted_audio

    def apply_specaugment(
        self,
        audio: torch.Tensor,
        attack_type: Optional[str] = None,
        attack_fn: callable = None,
        **kwargs
    ) -> Tuple[torch.Tensor, str, dict]:
        """
        Apply SpecAugment-based augmentations to the input audio tensor. This will apply
        1) STFT on the input audio waveform
        2) The attack onto the magnitude of the STFT
        3) Inverse STFT to get the distorted audio waveform.

        Args:
            audio: torch.Tensor
                Input audio tensor.
            attack_type: str
                'time_mask' or 'freq_mask'.
            attack_fn: callable
                Attack function to apply.

        Returns:
            Tuple[torch.Tensor, str, dict]: Distorted audio, attack type, and parameters.
        """
        if attack_type not in ['time_mask', 'freq_mask']:
            raise ValueError('attack_type should be one of time_mask, freq_mask')
        
        mag, phase = self.stft.transform(audio.squeeze(0))
        modified_mag, _, attack_params = attack_fn(mag, **kwargs)
        distorted_audio = self.stft.inverse(magnitude=modified_mag, 
                                            phase=phase)

        return distorted_audio, attack_type, attack_params

    def apply_gaussian_noise(
        self,
        x: torch.Tensor,
        snr: Optional[float] = None
    ) -> Tuple[torch.Tensor, str, dict]:
        """
        Add Gaussian noise to the input audio at a specified SNR.

        Args:
            x: torch.Tensor
                Input audio tensor.
            snr: float, optional (only for training mode)
                Signal-to-noise ratio.

        Returns:
            Tuple[torch.Tensor, str, dict]: 
                Distorted audio, attack type, and parameters.
        """
        if snr is None and self.mode == 'train':
            snr = choose_random_uniform_val(min_val=self.config.gaussian_noise.min_snr,
                                            max_val=self.config.gaussian_noise.max_snr,
                                            num_samples=1)
                                 
        elif snr is None and self.mode in ['test', 'val']:
            raise ValueError('snr should be provided in val and test modes.')

        noise = gaussian_noise(x, 
                               std=self.config.gaussian_noise.std)    
        
        snr = torch.Tensor([snr]).to(x.device).unsqueeze(0)

        return F.add_noise(x, noise=noise, snr=snr), 'gaussian_noise', {'snr': snr.item()}


    # Mixing with background noise
    def apply_background_noise(
        self, 
        x: torch.Tensor,
        snr: Optional[float] = None,
        noise_filepath: Optional[str] = None,
        noise_sr: Optional[int] = None
        ) -> Tuple[torch.Tensor, str, dict]:
        """
        Apply background noise to the input audio tensor at a specified SNR.

        Args:
            x: torch.Tensor
                Input audio tensor.
            snr: float, optional (only for training mode)
                Signal-to-noise ratio.
            noise_filepath: str, optional
                Filepath to the noise audio file.
            noise_sr: int, optional
                Sample rate of the noise audio.

        Returns:
            Tuple[torch.Tensor, str, dict]: Distorted audio, attack type, and parameters.
        """
        if self.mode == 'train' and snr is None:
            snr = choose_random_uniform_val(min_val=self.config.background_noise.min_snr,
                                            max_val=self.config.background_noise.max_snr,
                                            num_samples=1)
        
        elif self.mode in ['test', 'val'] and (snr is None or noise_filepath is None or noise_sr is None):
            raise ValueError('snr and noise_filepath should be provided in val and test modes.')

        snr = torch.Tensor([snr]).to(x.device).unsqueeze(0)

        audio_len = x.shape[-1]

        if noise_filepath is None:
            random_row = self.df_train_mixing.sample(n=1).iloc[0]
            noise_filepath = random_row['audio_filepath']
            noise_duration = random_row['duration']
            noise_sr = random_row['sample_rate']
            start_idx = int(torch.floor(torch.rand(1) * (int(noise_duration * noise_sr) - audio_len * (noise_sr / self.sr))).item()) 
            noise, orig_sr = torchaudio.load(f"{self.datapath['DEMAND']}/{noise_filepath}", 
                                             frame_offset=start_idx, 
                                             num_frames=int(audio_len*(noise_sr/self.sr)))          
                            
            if orig_sr != self.sr:
                noise = torchaudio.transforms.Resample(orig_freq=orig_sr, 
                                                       new_freq=self.sr)(noise)
        else:
            noise, orig_sr = torchaudio.load(f"{self.datapath['DEMAND']}/{noise_filepath}",
                                             num_frames=int(audio_len*(noise_sr/self.sr)))    
            if orig_sr != self.sr:
                noise = torchaudio.transforms.Resample(orig_freq=orig_sr, 
                                                       new_freq=self.sr)(noise)
        if noise_sr != orig_sr:
            assert ValueError(f'noise sample rate is incorrectly given as {noise_sr}, its correct value is {orig_sr}')

        return F.add_noise(x, 
                           noise=noise.unsqueeze(0).to(x.device), 
                           snr=snr), 'background_noise', {'snr': snr.item()}

    # Mixing the dry audio with reverberant (wet) audio
    def apply_reverb(
        self,
        x: torch.Tensor,
        rir_filepath: Optional[str] = None,
        snr: Optional[float] = None
        ) -> Tuple[torch.Tensor, str, dict]:
        """
        Apply reverberation (convolution with a room impulse response) to the input audio tensor.

        Args:
            x: torch.Tensor
                Input audio tensor.
            rir_filepath: str, optional
                Filepath to the room impulse response (RIR) audio file.
            snr: float, optional
                Signal-to-noise ratio for mixing dry and reverberant audio.

        Returns:
            Tuple[torch.Tensor, str, dict]: Distorted audio, attack type, and parameters.
        """
        while x.dim() > 2:
            try:
                x = x.squeeze(0)
            except:
                raise ValueError('The input audio tensor is invalid.')
        
        if (rir_filepath is None or snr is None) and self.mode == 'train':
            random_row = self.df_train_rir.sample(n=1).iloc[0]
            rir_filepath = random_row['audio_filepath']
            snr = choose_random_uniform_val(min_val=self.config.reverb.min_snr,
                                            max_val=self.config.reverb.max_snr,
                                            num_samples=1)
            
        elif (rir_filepath is None or snr is None) and self.mode in ['test', 'val']:
            raise ValueError('rir_filepath and snr should be provided in val and test modes.')

        snr = torch.Tensor([snr]).to(x.device)

        _rir_filepath = f"{self.datapath['AIR']}/{rir_filepath}"
        rir, orig_sr = torchaudio.load(_rir_filepath)
        if orig_sr != self.sr:
            rir = torchaudio.transforms.Resample(orig_freq=orig_sr, 
                                                 new_freq=self.sr)(rir)
        rir_rms_norm = rir / torch.linalg.vector_norm(rir, ord=2)
        rir_rms_norm = rir_rms_norm.to(x.device)

        # TODO: Not the best solution
        if rir_rms_norm.size(1) > self.sr * self.config.reverb.max_rir_len:
            rir_rms_norm = rir_rms_norm[:, :int(self.sr * self.config.reverb.max_rir_len)]

        audio_with_reverb = torch.zeros_like(x, device=x.device)

        # If the audio has multiple channels, apply reverb to each channel separately
        for channel in range(x.shape[0]):
            tmp = F.fftconvolve(x[channel, :].unsqueeze(0), rir_rms_norm)
            # Truncate the audio with reverb if it is longer than the original audio
            if tmp.shape[-1] > x.shape[-1]:
                tmp = tmp[:, :x.shape[-1]]
            audio_with_reverb[channel, :tmp.shape[-1]] = tmp.clone()

        return F.add_noise(x, noise=audio_with_reverb, snr=snr).unsqueeze(0), 'reverb', {'snr': snr.item(),
                                                                                         'rir_filepath': rir_filepath}

    # Compression (MP3, AAC, Vorbis)
    def apply_mp3(
        self, 
        x: torch.Tensor,
        bitrate: Optional[str] = None,
        ffmpeg4codecs: Optional[str] = None
        ) -> Tuple[torch.Tensor, str, dict]: 
        """
        Apply MP3 compression to the input audio tensor.

        Args:
            x: torch.Tensor
                Input audio tensor.
            bitrate: str, optional
                Bitrate for MP3 compression.
            ffmpeg4codecs: str, optional
                Path to ffmpeg binary for codecs.

        Returns:
            Tuple[torch.Tensor, str, dict]: Distorted audio, attack type, and parameters.
        """
        if self.mode == 'train' and bitrate is None:
            bitrate = self.config.mp3[
                torch.randint(0, len(self.config.mp3), (1,)).item()
            ]
        elif self.mode in ['test', 'val'] and bitrate is None:
            raise ValueError('bitrate should be provided in val and test modes.')
        
        if ffmpeg4codecs is None:
            ffmpeg4codecs = self.ffmpeg4codecs
                    
        return ste(original=x, 
                   compressed=mp3_wrapper(x, bitrate=bitrate, ffmpeg4codecs=ffmpeg4codecs)), 'mp3', {'bitrate': bitrate}

    def apply_vorbis(
        self, 
        x: torch.Tensor,
        bitrate: Optional[str] = None,
        ffmpeg4codecs: Optional[str] = None
        ) -> Tuple[torch.Tensor, str, dict]:
        """
        Apply Vorbis compression to the input audio tensor.

        Args:
            x: torch.Tensor
                Input audio tensor.
            bitrate: str, optional
                Bitrate for Vorbis compression.
            ffmpeg4codecs: str, optional
                Path to ffmpeg binary for codecs.

        Returns:
            Tuple[torch.Tensor, str, dict]: Distorted audio, attack type, and parameters.
        """        
        if self.mode == 'train' and bitrate is None:
            bitrate = self.config.vorbis[
                torch.randint(0, len(self.config.vorbis), (1,)).item()
            ]
        elif self.mode in ['test', 'val'] and bitrate is None:
            raise ValueError('bitrate should be provided in val and test modes.')
            
        if ffmpeg4codecs is None:
            ffmpeg4codecs = self.ffmpeg4codecs

        return ste(original=x, 
                   compressed=vorbis_wrapper(x, sr=self.sr, bitrate=bitrate, ffmpeg4codecs=ffmpeg4codecs)), 'vorbis', {'bitrate': bitrate}

    def apply_aac(
        self, 
        x: torch.Tensor,
        bitrate: Optional[str] = None,
        ffmpeg4codecs: Optional[str] = None
        ) -> Tuple[torch.Tensor, str, dict]:
        """
        Apply AAC compression to the input audio tensor.

        Args:
            x: torch.Tensor
                Input audio tensor.
            bitrate: str, optional
                Bitrate for AAC compression.
            ffmpeg4codecs: str, optional
                Path to ffmpeg binary for codecs.

        Returns:
            Tuple[torch.Tensor, str, dict]: Distorted audio, attack type, and parameters.
        """        
        if self.mode == 'train' and bitrate is None:
            bitrate = self.config.aac[
                torch.randint(0, len(self.config.aac), (1,)).item()
            ]
        elif self.mode in ['test', 'val'] and bitrate is None:
            raise ValueError('bitrate should be provided in val and test modes.')
            
        if ffmpeg4codecs is None:
            ffmpeg4codecs = self.ffmpeg4codecs

        return ste(original=x, 
                   compressed=aac_wrapper(x, sr=self.sr, bitrate=bitrate, ffmpeg4codecs=ffmpeg4codecs)), 'aac', {'bitrate': bitrate}
    
    # Filtering
    def apply_lowpass(
        self,
        x: torch.Tensor,
        cutoff: Optional[float] = None
    ) -> Tuple[torch.Tensor, str, dict]:
        """
        Apply a lowpass filter to the input audio tensor.

        Args:
            x: torch.Tensor
                Input audio tensor.
            cutoff: float, optional
                Cutoff frequency for the lowpass filter.

        Returns:
            Tuple[torch.Tensor, str, dict]:
                Filtered audio, attack type, and parameters.
        """
        if self.mode == 'train' and cutoff is None:
            cutoff = choose_random_uniform_val(min_val=self.config.lowpass.min_cutoff,
                                               max_val=self.config.lowpass.max_cutoff,
                                               num_samples=1)

        elif self.mode in ['test', 'val'] and cutoff is None:
            raise ValueError('cutoff should be provided in val and test modes.')

        return lowpass_filter(x, 
                              cutoff_freq=cutoff,
                              sr=self.sr), 'lowpass', {'cutoff': cutoff}

    def apply_highpass(
        self,
        x: torch.Tensor,
        cutoff: Optional[float] = None
    ) -> Tuple[torch.Tensor, str, dict]:
        """
        Apply a highpass filter to the input audio tensor.

        Args:
            x: torch.Tensor
                Input audio tensor.
            cutoff: float, optional
                Cutoff frequency for the highpass filter.

        Returns:
            Tuple[torch.Tensor, str, dict]:
                Filtered audio, attack type, and parameters.
        """
        if self.mode == 'train' and cutoff is None:
            cutoff = choose_random_uniform_val(min_val=self.config.highpass.min_cutoff,
                                               max_val=self.config.highpass.max_cutoff,
                                               num_samples=1)

        elif self.mode in ['test', 'val'] and cutoff is None:
            raise ValueError('cutoff should be provided in val and test modes.')

        return highpass_filter(x, 
                               cutoff_freq=cutoff,
                               sr=self.sr), 'highpass', {'cutoff': cutoff}

    def apply_eq(self,
                 x: torch.Tensor,
                 log_band_gains: Optional[List] = None) -> Tuple[torch.Tensor, str, dict]:
        """
        Apply a graphic equalizer to the input audio tensor based on grafx.

        Args:
            x: torch.Tensor
                Input audio tensor.
            log_band_gains: List, optional
                List of log band gains for the equalizer.

        Returns:
            Tuple[torch.Tensor, str, dict]:
                Equalized audio, attack type, and parameters.
        """
        
        length = x.shape[-1]
        if self.mode == 'train' and log_band_gains is None:
            log_band_gains = sample_from_intervals(intervals=self.config.eq.threshold_intervals, 
                                                   num_samples=self.eq.num_bands)

        
        elif self.mode in ['test', 'val'] and log_band_gains is None:
            raise ValueError('log_band_gains should be provided in val and test modes.')

        log_band_gains = torch.Tensor(log_band_gains).to(self.device)

        if self.sr != self.config.eq.sr:
            x_eq = self.model_sr2eq_sr(x)
        else:
            x_eq = x

        distorted = self.eq(x_eq, 
                            log_gains=log_band_gains)
        
        if self.sr != self.config.eq.sr:
            distorted = self.eq_sr2model_sr(distorted)
        
        if length < distorted.shape[-1]:
            distorted = distorted[..., :length]
        elif length > distorted.shape[-1]:
            distorted = torch.cat([distorted, torch.zeros_like(x, device=x.device)], dim=-1)
        
        return distorted, 'eq', {'log_band_gains': log_band_gains}

    # Neural compression
    def apply_encodec(
        self,
        x: torch.Tensor,
        n_codebook: Optional[int] = None
        ) -> Tuple[torch.Tensor, str, dict]:
        """
        Apply Encodec neural codec compression to the input audio tensor.

        Args:
            x: torch.Tensor
                Input audio tensor.
            n_codebook: int, optional
                Number of codebooks to use.

        Returns:
            Tuple[torch.Tensor, str, dict]:
                Compressed audio, attack type, and parameters.
        """
        if self.mode == 'train' and n_codebook is None:
            n_codebook = self.config.encodec.n_codebooks[
                torch.randint(0, len(self.config.encodec.n_codebooks), (1,)).item()
                ]
            
        elif self.mode in ['test', 'val'] and n_codebook is None:
            raise ValueError('n_codebook should be provided in val and test modes.')

        return self.encodec(x, target_n_codebook=n_codebook), 'encodec', {'n_codebook': n_codebook}

    def apply_dac(
        self,
        x: torch.Tensor,
        n_codebook: Optional[int] = None
        ) -> Tuple[torch.Tensor, str, dict]:
        """
        Apply DAC neural codec compression to the input audio tensor.

        Args:
            x: torch.Tensor
                Input audio tensor.
            n_codebook: int, optional
                Number of codebooks to use.

        Returns:
            Tuple[torch.Tensor, str, dict]:
                Compressed audio, attack type, and parameters.
        """
        if self.mode == 'train' and n_codebook is None:
            n_codebook = self.config.dac.n_codebooks[
                torch.randint(0, len(self.config.dac.n_codebooks), (1,)).item()
            ]

        elif self.mode in ['test', 'val'] and n_codebook is None:
            raise ValueError('n_codebook should be provided in val and test modes.')

        return self.dac(x, target_n_codebook=n_codebook), 'dac', {'n_codebook': n_codebook}

    # Dynamic
    def apply_dynamic_range_compression(
        self,
        x: torch.Tensor,
        threshold: Optional[float] = None,
        **kwargs
        ) -> Tuple[torch.Tensor, str, dict]:
        """
        Apply dynamic range compression to the input audio tensor.

        Args:
            x: torch.Tensor
                Input audio tensor.
            threshold: float, optional
                Compression threshold.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, str, dict]:
                Compressed audio, attack type, and parameters.
        """
        if self.mode == 'train' and threshold is None:
            threshold = choose_random_uniform_val(min_val=self.config.dynamic_range_compression.min_threshold,
                                                  max_val=self.config.dynamic_range_compression.max_threshold,
                                                  num_samples=1)
       
        elif self.mode in ['test', 'val'] and threshold is None:
            raise ValueError('threshold should be provided in val and test modes.')
        
        return dynamic_range_compression(x, 
                                         threshold=threshold,
                                         **kwargs), 'dynamic_range_compression', {'threshold': threshold, **kwargs}
    
    def apply_dynamic_range_expansion(
        self,
        x: torch.Tensor,
        threshold: Optional[float] = None,
        **kwargs
        ) -> Tuple[torch.Tensor, str, dict]:
        """
        Apply dynamic range expansion to the input audio tensor.

        Args:
            x: torch.Tensor
                Input audio tensor.
            threshold: float, optional
                Expansion threshold.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, str, dict]:
                Expanded audio, attack type, and parameters.
        """
        if self.mode == 'train' and threshold is None:
            threshold = choose_random_uniform_val(min_val=self.config.dynamic_range_expansion.min_threshold,
                                                  max_val=self.config.dynamic_range_expansion.max_threshold,
                                                  num_samples=1)
       
        elif self.mode in ['test', 'val'] and threshold is None:
            raise ValueError('threshold should be provided in val and test modes.')
        
        return dynamic_range_expansion(x,
                                       threshold=threshold,
                                       sr=self.sr,
                                       **kwargs), 'dynamic_range_expansion', {'threshold': threshold, **kwargs}
    
    def apply_limiter(self,
                      x: torch.Tensor,
                      threshold: Optional[float] = None,
                      **kwargs) -> Tuple[torch.Tensor, str, dict]:
        """
        Apply a limiter to the input audio tensor.

        Args:
            x: torch.Tensor
                Input audio tensor.
            threshold: float, optional
                Limiter threshold.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, str, dict]:
                Limited audio, attack type, and parameters.
        """
        if self.mode == 'train' and threshold is None:
            threshold = choose_random_uniform_val(min_val=self.config.limiter.min_threshold,
                                                  max_val=self.config.limiter.max_threshold,
                                                  num_samples=1)

        elif self.mode in ['test', 'val'] and threshold is None:
            raise ValueError('threshold should be provided in val and test modes.')

        return dynamic_range_compression(x,
                                         threshold=threshold,
                                         ratio=torch.inf,
                                         sr=self.sr,
                                         **kwargs), 'limiter', {'threshold': threshold, **kwargs}

    # Low level augmentations
    def apply_time_jitter(
        self,
        x: torch.Tensor,
        scale: Optional[float] = None
        ) -> Tuple[torch.Tensor, str, dict]:
        """
        Apply time jitter to the input audio tensor.

        Args:
            x: torch.Tensor
                Input audio tensor.
            scale: float, optional
                Jitter scale.

        Returns:
            Tuple[torch.Tensor, str, dict]:
                Jittered audio, attack type, and parameters.
        """
        if self.mode == 'train' and scale is None:
            scale = choose_random_uniform_val(min_val=self.config.time_jitter.min_scale,
                                              max_val=self.config.time_jitter.max_scale,
                                              num_samples=1)
        
        elif self.mode in ['test', 'val'] and scale is None:
            raise ValueError('scale should be provided in val and test modes.')
        
        # This is a bit tricky, because the interpolation function used for time_jitter causes issues
        # during backpropogation. So, we use torch.inference_mode() to avoid this.
        with torch.inference_mode():
            return time_jitter(x, scale=scale), 'time_jitter', {'scale': scale}
    
    def apply_polarity(
        self, 
        x: torch.Tensor
        ) -> Tuple[torch.Tensor, str, dict]:
        """
        Invert the polarity by multiplying -1 to input audio tensor.

        Args:
            x: torch.Tensor
                Input audio tensor.

        Returns:
            Tuple[torch.Tensor, str, dict]:
                Polarity-inverted audio, attack type, and parameters.
        """
        return inverse_polarity(x), 'polarity', {}
    
    def apply_gain(
        self, 
        x: torch.Tensor,
        rate: Optional[float] = None
        ) -> Tuple[torch.Tensor, str, dict]:
        """
        Apply gain to the input audio tensor.

        Args:
            x: torch.Tensor
                Input audio tensor.
            rate: float, optional
                Gain rate.

        Returns:
            Tuple[torch.Tensor, str, dict]:
                Gain-adjusted audio, attack type, and parameters.
        """
        
        if self.mode == 'train' and rate is None:
            rate =  sample_from_intervals(intervals=self.config.gain.rate_intervals) 
            
        elif self.mode in ['test', 'val'] and rate is None:
            raise ValueError('rate should be provided in val and test modes.')
    
        return x * rate, 'gain', {'rate': rate}

    def apply_quantization(
        self, 
        x: torch.Tensor,
        num_bits: Optional[int] = None
        ) -> Tuple[torch.Tensor, str, dict]:
        """
        Quantize the input audio tensor.

        Args:
            x: torch.Tensor
                Input audio tensor.
            num_bits: int, optional
                Number of quantization bits.

        Returns:
            Tuple[torch.Tensor, str, dict]:
                Quantized audio, attack type, and parameters.
        """
        
        if self.mode == 'train' and num_bits is None:
            num_bits = torch.randint(self.config.quantization.min_bits,
                                     self.config.quantization.max_bits + 1,  # Add 1 because the upper limit is exclusive
                                     (1,) ).item()
        
        elif self.mode in ['test', 'val'] and num_bits is None:
            raise ValueError('num_bits should be provided in val and test modes.')
        
        return quantize(x, 
                        num_bits=num_bits), 'quantization', {'num_bits': num_bits}
                        
    def apply_time_stretch(
        self, 
        x: torch.Tensor,
        rate: Optional[float] = None
        ) -> Tuple[torch.Tensor, str, dict]:
        """
        Apply time stretch to the input audio tensor.

        Args:
            x: torch.Tensor
                Input audio tensor.
            rate: float, optional
                Stretching rate.

        Returns:
            Tuple[torch.Tensor, str, dict]:
                Stretched audio, attack type, and parameters.
        """
        if self.mode == 'train' and rate is None:
            rate = sample_from_intervals(intervals=self.config.time_stretch.rate_intervals)
            
        elif self.mode in ['test', 'val'] and rate is None:
            raise ValueError('rate should be provided in val and test modes.')
        
        stretched_audio = time_stretch_wrapper(audio=x, rate=rate)

        output_len = x.shape[-1]
        output_tensor = torch.zeros_like(x, device=x.device)

        if stretched_audio.shape[-1] > output_len:
            output_tensor = stretched_audio[..., :output_len]
        else:
            output_tensor[..., :stretched_audio.shape[-1]] = stretched_audio
        
        return output_tensor, 'time_stretch', {'rate': rate}

    def apply_phase_shift(
        self,
        x: torch.Tensor,
        shift: Optional[int] = None
        ) -> Tuple[torch.Tensor, str, dict]:
        """
        Apply phase shift to the input audio tensor.

        Args:
            x: torch.Tensor
                Input audio tensor.
            shift: int, optional
                Phase shift value.

        Returns:
            Tuple[torch.Tensor, str, dict]:
                Phase-shifted audio, attack type, and parameters.
        """
        # For training, make this less than the half of the hop length, so that
        # the message does not have to be shifted. TODO: Is this a good idea?
        if self.mode == 'train' and shift is None:
            shift = torch.randint(-self.stft.hop_len // 2, self.stft.hop_len // 2 + 1, (1,)).item()

        elif self.mode in ['test', 'val'] and shift is None:
            raise ValueError('phase_shift should be provided in val and test modes.')

        if shift == 0:
            return x, 'phase_shift', {'shift': shift}

        if -1 < shift < 1:
            shift = int(shift * self.stft.hop_len // 2)

        return phase_shift(x, shift=shift), 'phase_shift', {'shift': shift}
    
    # SpecAugment
    def apply_time_mask(
        self, 
        spec: torch.Tensor,
        max_ratio: Optional[float] = None
        ) -> Tuple[torch.Tensor, str, dict]:
        """Apply time masking spectrogram augmentation to the input spectrogram tensor.

        Parameters
        ----------
        spec: torch.Tensor (shape=[B, F, T])
            The input audio tensor spectrogram.
        
        max_ratio: float
            The maximum ratio of the time frames to mask.

        Returns
        -------
        torch.Tensor
            The time-masked audio tensor.       
        """
        num_time_frames = spec.shape[-1]
        if self.mode == 'train' and max_ratio is None:
            max_ratio = self.config.time_mask.max_ratio

        elif self.mode in ['test', 'val'] and max_ratio is None:
            raise ValueError('max_ratio should be provided in val and test modes.')

        mask_width = int(max_ratio * num_time_frames)

        return time_mask(spec, width=mask_width), 'time_mask', {'max_ratio': max_ratio}

    def apply_freq_mask(
        self, 
        spec: torch.Tensor,
        max_ratio: Optional[float] = None
        ) -> Tuple[torch.Tensor, str, dict]:
        """Apply frequency masking spectrogram augmentation to the input spectrogram tensor.

        Parameters
        ----------
        spec: torch.Tensor (shape=[B, F, T])
            The input spectrogram tensor.
        
        ratio: float
            The maximum ratio of the frequency bands to mask.

        Returns
        -------
        torch.Tensor
            The frequency-masked audio tensor.
        """
        num_freq_bins = spec.shape[-2]

        if self.mode == 'train' and max_ratio is None:
            max_ratio = self.config.freq_mask.max_ratio

        elif self.mode in ['test', 'val'] and max_ratio is None:
            raise ValueError('max_ratio should be provided in val and test modes.')

        mask_width = int(max_ratio * num_freq_bins)

        return freq_mask(spec, width=mask_width), 'freq_mask', {'max_ratio': max_ratio}
