import math
import random
import torch
import torch.nn as nn
from audiotools import AudioSignal
from dac.utils import download
from dac import DAC
from transformers import EncodecModel
from typing import List


class NeuralCodecWrapper(nn.Module):
    def __init__(self, 
                 codec_sr: int = None, 
                 model_sr: int = None, 
                 supported_bandwidths: List[float] = None, 
                 supported_n_codebooks: List[int] = None, 
                 downsampling_ratio: float = None, 
                 n_codebooks: int = None,
                 device: str = 'cuda'):
        """
        Initializes the NeuralCodecWrapper with the given parameters.
        
        Parameters
        ----------
        codec_sr : int
            The sample rate of the codec.

        model_sr : int
            The sample rate of the model.

        supported_bandwidths : list of float
            List of supported bandwidths.

        supported_n_codebooks : list of int
            List of supported number of codebooks.

        downsampling_ratio : float
            The downsampling ratio of the codec.

        n_codebooks : int
            The number of the  maximum codebooks that codec has.
        """
        super(NeuralCodecWrapper, self).__init__()
        self.codec_sr = codec_sr
        self.model_sr = model_sr
        self.supported_bandwidths = supported_bandwidths
        self.supported_n_codebooks = supported_n_codebooks
        self.downsampling_ratio = downsampling_ratio
        self.n_codebooks = n_codebooks
        self.device = device

    def validate_properties(self):
        """
        Ensures all necessary properties are set.
        
        Raises
        ------
        AssertionError
            If any of the required properties are not set.
        """
        assert self.codec_sr, "codec_sr must be set"
        assert self.model_sr, "model_sr must be set"
        assert self.supported_bandwidths, "supported_bandwidths must be set"
        assert self.supported_n_codebooks, "supported_n_codebooks must be set"
        assert self.downsampling_ratio, "downsampling_ratio must be set"
        assert self.n_codebooks, "n_codebooks must be set"

    def print_internal_variables(self):
        """
        Prints all internal variables of the NeuralCodecWrapper.
        """
        print(f"watermark model_sr: {self.model_sr}")
        print(f"codec_sr: {self.codec_sr}")
        print(f"supported_bandwidths: {self.supported_bandwidths}")
        print(f"supported_n_codebooks: {self.supported_n_codebooks}")
        print(f"downsampling_ratio: {self.downsampling_ratio}")
        print(f"n_codebooks: {self.n_codebooks}")


class DACWrapper(NeuralCodecWrapper):
    def __init__(self, 
                 model_sr: int = 44100,
                 codec_type: str = '44khz', 
                 verbose : bool = False,
                 device: str = 'cuda'):
        """
        Initializes the Wrapper of DAC with the given model sample rate and codec type.
        
        Parameters
        ----------
        model_sr : int
            The sample rate of the model.

        codec_type : str
            The type of codec to use. Currently, only '44khz' is supported.

        verbose : bool
            Whether you want to print warning msg etc or not
        """
        super(DACWrapper, self).__init__(device=device)
        self.model_sr = model_sr
        self.verbose  = verbose 
        
        # Set codec sample rate based on codec type
        # see https://github.com/descriptinc/descript-audio-codec
        type_to_sr = {'44khz': 44100, '24khz': 24000, '16khz': 16000}
        self.codec_sr = type_to_sr[codec_type]
        assert self.model_sr == self.codec_sr, f"currently model_sr({self.model_sr}) must be the same as codec_sr({self.codec_sr})" 

        # Download and load the pretrained model
        model_path = download(model_type=codec_type)
        self.model = DAC.load(model_path).to(self.device)
        
        # Calculate the codebook size and downsampling ratio
        self.codebook_size = self.model.codebook_size
        self.downsampling_ratio = math.prod(
            [block.block[-1].stride[0]
             for block in self.model.encoder.block
             if 'EncoderBlock' in str(block.__class__)]
        )
        
        # Set supported number of codebooks and bandwidths
        self.n_codebooks = self.model.n_codebooks
        self.supported_n_codebooks = [i + 1 for i in range(self.model.n_codebooks)]
        self.supported_bandwidths = [
            self.codec_sr / self.downsampling_ratio * math.log2(self.codebook_size) * i
            for i in self.supported_n_codebooks
        ]
        
        # Map bandwidth to the number of codebooks
        self.bandwith_to_ncodebook = {
            bandwidth: n_codebook
            for bandwidth, n_codebook in zip(self.supported_bandwidths, self.supported_n_codebooks)
        }

        # Validate properties
        self.validate_properties()
        
    def forward(self, x: torch.Tensor, target_n_codebook=None, target_bandwidth=None) -> torch.Tensor:
        """
        Forward pass to reconstruct audio based on target bandwidth or codebook.
        
        Parameters
        ----------
        x : torch.Tensor
            The input audio tensor.

        target_n_codebook : int, optional
            The target number of codebooks to use for reconstruction.

        target_bandwidth : float, optional
            The target bandwidth to use for reconstruction.
        
        Returns
        -------
        torch.Tensor
            The reconstructed audio tensor.
        
        Raises
        ------
        AssertionError
            If both target_n_codebook and target_bandwidth are provided.
        """
        assert not (target_n_codebook is not None and target_bandwidth is not None), \
            "Specify either target_n_codebook or target_bandwidth, not both"

        if target_bandwidth is not None:
            if self.verbose:
                print('target bandwidth mode')
            return self.reconstruct_target_bandwidth(x, target_bandwidth)
        if target_n_codebook is not None:
            if self.verbose:
                print('target number of codebook mode')
            return self.reconstruct_target_n_codebook(x, target_n_codebook)
        if self.verbose:
            print('random budget mode')
        return self.reconstruct_random_budget(x)

    def reconstruct_random_budget(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs audio using a random number of codebooks.
        
        Parameters
        ----------
        x : torch.Tensor
            The input audio tensor.
        
        Returns
        -------
        torch.Tensor
            The reconstructed audio tensor.
        """
        n_quantizers = random.choice(self.supported_n_codebooks)
        return self.reconstruct_target_n_codebook(x, n_quantizers)

    def reconstruct_target_bandwidth(self, x: torch.Tensor, bandwidth: float) -> torch.Tensor:
        """
        Reconstructs audio using the closest supported bandwidth.
        
        Parameters
        ----------
        x : torch.Tensor
            The input audio tensor.
        bandwidth : float
            The target bandwidth for reconstruction.
        
        Returns
        -------
        torch.Tensor
            The reconstructed audio tensor.
        """
        closest_bandwidth = min(self.supported_bandwidths, key=lambda x: abs(x - bandwidth))
        if self.verbose and not math.isclose(bandwidth, closest_bandwidth):
            print(f'Instead of bandwidth {bandwidth}, we use {closest_bandwidth} that model supports')
        n_quantizers = self.bandwith_to_ncodebook[closest_bandwidth]
        return self.reconstruct_target_n_codebook(x, n_quantizers)

    def reconstruct_target_n_codebook(self, x: torch.Tensor, n_codebook: int) -> torch.Tensor:
        """
        Reconstructs audio using a specified number of codebooks.
        
        Parameters
        ----------
        x : torch.Tensor
            The input audio tensor.
        n_codebook : int
            The number of codebooks to use for reconstruction.
        
        Returns
        -------
        torch.Tensor
            The reconstructed audio tensor.
        """
        if self.verbose :
            print(f'Reconstruct the audio using {n_codebook} codebooks')
        self.model.eval()
        with torch.inference_mode():
            length = x.shape[-1]
            assert n_codebook in self.supported_n_codebooks, f'n_codebook({n_codebook}) must be in self.supported_n_codebooks({self.supported_n_codebooks})'        
            assert x.shape[1] == 1, 'DAC only supporst monaural audio input'
            return self.model(x, n_quantizers=n_codebook)['audio'][..., :length]


class EncodecWrapper(NeuralCodecWrapper):
    def __init__(self, 
                 model_sr: int = 44100, 
                 codec_type: str = 'facebook/encodec_32khz', 
                 verbose : bool = False,
                 device: str = 'cuda'):
        """
        Initializes the Wrapper of DAC with the given model sample rate and codec type.
        
        Parameters
        ----------
        model_sr : int
            The sample rate of the model.
        codec_type : str
            The type of codec to use. Default is facebook/encodec_32khz. encodec_48khz it not supported
        verbose : bool
            Whether you want to print warning msg etc or not
        """
        super(EncodecWrapper, self).__init__(device=device)
        self.model_sr = model_sr
        self.verbose  = verbose 
        
        # Set codec sample rate based on codec type
        type_to_sr = {
            'facebook/encodec_24khz': 24000,
            'facebook/encodec_32khz': 32000, 
            # 'facebook/encodec_48khz': 48000, Currently this model is not supported. It requires a special paddding scheme
        }
        self.codec_sr = type_to_sr[codec_type]

        # Download and load the pretrained model
        self.model = EncodecModel.from_pretrained(codec_type).to(self.device)
        
        # Calculate the codebook size and downsampling ratio
        self.codebook_size = self.model.config.codebook_size
        self.downsampling_ratio = math.prod(self.model.config.upsampling_ratios)
        
        # Set supported number of codebooks and bandwidths
        self.n_codebooks = len(self.model.quantizer.layers)
        self.supported_bandwidths =[kbps*1000 for kbps in self.model.config.target_bandwidths]
        self.supported_n_codebooks = [int(bps/self.codec_sr*self.downsampling_ratio/math.log2(self.codebook_size))
                                          for bps 
                                          in self.supported_bandwidths]

        # Map bandwidth to the number of codebooks
        self.bandwith_to_ncodebook = {
            bandwidth: n_codebook
            for bandwidth, n_codebook in zip(self.supported_bandwidths, self.supported_n_codebooks)
        }
        # reverse dictionary
        self.ncodebook_to_bandwith = {self.bandwith_to_ncodebook[bandwidth]: bandwidth for bandwidth in self.supported_bandwidths}

        self.requires_resampling = (self.model_sr != self.codec_sr)
        # Validate properties
        self.validate_properties()
        
    def forward(self, x: torch.Tensor, target_n_codebook=None, target_bandwidth=None) -> torch.Tensor:
        """
        Forward pass to reconstruct audio based on target bandwidth or codebook.
        
        Parameters
        ----------
        x : torch.Tensor
            The input audio tensor.

        target_n_codebook : int, optional
            The target number of codebooks to use for reconstruction.

        target_bandwidth : float, optional
            The target bandwidth to use for reconstruction.
        
        Returns
        -------
        torch.Tensor
            The reconstructed audio tensor.
        
        Raises
        ------
        AssertionError
            If both target_n_codebook and target_bandwidth are provided.
        """
        assert not (target_n_codebook is not None and target_bandwidth is not None), \
            "Specify either target_n_codebook or target_bandwidth, not both"

        if target_bandwidth is not None:
            if self.verbose:
                print('target bandwidth mode')
            return self.reconstruct_target_bandwidth(x, target_bandwidth)
        if target_n_codebook is not None:
            if self.verbose:
                print('target number of codebook mode')
            return self.reconstruct_target_n_codebook(x, target_n_codebook)
        if self.verbose:
            print('random budget mode')
        return self.reconstruct_random_budget(x)

    def reconstruct_random_budget(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs audio using a random number of codebooks.
        
        Parameters
        ----------
        x : torch.Tensor
            The input audio tensor.
        
        Returns
        -------
        torch.Tensor
            The reconstructed audio tensor.
        """
        bandwidth = random.choice(self.supported_bandwidths)
        return self.reconstruct_target_bandwidth(x, bandwidth)

    def reconstruct_target_bandwidth(self, x: torch.Tensor, bandwidth: float) -> torch.Tensor:
        """
        Reconstructs audio using the closest supported bandwidth.
        
        Parameters
        ----------
        x : torch.Tensor
            The input audio tensor.
        bandwidth : float
            The target bandwidth for reconstruction.
        
        Returns
        -------
        torch.Tensor
            The reconstructed audio tensor.
        """
        closest_bandwidth = min(self.supported_bandwidths, key=lambda x: abs(x - bandwidth))
        if self.verbose:
            if not math.isclose(bandwidth, closest_bandwidth):
                print(f'Instead of bandwidth {bandwidth}, we use {closest_bandwidth} that model supports')
            print(f'Reconstruct the audio using {self.bandwith_to_ncodebook[closest_bandwidth]} codebooks for bandwidth of {closest_bandwidth}')

        self.model.eval()
        with torch.inference_mode():
            length = x.shape[-1]
            assert x.shape[1] == 1, 'Encodec 24/32khz only support monaural audio input'

            if self.requires_resampling:
                if self.verbose:
                    print(f"resample audio from model_sr:{self.model_sr} to codec_sr: {self.codec_sr}")
                signal = AudioSignal(x, self.model_sr)
                signal.resample(self.codec_sr)
                x = signal.audio_data
                if self.verbose:
                    print("compress audio with audio codec")

            x = self.model(x, bandwidth=closest_bandwidth/1000.)['audio_values'] # / 1000. for bps -> kbps
            
            if self.requires_resampling:
                if self.verbose:
                    print(f"resample audio from codec_sr:{self.codec_sr} to model_sr: {self.model_sr}")
                signal = AudioSignal(x, self.codec_sr)
                signal.resample(self.model_sr)
                x = signal.audio_data

            # pad if required
            num_padding = length - x.shape[-1]               
            if num_padding > 0:
                return torch.cat([x, torch.zeros_like(x[..., :num_padding])], dim=-1)
            if num_padding < 0:
                return x[..., :length]
            
        return x

    def reconstruct_target_n_codebook(self, x: torch.Tensor, n_codebook: int) -> torch.Tensor:
        """
        Reconstructs audio using a specified number of codebooks.
        
        Parameters
        ----------
        x : torch.Tensor
            The input audio tensor.
        n_codebook : int
            The number of codebooks to use for reconstruction.
        
        Returns
        -------
        torch.Tensor
            The reconstructed audio tensor.
        """
        if self.verbose :
            print(f'Reconstruct the audio using {n_codebook} codebooks')

        assert n_codebook in self.supported_n_codebooks, f'n_codebook({n_codebook}) must be in self.supported_n_codebooks({self.supported_n_codebooks})'        
        target_bandwidth = self.ncodebook_to_bandwith[n_codebook]
        return self.reconstruct_target_bandwidth(x, bandwidth=target_bandwidth)
        