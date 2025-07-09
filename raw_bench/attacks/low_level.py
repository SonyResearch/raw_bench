from librosa.effects import time_stretch 
from torchinterp1d.interp1d import interp1d as tinterp
import torch

def quantize(audio: torch.Tensor, 
             num_bits: int) -> torch.Tensor:
    """Quantizes the audio signal to a given bit depth.
    
    Parameters
    ----------
    audio: torch.Tensor (torch.Tensor): The audio signal to be quantized, assumed to be normalized in the range [-1, 1].
        num_bits (int): The number of bits to quantize to. Lower values increase quantization distortion.
        
    Returns:
        torch.Tensor: Quantized audio signal.
    """
    # Define the quantization levels
    quant_levels = 2 ** num_bits
    
    # Scale audio to range [0, quant_levels-1], round to nearest level, then re-scale back to [-1, 1]
    quantized_audio = torch.round((audio + 1) * (quant_levels / 2 - 1)) / (quant_levels / 2 - 1) - 1
    
    return quantized_audio


def phase_shift(audio: torch.Tensor,
                shift: int) -> torch.Tensor:
    """Shift the phase of an audio tensor by `phase_shift` samples.
    
    Parameters
    ----------
    audio: torch.Tensor
        The input audio tensor.

    shift: int
        The number of samples to shift the phase by.

    Returns
    -------
    shift: torch.Tensor
        The audio tensor with the phase shifted.
    """
    padding = torch.zeros([audio.shape[0], audio.shape[1], abs(shift)]).to(audio.device)
    array = [audio[:,:,shift:], padding] if shift > 0 else [padding, audio[:,:,:shift]]

    return torch.cat(array, 
                     dim=-1)


def time_jitter(audio: torch.Tensor, 
                scale: float = 0.1) -> torch.Tensor:
    """Apply time jitter to audio signal
    Sampling jitter: "https://www.peak-studios.de/en/audio-jitter/#:~:text=Audio%20jitter%20is%20a%20variance,problems%20with%20the%20audio%20hardware."

    Parameters
    ----------
    audio: torch.Tensor
        Audio signal

    scale: float (default=0.1)
        Scale of jitter

    sr: int (default=44100)
        Sample rate

    Returns
    -------
    jittered_audio: torch.Tensor
        Jittered audio signal       
    """
    audio = audio.squeeze(1)
    x = torch.arange(audio.shape[1])[None].repeat(audio.shape[0], 1).to(audio.device)
    x_new = x + torch.normal(mean=0, std=scale, size=x.shape).to(audio.device)
    jittered_audio = tinterp(x, audio, x_new).unsqueeze(1)
    
    return jittered_audio


def inverse_polarity(audio: torch.Tensor):
    """Invert the polarity of an audio signal.
    
    Parameters
    ----------
    audio: torch.Tensor
        The input audio signal.
    
    Returns
    -------
    inverted_audio: torch.Tensor
        The audio signal with inverted polarity.
    """
    inverted_audio = -audio
    
    return inverted_audio

def time_stretch_wrapper(audio: torch.Tensor, 
                         rate: float) -> torch.Tensor:
    """Wrapper for time stretching an audio signal using librosa.
    
    Parameters
    ----------
    audio: torch.Tensor
        The input audio signal.
    
    rate: float
        The rate at which to stretch the audio signal.
        
    Returns
    -------
    torch.Tensor
        The time-stretched audio signal.
    """
    if audio.dim() == 3:
        audio = audio.squeeze(0)
    audio = audio.squeeze(1)
    stretched = time_stretch(audio.detach().cpu().numpy(),
                             rate=rate)

    return torch.tensor(stretched).to(audio.device).unsqueeze(0)