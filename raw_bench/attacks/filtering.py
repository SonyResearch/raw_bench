import torch
from torchaudio.functional import highpass_biquad, lowpass_biquad


def lowpass_filter(audio: torch.Tensor, 
                   cutoff_freq: float, 
                   sr: int) -> torch.Tensor:
    """
    Apply a low-pass filter to the input audio signal.
    
    Parameters
    ----------
    audio: torch.Tensor
        The input audio signal, expected as a 1D tensor.
    
    cutoff_freq: float
        The cutoff frequency of the low-pass filter in Hz.
    
    sr: int
        The sample rate of the audio signal.
    
    Returns
    -------
    filtered_audio: torch.Tensor
        The LP-filtered audio signal.
    """
    return lowpass_biquad(waveform=audio, sample_rate=sr, cutoff_freq=cutoff_freq)


def highpass_filter(audio: torch.Tensor, 
                    cutoff_freq: float, 
                    sr: int) -> torch.Tensor:
    """
    Apply a high-pass filter to the input audio signal.
    
    Parameters
    ----------
    audio: torch.Tensor
        The input audio signal, expected as a 1D tensor.
    
    cutoff_freq: float
        The cutoff frequency of the high-pass filter in Hz.
    
    sr: int
        The sample rate of the audio signal.
    
    Returns
    -------
    filtered_audio: torch.Tensor
        The HP-filtered audio signal.
    """
    return highpass_biquad(waveform=audio, sample_rate=sr, cutoff_freq=cutoff_freq)
