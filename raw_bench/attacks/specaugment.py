import torch
import torchaudio.transforms as T


def time_mask(spec: torch.Tensor,
              width: int = 50) -> torch.Tensor:
    """Apply time masking to a spectrogram
    
    Parameters
    ----------
    spec: torch.Tensor
        Input spectrogram

    width: int (default=50)
        Maximum width of the mask

    Returns
    -------
    torch.Tensor
        Time-masked spectrogram
    """
    return T.TimeMasking(width)(spec)


def freq_mask(spec: torch.Tensor,
              width: int = 50) -> torch.Tensor:
    """Apply frequency masking to a spectrogram
    
    Parameters
    ----------
    spec: torch.Tensor
        Input spectrogram

    width: int (default=50)
        Width of the mask

    Returns
    -------
    torch.Tensor
        Frequency-masked spectrogram
    """
    return T.FrequencyMasking(width)(spec)