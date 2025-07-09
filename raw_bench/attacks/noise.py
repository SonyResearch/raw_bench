import torch


def gaussian_noise(audio: torch.Tensor, 
                   mean: float = 0.0, 
                   std: float = 0.1):
    """Generate gaussian noise with specified mean and standard deviation

    Parameters
    ----------
    audio: torch.Tensor
        Input audio tensor

    mean: float (default=0.0)
        Mean of the gaussian noise

    std: float (default=0.1)
        Standard deviation of the gaussian noise

    Returns
    -------
    torch.Tensor
        Gaussian noise
    """
    # This just returns the gaussian noise so that it can be added to the carrier.
    noise = torch.randn_like(audio) * std + mean
    noise = noise.to(audio.device)

    return noise


def pink_noise(
    audio: torch.Tensor,
    noise_std: float = 0.01,
):
    """Add pink background noise to the waveform.
    
    Parameters
    ----------
    audio: torch.Tensor
        Input audio tensor

    noise_std: float (default=0.01)
        Standard deviation of the pink noise    

    Returns
    -------
    torch.Tensor
        Audio tensor with added pink noise    
    """
    noise = generate_pink_noise(audio.shape[-1]) * noise_std
    noise = noise.to(audio.device)
    # Assuming waveform is of shape (bsz, channels, length)
    return audio + noise.unsqueeze(0).unsqueeze(0).to(audio.device)


def generate_pink_noise(length: int) -> torch.Tensor:
    """Generate pink noise using Voss-McCartney algorithm with PyTorch.
    
    Parameters
    ----------
    length: int
        Length of the pink noise

    Returns
    -------
    torch.Tensor
        Pink noise tensor    
    """
    num_rows = 16
    array = torch.randn(num_rows, length // num_rows + 1)
    reshaped_array = torch.cumsum(array, dim=1)
    reshaped_array = reshaped_array.reshape(-1)
    reshaped_array = reshaped_array[:length]
    
    # Normalize
    pink_noise = reshaped_array / torch.max(torch.abs(reshaped_array))

    return pink_noise

