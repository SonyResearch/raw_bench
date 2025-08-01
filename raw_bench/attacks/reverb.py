import torch
from typing import Tuple


from ..utils import batch_convolution


def apply_reverb(sample: torch.Tensor, 
                 rir: torch.Tensor, 
                 drr_target: torch.Tensor = None, 
                 sr: int = 44100) -> torch.Tensor:
    """Convolve batch of samples with batch of room impulse responses scaled to achieve a target direct-to-reverberation ratio
    """
    if drr_target is not None:
        direct_ir, reverb_ir = decompose_rir(rir=rir, 
                                             sr=sr)
        drr_db = drr(direct_ir, reverb_ir)
        scale = 10**((drr_db - drr_target)/20)
        reverb_ir_scaled = scale[:, None, None]*reverb_ir
        rir_scaled = torch.cat((direct_ir, reverb_ir_scaled), axis=2)

    else:
        rir_scaled = rir

    return batch_convolution(sample, 
                             rir_scaled, 
                             pad_both_sides=False)


def drr(direct_ir: torch.Tensor, 
        reverb_ir: torch.Tensor) -> torch.Tensor:
    """Compute direct-to-reverberation ratio
    
    Parameters
    ----------
    direct_ir: torch.Tensor

    reverb_ir: torch.Tensor
    
    Returns
    -------
    drr_db: torch.Tensor
    
    """
    direct_ir_flat = direct_ir.view(direct_ir.shape[0], -1)
    reverb_ir_flat = reverb_ir.view(reverb_ir.shape[0], -1)
    drr_db = 10*torch.log10(torch.square(direct_ir_flat).sum(dim=1)/torch.square(reverb_ir_flat).sum(dim=1))

    return drr_db


def decompose_rir(rir: torch.Tensor, 
                  sr: int = 44100, 
                  window_ms: float = 5.0) -> Tuple[torch.Tensor, torch.Tensor]:
    direct_window = int(window_ms/1000*sr)
    direct_ir, reverb_ir = rir[:,:,:direct_window], rir[:,:,direct_window:]

    return direct_ir, reverb_ir