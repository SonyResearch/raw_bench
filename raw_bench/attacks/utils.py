from typing import List, Tuple, Union
import torch

def ste(
    original: torch.Tensor,
    compressed: torch.Tensor
) -> torch.Tensor:
    """
    Straight-through estimator for differentiable compression attacks.

    Args:
        original: torch.Tensor
            Original input tensor.
        compressed: torch.Tensor
            Compressed tensor.

    Returns:
        torch.Tensor: Output tensor with gradients passed through the original.
    """
    return original + (compressed-original).detach()        

def choose_random_uniform_val(
    min_val: float,
    max_val: float,
    num_samples: int = 1
) -> Union[float, torch.Tensor]:
    """
    Choose a random value from a uniform distribution between min_val and max_val.

    Args:
        min_val: float
            Minimum value.
        max_val: float
            Maximum value.
        num_samples: int
            Number of samples to draw.

    Returns:
        float or torch.Tensor: Random value(s) drawn from the uniform distribution.
    """
    rand_val = torch.rand(num_samples) * (max_val - min_val) + min_val

    if num_samples == 1:
        rand_val = rand_val.item()

    return rand_val

def sample_from_intervals(
    intervals: List[Tuple[float, float]],
    num_samples: int = 1
) -> Union[float, torch.Tensor]:
    """
    Sample random values from a list of intervals.

    Args:
        intervals: List[Tuple[float, float]]
            List of (min, max) intervals.
        num_samples: int
            Number of samples to draw.

    Returns:
        float or torch.Tensor: Randomly sampled value(s) from the intervals.
    """
    # Randomly select intervals for each sample
    selected_indices = torch.randint(len(intervals), (num_samples,)).tolist()  # Convert tensor to list
    
    # Generate random values for the selected intervals
    random_values = torch.empty(num_samples)
    for i, interval_index in enumerate(selected_indices):
        random_values[i] = torch.empty(1).uniform_(*intervals[interval_index])
    
    # Return a scalar if num_samples is 1, otherwise return a tensor
    if num_samples == 1:
        return random_values.item()
    
    return random_values