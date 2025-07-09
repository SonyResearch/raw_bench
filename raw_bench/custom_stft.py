import torch
from typing import Tuple

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class STFT(torch.nn.Module, metaclass=Singleton):
    def __init__(
        self,
        filter_length: int = 4096,
        hop_length: int = 2048,
        win_length: int = 4096,
        num_samples: int = None
    ):
        """
        Short-Time Fourier Transform (STFT) module with singleton pattern.

        Args:
            filter_length: int
                Length of the FFT window.
            hop_length: int
                Number of samples between successive frames.
            win_length: int
                Window size.
            num_samples: int, optional
                Number of samples in the original signal.
        """
        super(STFT, self).__init__()

        self.filter_length = filter_length
        self.hop_len = hop_length
        if win_length is not None:
            self.win_len = win_length
        else:
            self.win_len = filter_length
        self.window = torch.hann_window(self.win_len)
        self.num_samples = num_samples

    def transform(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the magnitude and phase of the STFT of the input tensor.

        Args:
            x: torch.Tensor
                Input audio tensor of shape (batch, time).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                Magnitude and phase tensors.
        """
        # # starting indices: [0] + [hop_length] + [hop_length*1] + [hop_length*2] + ... + [hop_length*(n_frames-1)]
        # # ending indices: [starting_indices[i] + win_len - 1 for i in range(n_frames)]
        # centered_length = x.shape[-1] + self.win_len <- this is done by torch.stft when center=True
        # n_frames = (centered_length - self.win_len) // self.hop_len + 1
        # # here, + 1 means the first frame, and (centered_length - self.win_len) // self.hop_len means the rest of the frames
        # n_frames == # frames in fft's

        if x.shape[1] % self.hop_len != 0:
            x = torch.nn.functional.pad(x, (0, self.hop_len - x.shape[-1]%self.hop_len))

        fft = torch.stft(x, self.filter_length, self.hop_len, self.win_len, window=self.window.to(x.device), return_complex=True)
        magnitude, phase = self.complex_to_mag_phase(fft)

        return magnitude, phase

    def inverse(
        self,
        magnitude: torch.Tensor,
        phase: torch.Tensor,
        num_samples: int = None
    ):
        """
        Inverse STFT: reconstruct the waveform from magnitude and phase.

        Args:
            magnitude: torch.Tensor
                Magnitude tensor.
            phase: torch.Tensor
                Phase tensor.
            num_samples: int, optional
                Number of output samples.

        Returns:
            torch.Tensor: Reconstructed waveform.
        """
        recombine_magnitude_phase = self.mag_phase_to_complex(magnitude, phase)

        inverse_transform = torch.istft(recombine_magnitude_phase, 
                                        self.filter_length, 
                                        hop_length=self.hop_len, 
                                        win_length=self.win_len,
                                        window=self.window.to(magnitude.device)).unsqueeze(1)  # , length=self.num_samples
        if num_samples is None:
            num_samples = self.num_samples

        padding = self.hop_len - (num_samples % self.hop_len)

        if padding != self.hop_len:
            inverse_transform = inverse_transform[:, :, :-padding]

        return inverse_transform

    @staticmethod
    def complex_to_mag_phase(
        complex_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert a complex tensor to magnitude and phase.

        Args:
            complex_tensor: torch.Tensor
                Complex-valued tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                Magnitude and phase tensors.
        """
        real_part, imag_part = complex_tensor.real, complex_tensor.imag
        squared = real_part**2 + imag_part**2
        additive_epsilon = torch.ones_like(squared) * (squared == 0).float() * 1e-24
        magnitude = torch.sqrt(squared + additive_epsilon) - torch.sqrt(additive_epsilon)
        phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data)).float()

        return magnitude, phase

    @staticmethod
    def mag_phase_to_complex(
        magnitude: torch.Tensor,
        phase: torch.Tensor
    ):
        """
        Combine magnitude and phase into a complex tensor.

        Args:
            magnitude: torch.Tensor
                Magnitude tensor.
            phase: torch.Tensor
                Phase tensor.

        Returns:
            torch.Tensor: Complex-valued tensor.
        """
        recombine_magnitude_phase = magnitude*torch.cos(phase) + 1j*magnitude*torch.sin(phase)
        return recombine_magnitude_phase