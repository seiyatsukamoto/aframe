import math
import torch
from torch import Tensor
from typing import Literal

from ml4gw.transforms import Heterodyne
import numpy as np

class HeterodyneAugmentor(torch.nn.Module):
    """
    Apply a heterodyne transform over a grid of chirp masses to a batch
    of time-series data.

    Args:
        sample_rate (float): Input sampling rate in Hz.
        kernel_length (float): Length of output kernels in seconds.
        
        keep_last_n_seconds (float):
            If provided, only the last `n` seconds of the kernel_length are
            returned. Otherwise, the full kernel_length is returned.
    Shape:
        Input: (batch_size, channels, time)
        Output: (batch_size, channels * num_chirp_masses, time_out)

        where `time_out = time` unless `keep_last_n_seconds` is set.

    Note:
        The output shape of the `BatchWhitener` is
        (batch_size, channels, kernel_size).
        The `HeterodyneAugmentor` changes the output shape to
        (batch_size, channels * num_chirp_masses, time_out)
        where time_out is determined by the `keep_last_n_seconds` parameter.

    Example:
        >>> augmentor = HeterodyneAugmentor(
        ...     sample_rate=2048, kernel_length=8, chirp_mass_low=1.0,
        ...     chirp_mass_high=2.5, num_chirp_masses=100,
        ...     chirp_mass_spacing="log", keep_last_n_seconds=4.0,
        ... )
        >>> x = torch.randn(8, 2, 16384)  # (batch, channels, time)
        >>> y = augmentor(x)
        >>> # y: (8, 2 * 100, 8192) since we keep the last 4 seconds at 2048 Hz
    """

    def __init__(
        self,
        sample_rate: float,
        kernel_length: float,
        chirp_mass_file: str,
        keep_last_n_seconds: float = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.kernel_length = kernel_length
        self.keep_last_n_seconds = keep_last_n_seconds

        self.chirp_mass_grid = torch.Tensor(np.load(chirp_mass_file))
        self.num_chirp_masses = len(self.chirp_mass_grid)
        

        if self.keep_last_n_seconds is not None:
            self.keep_last_n_samples = int(
                self.keep_last_n_seconds * sample_rate
            )

        self.heterodyne_transform = Heterodyne(
            sample_rate=sample_rate,
            kernel_length=kernel_length,
            chirp_mass=self.chirp_mass_grid,
            return_type="time",
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input data of shape (batch, channels, time).

        Returns:
            Tensor: Output data of shape
                (batch, channels * num_chirp_masses, time_out),
                where `time_out` is either the same as input time
                or determined by `keep_last_n_seconds`.
        """
        _B, _C, _T = x.shape
        x_heterodyned = torch.empty((_B, _C * self.num_chirp_masses, _T))
        # Heterodyne the whitened timeseries
        x = self.heterodyne_transform(x)
        # Reshaping x from (batch_size, channels, num_chirp_mass, kernel_size)
        # to (batch_size, channels x num_chirp_mass, kernel_size)
        x = x.reshape(_B, _C * self.num_chirp_masses, _T)
        x_heterodyned[:, :, :] = x
        # Returning the desired length of heterodyned strain in the
        # time dimension
        if self.keep_last_n_seconds is not None:
            return x_heterodyned[..., -self.keep_last_n_samples :]
        else:
            return x_heterodyned
