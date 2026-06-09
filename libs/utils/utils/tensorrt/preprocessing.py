from collections.abc import Callable

import torch
from torch import Tensor

from ml4gw.transforms import (
    SpectralDensity,
    Whiten,
    Decimator,
    SingleQTransform,
)
from ml4gw.utils.slicing import unfold_windows

class PsdEstimator(torch.nn.Module):
    def __init__(
        self,
        length: float,
        sample_rate: float,
        fftlength: float,
        window: Tensor | None = None,
        overlap: float | None = None,
        average: str = "median",
        fast: bool = True,
    ) -> None:
        super().__init__()
        self.size = int(length * sample_rate)
        # Initialize spectral density estimator
        self.spectral_density = SpectralDensity(
            sample_rate, fftlength, overlap, average, window=window, fast=fast
        )

    def forward(self, X: Tensor) -> tuple[Tensor, Tensor]:
        splits = [X.size(-1) - self.size, self.size]
        background, X = torch.split(X, splits, dim=-1)
        background = background[0]
        X = X[1]
        
        psds = self.spectral_density(background.double())
        return X, psds


class BatchWhitener(torch.nn.Module):
    def __init__(
        self,
        kernel_length: float,
        sample_rate: float,
        inference_sampling_rate: float,
        batch_size: int,
        fduration: float,
        fftlength: float,
        augmentor: Callable[[Tensor], Tensor] | None = None,
        highpass: float | None = None,
        lowpass: float | None = None,
        return_whitened: bool = False,
        num_channels: int = 2
    ) -> None:
        super().__init__()
        # Calculate stride between kernels based on inference sampling rate
        self.stride_size = int(sample_rate / inference_sampling_rate)
        # Convert kernel length to samples
        self.kernel_size = int(kernel_length * sample_rate)
        self.augmentor = augmentor
        self.return_whitened = return_whitened

        # do length calculations in units of samples,
        # then convert back to length to guard for intification
        strides = (batch_size - 1) * self.stride_size
        fsize = int(fduration * sample_rate)
        size = strides + self.kernel_size + fsize
        length = size / sample_rate

        # Initialize PSD estimator with calculated total length
        self.psd_estimator = PsdEstimator(
            length,
            sample_rate,
            fftlength=fftlength,
            overlap=None,
            average="median",
            fast=highpass is not None,
        )
        # Initialize whitening module
        self.whitener = Whiten(fduration, sample_rate, highpass, lowpass)
        self.num_channels = num_channels

    def forward(self, x: Tensor) -> Tensor:
        # Determine number of channels for later reshaping

        x, psd = self.psd_estimator(x.double())
        whitened = self.whitener(x, psd)
        
        x = unfold_windows(whitened, self.kernel_size, self.stride_size)
        x = x.reshape(-1, self.num_channels, self.kernel_size)

        # Apply optional augmentation
        if self.augmentor is not None:
            x = self.augmentor(x)

        if self.return_whitened:
            return x, whitened
        return x