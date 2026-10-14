from typing import Literal

import torch
import numpy as np

from ml4gw.constants import MTSUN_SI


class Heterodyne_from_file(torch.nn.Module):
    def __init__(
        self,
        sample_rate: float,
        kernel_length: float,
        chirp_mass_file: str,
        return_type: Literal["time", "freq", "both"],
        highpass: float = 0,
        lowpass: float = 2048,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.kernel_length = kernel_length
        heterodyning_phase = torch.tensor(np.load(chirp_mass_file))
        N = len(heterodyning_phase[0])
        N = (N-1)*2
        df = sample_rate / N
        if highpass is not None:
            idx = int(highpass / df)
            heterodyning_phase[...,:idx] = 0
        if lowpass is not None:
            idx = int(lowpass / df)
            heterodyning_phase[..., idx:] = 0
        self.register_buffer("heterodyning_phase", torch.tensor(np.load(chirp_mass_file)))
        self.return_type = return_type
        if self.return_type not in {"time", "freq", "both"}:
            raise ValueError(
                "Invalid return_type. Must be one of {'time', 'freq', 'both'}."
            )
    
    def forward(
        self, X: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        X_fft = torch.fft.rfft(X, dim=-1)
        X_fft /= self.sample_rate
        X_heterodyned = X_fft[:, :, None] * self.heterodyning_phase[None, :, :]
        X_ifft = torch.fft.irfft(X_heterodyned, dim=-1)
        X_ifft *= self.sample_rate

        if self.return_type == "time":
            return X_ifft
        elif self.return_type == "freq":
            return X_heterodyned
        else:
            return X_ifft, X_heterodyned
