import torch
from jaxtyping import Float
from torch import Tensor

from ml4gw.spectral import fast_spectral_density
from ml4gw.types import FrequencySeries1to3d, TimeSeries1to3d


class SpectralDensity(torch.nn.Module):
    def __init__( #assum that it's always fast
        self,
        sample_rate: float,
        fftlength: float,
        overlap: float | None = None,
        average: str = "mean",
        window: Float[Tensor, " {int(fftlength*sample_rate)}"] | None = None,
    ) -> None:
        if overlap is None:
            overlap = fftlength / 2
        elif overlap >= fftlength:
            raise ValueError(
                f"Can't have overlap {overlap} longer than fftlength "
                f"{fftlength}"
            )

        super().__init__()

        self.nperseg = int(fftlength * sample_rate)
        self.nstride = self.nperseg - int(overlap * sample_rate)
        if window is None:
            window = torch.hann_window(self.nperseg)

        if window.size(0) != self.nperseg:
            raise ValueError(
                f"Window must have length {self.nperseg} got {window.size(0)}"
            )
        self.register_buffer("window", window)
        
        scale = 1.0 / (sample_rate * (self.window**2).sum())
        self.register_buffer("scale", scale)

        if average not in ("mean", "median"):
            raise ValueError(
                f'average must be "mean" or "median", got {average} instead'
            )
        self.average = average
        n = 34
        ii_2 = 2 * torch.arange(1.0, (n - 1) // 2 + 1)
        self.bias = 1 + torch.sum(1.0 / (ii_2 + 1) - 1.0 / ii_2)

    def forward( #Assume that its always fast
        self, x: TimeSeries1to3d, y: TimeSeries1to3d | None = None
    ) -> FrequencySeries1to3d:
        return fast_spectral_density(x,
                                     nperseg=self.nperseg,
                                     nstride=self.nstride,
                                     window=self.window,
                                     scale=self.scale,
                                     average=self.average,
                                     bias = self.bias)