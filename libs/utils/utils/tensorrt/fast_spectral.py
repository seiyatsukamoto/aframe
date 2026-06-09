import torch
from jaxtyping import Float
from torch import Tensor

from .types import (
    FrequencySeries1to3d,
    PSDTensor,
    TimeSeries1to3d,
    WaveformTensor,
)


def fast_spectral_density(
    x: TimeSeries1to3d,
    nperseg: int,
    nstride: int,
    window: Float[Tensor, " {nperseg//2+1}"],
    scale: float,
    bias: float,
    average: str = "median",
    y: TimeSeries1to3d | None = None,
) -> FrequencySeries1to3d:

    x = x.double()
    
    x = x - x.mean(axis=-1, keepdims=True)
    fft = torch.stft(
            x,
            n_fft=nperseg,
            hop_length=nstride,
            window=window,
            normalized=False,
            center=False,
            return_complex=False,
    )
    
    real = torch.select(fft, dim=-1, index=0)
    imag = torch.select(fft, dim=-1, index=1)
    fft = real ** 2 + imag ** 2
    fft[:, 1:-1] *= 2
    fft *= scale
    fft, _ = torch.sort(fft, dim=-1)
    fft = fft[..., fft.shape[-1] // 2] / bias
    return fft