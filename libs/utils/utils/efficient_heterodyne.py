import torch
from jaxtyping import Float
from torch import Tensor

from ml4gw.types import (
    FrequencySeries1to3d,
    PSDTensor,
    TimeSeries1to3d,
    WaveformTensor,
    TimeSeries3d
)
import torch.nn.functional as F

def unwrap(phase, dim=-1):
    dphase = torch.diff(phase, dim=dim)
    dphase_wrapped = (dphase + torch.pi) % (2 * torch.pi) - torch.pi
    dphase_wrapped = torch.where(
        (dphase_wrapped == -torch.pi) & (dphase > 0),
        torch.full_like(dphase_wrapped, torch.pi),
        dphase_wrapped,
    )
    correction = dphase_wrapped - dphase
    cumulative_correction = torch.cumsum(correction, dim=dim)
    pad_shape = list(phase.shape)
    pad_shape[dim] = 1
    cumulative_correction = torch.cat(
        [torch.zeros(pad_shape, dtype=phase.dtype, device=phase.device), cumulative_correction],
        dim=dim,
    )
    return phase + cumulative_correction

def topk_bin(X, k, _M, _B, return_preds = False):
    pooled = F.avg_pool1d(X.abs(), kernel_size = 31, stride = 5, padding = 0)
    hl = torch.max(pooled[:, :_M]*pooled[:, _M:], dim = -1)[0]
    preds = hl.topk(k, dim = -1)[1]
    preds = torch.concat([preds, preds+_M], dim = -1)
    preds = preds.unsqueeze(-1).expand(-1, -1, X.size(-1))
    if return_preds == False:
        return torch.gather(X, dim=1, index=preds), None
    else:
        return torch.gather(X, dim=1, index=preds), preds

def truncate_inverse_power_spectrum(
    psd: PSDTensor,
    fduration: Float[Tensor, " time"] | float,
    sample_rate: float,
    highpass: float | None = None,
    lowpass: float | None = None,
) -> PSDTensor:
    
    num_freqs = psd.size(-1)
    N = (num_freqs - 1) * 2

    # use the inverse of the ASD as the
    # impulse response function
    inv_asd = 1 / psd**0.5

    # zero out frequencies if we want the filter
    # to perform highpass/lowpass filtering
    df = sample_rate / N
    if highpass is not None:
        idx = int(highpass / df)
        inv_asd[:, :, :idx] = 0
    if lowpass is not None:
        idx = int(lowpass / df)
        inv_asd[:, :, idx:] = 0

    if inv_asd.size(-1) % 2:
        inv_asd[:, :, -1] = 0

    # now convert to time domain representation
    q = torch.fft.irfft(inv_asd, n=N, norm="forward", dim=-1)

    # taper the edges of the TD filter
    if isinstance(fduration, Tensor):
        pad = fduration.size(-1) // 2
        window = fduration
    else:
        pad = int(fduration * sample_rate / 2)
        window = torch.hann_window(2 * pad, dtype=torch.float64)
        window = window.to(q.device)

    # 0 out anything else between the tapering regions
    q[:, :, :pad] *= window[-pad:]
    q[:, :, -pad:] *= window[:pad]
    if 2 * pad < q.size(-1):
        q[:, :, pad : q.size(-1) - pad] = 0

    # convert back to the frequency domain
    # to build the desired PSD
    inv_asd = torch.fft.rfft(q, n=N, norm="forward", dim=-1)
    inv_psd = inv_asd * inv_asd.conj()
    psd = 1 / inv_psd.abs()
    return psd / 2

def normalize_by_psd(
    X: WaveformTensor,
    psd: PSDTensor,
    sample_rate: float,
    pad: int,
    k: int, 
    heterodyning_phase: Tensor | None = None,
    keep_last_n_samples: int | None = None,
    crop: bool = True,
    return_preds: bool = False
):
    # compute the FFT of the section we want to whiten
    # and divide it by the ASD of the background section.
    # If the ASD of any background bin hit inf, set the
    # corresponding bin to 0
    X = X - X.mean(-1, keepdims=True)
    X_tilde = torch.fft.rfft(X.double(), norm="forward", dim=-1)
    inv_asd = torch.nan_to_num(psd**-0.5) #B C F
    X_tilde = X_tilde * inv_asd
    X_tilde[..., 0] = 0
    X = torch.fft.irfft(X_tilde, n=X.shape[-1], norm="forward", dim=-1)[:, :, pad:-pad]
    X = X.float() / sample_rate**0.5
    
    X_tilde = torch.fft.rfft(X, norm="forward", dim=-1)
    B, C, F = X_tilde.shape
    H, _ = heterodyning_phase.shape
    X_tilde = heterodyning_phase[None, :, :] * X_tilde[:, :, None]   # (B, C, H, F)
    X_tilde[..., 0] = 0
    X_tilde = X_tilde.reshape(B, C*H, F)
    
    X_ht = torch.fft.irfft(X_tilde, n=X.shape[-1], norm="forward", dim=-1)[:, :, -keep_last_n_samples:]
    _B, _C, _W, = X_ht.shape
    X_ht, preds = topk_bin(X_ht, k, _C//2, _W, return_preds)
    X_ht = X_ht.float()
    if return_preds == True:
        return torch.cat((X[..., -keep_last_n_samples:], X_ht), dim = 1), preds
    else:
        return torch.cat((X[..., -keep_last_n_samples:], X_ht), dim = 1)


def whiten(
    X: WaveformTensor,
    psd: PSDTensor,
    fduration: Float[Tensor, " time"] | float,
    sample_rate: float,
    k: int,
    heterodyning_phase: Tensor | None = None,
    keep_last_n_samples: int | None = None,
    highpass: float | None = None,
    lowpass: float | None = None,
    crop: bool = True,
    return_preds: bool = False,
) -> WaveformTensor:

    # figure out how much data we'll need to slice
    # off after whitening
    if isinstance(fduration, Tensor):
        pad = fduration.size(-1) // 2
    else:
        pad = int(fduration * sample_rate / 2)

    N = X.size(-1)
    if N <= (2 * pad):
        raise ValueError(
            f"Not enough timeseries samples {N} for number of "
            f"padded samples {2 * pad}"
        )

    # normalize the number of expected dimensions in the PSD
    while psd.ndim < 3:
        psd = psd[None]

    # possibly interpolate our PSD to match the number
    # of frequency bins we expect to get from X
    num_freqs = N // 2 + 1
    if psd.size(-1) != num_freqs:
        # TODO: does there need to be any rescaling to
        # keep the integral of the PSD constant?
        psd = torch.nn.functional.interpolate(
            psd, size=(num_freqs,), mode="linear"
        )
    
    # truncate it to have the desired
    # time domain response length
    psd = truncate_inverse_power_spectrum(
        psd,
        fduration,
        sample_rate,
        highpass,
        lowpass,
    )
    if return_preds == True:
        return normalize_by_psd(X = X,
                                psd = psd,
                                sample_rate = sample_rate,
                                pad = pad,
                                k = k,
                                heterodyning_phase = heterodyning_phase,
                                keep_last_n_samples = keep_last_n_samples,
                                crop = crop,
                                return_preds = return_preds)
    else:
        return normalize_by_psd(X = X,
                                psd = psd,
                                sample_rate = sample_rate,
                                pad = pad,
                                k = k,
                                heterodyning_phase = heterodyning_phase,
                                keep_last_n_samples = keep_last_n_samples,
                                crop = crop,
                                return_preds = return_preds)
        
class ht_Whiten(torch.nn.Module):
    """
    Normalize the frequency content of timeseries
    data by a provided power spectral density, such
    that if the timeseries are sampled from the same
    distribution as the PSD the normalized power will
    be approximately unity across all frequency bins.
    The whitened timeseries will then also have
    0 mean and unit variance.

    In order to avoid edge effects due to filter settle-in,
    the provided PSDs will have their spectrum truncated
    such that their impulse response time in the time
    domain is ``fduration`` seconds, and ``fduration / 2``
    seconds worth of data will be removed from each
    edge of the whitened timeseries.

    For more information, see the documentation for
    :meth:`~ml4gw.spectral.whiten`.

    Args:
        fduration:
            The length of the whitening filter's impulse
            response, in seconds. ``fduration / 2`` seconds
            worth of data will be cropped from the edges
            of the whitened timeseries.
        sample_rate:
            Rate at which timeseries data passed at call
            time is expected to be sampled
        highpass:
            Cutoff frequency to apply highpass filtering
            during whitening. If left as ``None``, no highpass
            filtering will be performed.
        lowpass:
            Cutoff frequency to apply lowpass filtering
            during whitening. If left as ``None``, no lowpass
            filtering will be performed.
    """

    def __init__(
        self,
        kernel_length: float,
        fduration: float,
        sample_rate: float,
        k: int,
        heterodyning_phase: Tensor | None = None,
        keep_last_n_samples: int | None = None,
        highpass: float | None = None,
        lowpass: float | None = None,
        ht_highpass: float | None = None,
        ht_lowpass: float | None = None,
        return_preds: bool = False
    ) -> None:
        super().__init__()
        self.fduration = fduration
        self.sample_rate = sample_rate
        self.highpass = highpass
        self.lowpass = lowpass
        self.ht_highpass = ht_highpass
        self.ht_lowpass = ht_lowpass
        self.return_preds = return_preds
        self.k = k
        N = kernel_length*sample_rate
        num_freqs = int(N//2+1)
        if num_freqs != heterodyning_phase.shape[-1]: #This is broke 
            print('heterodyne rescaling is broken')
            angle = torch.angle(heterodyning_phase)
            mag = torch.abs(heterodyning_phase)
            angle = unwrap(angle, dim=-1)
            angle = torch.nn.functional.interpolate(angle.unsqueeze(1), size=(num_freqs,), mode="linear").squeeze(1)
            mag = torch.nn.functional.interpolate(mag.unsqueeze(1), size=(num_freqs,), mode="linear").squeeze(1)
            heterodyning_phase = mag * torch.exp(1j * angle)
        
        df = sample_rate / N
        if ht_highpass is not None:
            idx = int(ht_highpass / df)
            heterodyning_phase[...,:idx] = 0
        if ht_lowpass is not None:
            idx = int(ht_lowpass / df)
            heterodyning_phase[..., idx:] = 0
            
        self.register_buffer("heterodyning_phase", heterodyning_phase)
        size = int(fduration * sample_rate)
        window = torch.hann_window(size, dtype=torch.float64)
        self.register_buffer("window", window)
        self.keep_last_n_samples = keep_last_n_samples
        
    def forward(
        self,
        X: TimeSeries3d,
        psd: FrequencySeries1to3d,
        crop: bool = True,
    ) -> TimeSeries3d:
        return whiten(
            X,
            psd,
            fduration=self.window,
            sample_rate=self.sample_rate,
            highpass=self.highpass,
            lowpass=self.lowpass,
            crop=crop,
            heterodyning_phase = self.heterodyning_phase,
            keep_last_n_samples = self.keep_last_n_samples,
            k = self.k,
            return_preds = self.return_preds,
        )