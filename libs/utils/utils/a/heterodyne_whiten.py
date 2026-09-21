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
    validating: bool,
    sample_rate: float,
    pad: int,
    heterodyning_phase: Tensor | None = None,
    keep_last_n_samples: int | None = None,
    batches: int | None = None,
    crop: bool = True,
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
    
    B, C, F = X_tilde.shape
    H, _ = heterodyning_phase.shape
    X_ht_tilde = heterodyning_phase[None, :, :] * X_tilde[:, :, None]   # (B, 2, 100, F)
    X_ht_tilde = X_ht_tilde.reshape(B, C*H, F)
    
    X_td = []
    X_ht = []
    if (batches is not None) & (validating == False):
        step = int(B//batches)
        for i in range(batches):
            X_td.append(torch.fft.irfft(X_tilde[step*i:step*(i+1)], n=X.shape[-1], 
                                        norm="forward", dim=-1)[:, :, -keep_last_n_samples-pad:-pad])
            X_ht.append(torch.fft.irfft(X_ht_tilde[step*i:step*(i+1)], n=X.shape[-1], 
                                        norm="forward", dim=-1)[:, :, -keep_last_n_samples-pad:-pad])
        X_td = torch.concatenate(X_td)
        X_ht = torch.concatenate(X_ht)
    else:
        X_td = torch.fft.irfft(X_tilde, n=X.shape[-1], 
                               norm="forward", dim=-1)[:, :, -keep_last_n_samples-pad:-pad]
        X_ht = torch.fft.irfft(X_ht_tilde, n=X.shape[-1], 
                               norm="forward", dim=-1)[:, :, -keep_last_n_samples-pad:-pad]

    X_td = X_td.float() / sample_rate**0.5
    X_ht = X_ht.float() / sample_rate**0.5
    return X_td, X_ht


def whiten(
    X: WaveformTensor,
    psd: PSDTensor,
    validating: bool,
    fduration: Float[Tensor, " time"] | float,
    sample_rate: float,
    heterodyning_phase: Tensor | None = None,
    keep_last_n_samples: int | None = None,
    highpass: float | None = None,
    lowpass: float | None = None,
    crop: bool = True,
    batches: int | None = None,
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
    if heterodyning_phase.size(-1) != num_freqs:
        angle = torch.angle(heterodyning_phase)
        mag = torch.abs(heterodyning_phase)
        angle = unwrap(angle, dim=-1)
        angle = torch.nn.functional.interpolate(angle.unsqueeze(1), size=(num_freqs,), mode="linear").squeeze(1)
        mag = torch.nn.functional.interpolate(mag.unsqueeze(1), size=(num_freqs,), mode="linear").squeeze(1)
        heterodyning_phase = mag * torch.exp(1j * angle)
    
    # truncate it to have the desired
    # time domain response length
    psd = truncate_inverse_power_spectrum(
        psd,
        fduration,
        sample_rate,
        highpass,
        lowpass,
    )
    return normalize_by_psd(X, psd, validating, sample_rate, pad, 
                            heterodyning_phase, keep_last_n_samples, 
                            batches, crop)

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
        fduration: float,
        sample_rate: float,
        heterodyning_phase: Tensor | None = None,
        keep_last_n_samples: int | None = None,
        highpass: float | None = None,
        lowpass: float | None = None,
        batches: int | None = None
    ) -> None:
        super().__init__()
        self.fduration = fduration
        self.sample_rate = sample_rate
        self.highpass = highpass
        self.lowpass = lowpass
        self.register_buffer("heterodyning_phase", heterodyning_phase)
        size = int(fduration * sample_rate)
        window = torch.hann_window(size, dtype=torch.float64)
        self.register_buffer("window", window)
        self.keep_last_n_samples = keep_last_n_samples
        self.batches = batches

    def forward(
        self,
        X: TimeSeries3d,
        psd: FrequencySeries1to3d,
        validating: bool = True,
        crop: bool = True,
    ) -> TimeSeries3d:
        return whiten(
            X,
            psd,
            validating,
            fduration=self.window,
            sample_rate=self.sample_rate,
            highpass=self.highpass,
            lowpass=self.lowpass,
            crop=crop,
            heterodyning_phase = self.heterodyning_phase,
            keep_last_n_samples = self.keep_last_n_samples,
            batches = self.batches,
        )