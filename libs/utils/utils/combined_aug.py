import math
import torch
from torch import Tensor
from typing import Literal
from collections.abc import Callable
import torch.nn.functional as F

from .heterodyne_whiten import ht_Whiten
from .preprocessing import PsdEstimator

from ml4gw.transforms import (
    SpectralDensity,
    Whiten,
    Decimator,
    SingleQTransform,
)
from ml4gw.utils.slicing import unfold_windows
import numpy as np

class TopKHeterodyneBatchWhitener(torch.nn.Module):
    """
    BatchWhitener with heterodyne in whiten function
    Added Topk
    """
    def __init__(
        self,
        kernel_length: float,
        sample_rate: float,
        inference_sampling_rate: float,
        batch_size: int,
        fduration: float,
        fftlength: float,
        chirp_mass_file: str,
        k: int,
        ht_highpass: float = 32,
        ht_lowpass: float = 256,
        keep_last_n_seconds: float = None,
        augmentor: Callable[[Tensor], Tensor] | None = None,
        highpass: float | None = None,
        lowpass: float | None = None,
        return_whitened: bool = False,
        batches: int | None = None,
    ) -> None:
        super().__init__()
        #Same as BatchWhitener
        self.stride_size = int(sample_rate / inference_sampling_rate)
        self.kernel_size = int((kernel_length+fduration) * sample_rate)
        self.augmentor = augmentor
        self.return_whitened = return_whitened
        strides = (batch_size - 1) * self.stride_size
        size = strides + self.kernel_size
        length = size / sample_rate
        self.psd_estimator = PsdEstimator(
            length,
            sample_rate,
            fftlength=fftlength,
            overlap=None,
            average="median",
            fast=highpass is not None,
        )
        #Heterodyne __init__
        heterodyning_phase = torch.tensor(np.load(chirp_mass_file))
        self.num_chirp_masses = heterodyning_phase.shape[0]
        self.k = k
        self.whitener = ht_Whiten(fduration, sample_rate, heterodyning_phase, 
                                  int(sample_rate*keep_last_n_seconds), ht_highpass, ht_lowpass, batches)
    
    def topk_bin(self, X, _M, _B):
        pooled = F.avg_pool1d(X.abs(), kernel_size = 31, stride = 5, padding = 0)
        hl = torch.max(pooled[:, :_M]*pooled[:, _M:], dim = -1)[0]
        pred = hl.topk(self.k, dim = -1)[1]
        pred = torch.concat([pred, pred+_M], dim = -1)
        pred = pred.unsqueeze(-1).expand(-1, -1, X.size(-1))
        return torch.gather(X, dim=1, index=pred)
    
    def forward(self, x: Tensor) -> Tensor:
        # Determine number of channels for later reshaping
        if x.ndim == 3:
            num_channels = x.size(1)
        elif x.ndim == 2:
            num_channels = x.size(0)
        else:
            raise ValueError(
                "Expected input to be either 2 or 3 dimensional, "
                "but found shape {}".format(x.shape)
            )

        # Estimate PSD and prepare data
        x, psd = self.psd_estimator(x.double())
        x = unfold_windows(x, self.kernel_size, self.stride_size)
        X_td, x = self.whitener(x, psd)
        
        _B, _C, _T = x.shape
        x = self.topk_bin(x, self.num_chirp_masses, _B)
        x = torch.cat([X_td, x], dim=1)
        # Apply optional augmentation
        if self.augmentor is not None:
            x = self.augmentor(x)
        return x