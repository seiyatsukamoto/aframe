import math
import torch
from torch import Tensor
from typing import Literal

#from ml4gw.transforms import Heterodyne
#from ml4gw.transforms.heterodyne_from_dir import Heterodyne_from_dir as Heterodyne
from .heterodyne_from_file import Heterodyne_from_file as Heterodyne

import torch.nn.functional as F

class HeterodyneAugmentor(torch.nn.Module):
    def __init__(
        self,
        sample_rate: float,
        kernel_length: float,
        chirp_mass_file: str,
        num_chirp_masses: int = 100,
        keep_last_n_seconds: float = None,
        lowpass: float = None,
        highpass: float = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.kernel_length = kernel_length
        self.keep_last_n_seconds = keep_last_n_seconds
        self.num_chirp_masses = num_chirp_masses
        self.highpass = highpass
        self.lowpass = lowpass
        if self.keep_last_n_seconds is not None:
            self.keep_last_n_samples = int(
                self.keep_last_n_seconds * sample_rate
            )

        self.heterodyne_transform = Heterodyne(
            sample_rate = sample_rate,
            kernel_length = kernel_length,
            chirp_mass_file = chirp_mass_file,
            return_type = 'time',
            highpass = highpass,
            lowpass = lowpass,
        )
    def forward(self, x: Tensor) -> Tensor:
        _B, _C, _T = x.shape
        x_heterodyned = torch.empty((_B, _C * self.num_chirp_masses, _T))
        x = self.heterodyne_transform(x)
        x = x.reshape(_B, _C * self.num_chirp_masses, _T)
        x_heterodyned[:, :, :] = x
        if self.keep_last_n_seconds is not None:
            return x_heterodyned[..., -self.keep_last_n_samples :]
        else:
            return x_heterodyned


class TopKHeterodyneAugmentor(torch.nn.Module):
    def __init__(
        self,
        sample_rate: float,
        kernel_length: float,
        chirp_mass_file: str,
        k: int = 5,
        num_chirp_masses: int = 100,
        keep_last_n_seconds: float = None,
        ht_lowpass: float = None,
        ht_highpass: float = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.kernel_length = kernel_length
        self.k = k
        self.keep_last_n_seconds = keep_last_n_seconds
        self.num_chirp_masses = num_chirp_masses
        self.ht_lowpass = ht_lowpass
        self.ht_highpass = ht_highpass
        if self.keep_last_n_seconds is not None:
            self.keep_last_n_samples = int(
                self.keep_last_n_seconds * sample_rate
            )
        self.heterodyne_transform = Heterodyne(
            sample_rate = sample_rate,
            kernel_length = kernel_length,
            chirp_mass_file = chirp_mass_file,
            return_type = 'time',
            highpass = ht_highpass,
            lowpass = ht_lowpass,
        )
    def topk_bin(self, X, _M, _B):
        pooled = F.avg_pool1d(X.abs(), kernel_size = 31, stride = 5, padding = 0)
        hl = torch.max(pooled[:, :_M]*pooled[:, _M:], dim = -1)[0]
        pred = hl.topk(self.k, dim = -1)[1]
        pred = torch.concat([pred, pred+_M], dim = -1)
        pred = pred.unsqueeze(-1).expand(-1, -1, X.size(-1))
        return torch.gather(X, dim=1, index=pred)
    def forward(self, x: Tensor) -> Tensor:
        _B, _C, _T = x.shape
        x_td = x.clone()
        
        x = self.heterodyne_transform(x)
        x = x.reshape(_B, _C * self.num_chirp_masses, _T)
        x = x[..., -self.keep_last_n_samples :]
        x = self.topk_bin(x, self.num_chirp_masses, _B).float()
        
        x_td = x_td[..., -self.keep_last_n_samples :].float()
        x = torch.cat([x_td, x], dim = 1)
        return x