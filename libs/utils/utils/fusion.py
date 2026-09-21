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
from utils.s3 import open_file

class FusedBatchWhitener(torch.nn.Module):
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
        models: list[str],
        k: int,
        ht_highpass: float = 32,
        ht_lowpass: float = 256,
        keep_last_n_seconds: float = None,
        highpass: float | None = None,
        lowpass: float | None = None,
        return_whitened: bool = False,
        batches: int | None = None,
    ) -> None:
        super().__init__()
        self.stride_size = int(sample_rate / inference_sampling_rate)
        self.kernel_size = int(kernel_length * sample_rate)
        self.return_whitened = return_whitened
        strides = (batch_size - 1) * self.stride_size
        fsize = int(fduration * sample_rate)
        size = strides + self.kernel_size + fsize
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
        self.kernel_size = int(keep_last_n_seconds * sample_rate)
        heterodyning_phase = torch.tensor(np.load(chirp_mass_file))
        self.num_chirp_masses = heterodyning_phase.shape[0]
        self.k = k
        length = length - kernel_length + keep_last_n_seconds - fduration
        input_size = kernel_length + (batch_size - 1) / inference_sampling_rate
        self.whitener = ht_Whiten(input_size, fduration, sample_rate, heterodyning_phase, 
                                  int(sample_rate*length), highpass, lowpass, ht_highpass, ht_lowpass, batches)
        graphs = []
        for model in models:
            with open_file(model, "rb") as f:
                graph = torch.jit.load(f, map_location="cpu")
            graph.eval()
            graphs.append(graph)
        
        self.graphs = torch.nn.ModuleList(graphs)
    
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
        x_td, x = self.whitener(x, psd, validating = True)
        
        x_td = unfold_windows(x_td, self.kernel_size, self.stride_size)
        x_td = x_td.reshape(-1, num_channels, self.kernel_size)
        
        x = unfold_windows(x, self.kernel_size, self.stride_size)
        x = x.reshape(-1, num_channels*self.num_chirp_masses, self.kernel_size)
        
        _B, _C, _T = x.shape
        x = self.topk_bin(x, self.num_chirp_masses, _B)

        low = self.graphs[0].time_domain_resnet.conv1(torch.cat([x_td, x], dim=1))
        low = self.graphs[0].time_domain_resnet.bn1(low)
        low = self.graphs[0].time_domain_resnet.relu(low)
        low = self.graphs[0].time_domain_resnet.maxpool(low)
        for layer in self.graphs[0].time_domain_resnet.residual_layers.children():
            low = layer(low)
        low = self.graphs[0].time_domain_resnet.avgpool(low)
        low = low.squeeze(-1)
        
        high = self.graphs[1].conv1(x_td)
        high = self.graphs[1].bn1(high)
        high = self.graphs[1].relu(high)
        high = self.graphs[1].maxpool(high)
        for layer in self.graphs[1].residual_layers.children():
            high = layer(high)
        high = self.graphs[1].avgpool(high)
        high = high.squeeze(-1)
        return torch.cat([low, high], dim = 1)