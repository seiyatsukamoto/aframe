from typing import Literal, Optional

from architectures import Architecture, SupervisedArchitecture
from architectures.networks import S4Model, WaveNet, Xylophone
from jaxtyping import Float
from ml4gw.nn.resnet.resnet_1d import NormLayer, ResNet1D
from ml4gw.nn.resnet.resnet_2d import ResNet2D
from torch import Tensor
import torch

class MutedModel(SupervisedArchitecture):
    def __init__(
        self,
        num_ifos: int,
        num_chirp_masses: int,
        layers: list[int],
        kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[list[Literal["stride", "dilation"]]] = None,
        norm_layer: Optional[NormLayer] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_chirp_masses = num_chirp_masses
        self.time_domain_resnet = ResNet1D(
            in_channels=num_ifos * num_chirp_masses,
            layers=layers,
            classes=1,
            kernel_size=kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )
        mask = torch.zeros(10, num_ifos * num_chirp_masses, 3072) == 0
        for i in range(10):
            mask[i, 10*i, :] = False
            mask[i, 100+10*i, :] = False 

        self.mask = mask
    
    def forward(self, X):
        B, C, T = X.shape
        X = X.repeat(10, 1, 1)
        y = torch.repeat_interleave(self.mask, B, dim = 0)
        X = torch.where(y, X, 0)
        X = self.time_domain_resnet(X)
        return X.view(10, B, 1)
