from typing import Literal, Optional

import torch
from architectures.supervised import SupervisedArchitecture
from ml4gw.nn.resnet.resnet_1d import NormLayer, ResNet1D

class Bandpass(SupervisedArchitecture):
    def __init__(
        self,
        classes: int,
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
        layers: list[int],
        kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[list[Literal["stride", "dilation"]]] = None,
        norm_layer: Optional[NormLayer] = None,
    ) -> None:
        super().__init__()
        self.resnet = ResNet1D(num_ifos,
                               layers=layers,
                               classes=classes,
                               kernel_size=kernel_size,
                               zero_init_residual=zero_init_residual,
                               groups=groups,
                               width_per_group=width_per_group,
                               stride_type=stride_type,
                               norm_layer=norm_layer,
        )
        self.fc = torch.nn.Linear(classes, 1)

    def forward(self, X):
        X = self.resnet(X)
        return self.fc(X)

class SupervisedFrequencyDomainResNetClasses(SupervisedArchitecture):
    def __init__(
        self,
        classes: int,
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
        layers: list[int],
        kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[list[Literal["stride", "dilation"]]] = None,
        norm_layer: Optional[NormLayer] = None,
    ) -> None:
        super().__init__()
        self.resnet = ResNet1D(num_ifos * 3,
                               layers=layers,
                               classes=classes,
                               kernel_size=kernel_size,
                               zero_init_residual=zero_init_residual,
                               groups=groups,
                               width_per_group=width_per_group,
                               stride_type=stride_type,
                               norm_layer=norm_layer,
        )
        self.fc = torch.nn.Linear(classes, 1)
    
    def forward(self, X):
        X = self.resnet(X)
        return self.fc(X)
