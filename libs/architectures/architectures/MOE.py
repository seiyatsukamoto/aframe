from typing import Literal, Optional

from architectures import Architecture
from architectures.networks import S4Model, WaveNet, Xylophone
from jaxtyping import Float
from ml4gw.nn.resnet.resnet_1d import NormLayer, ResNet1D
from ml4gw.nn.resnet.resnet_2d import ResNet2D
from torch import Tensor
import torch
from architectures.supervised import (
    SupervisedArchitecture,
    SupervisedFrequencyDomainResNet,
    SupervisedMultiModalResNet,
    SupervisedSpectrogramDomainResNet,
    SupervisedTimeDomainResNet,
)



class MOE(SupervisedArchitecture):
    def __init__(
        self,
        layers: list,
        **kwargs,
    ):
        super().__init__()
        self.classifier = torch.nn.Sequential()
        i = 0
        for in_nodes, out_nodes in zip(layers[:-1], layers[1:]):
            self.classifier.add_module(f'layer_{i}', torch.nn.Linear(in_nodes, out_nodes))
            self.classifier.add_module(f'act_{i}', torch.nn.LeakyReLU())
            i += 1
        self.classifier.add_module(f'output', torch.nn.Linear(layers[-1], 1))

    def forward(self, X):
        return self.classifier(X)