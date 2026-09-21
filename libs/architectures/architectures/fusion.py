from typing import Literal, Optional

from architectures import SupervisedArchitecture
from torch import Tensor
import torch

class MLP(SupervisedArchitecture):
    def __init__(
        self,
        shape: list[int],
    ) -> None:
        super().__init__()
        self.model = torch.nn.Sequential()
        self.model.add_module('bn_in', torch.nn.BatchNorm1d(shape[0]))
        for i in range(len(shape)-1):
            self.model.add_module(f'layer_{i}', torch.nn.Linear(shape[i], shape[i+1]))
            self.model.add_module(f'bn_{i}', torch.nn.BatchNorm1d(shape[i+1]))
            self.model.add_module(f'act_{i}', torch.nn.ReLU())
        self.model.add_module(f'output', torch.nn.Linear(shape[-1], 1))
    def forward(self, x):
        x = self.model(x)
        return x