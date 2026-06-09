from architectures import SupervisedHeterodyneTimeDomainResNet
from utils.augmentation import HeterodyneAugmentor
from torch import Tensor
import torch


class FusedHeterodyneModel(torch.nn.Module):
    def __init__(
        self,
        preprocessor: BatchWhitener,
        arch: SupervisedHeterodyneTimeDomainResNet,
    ) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.arch = arch
    
    def forward(self, strain: Tensor) -> Tensor:
        strain = self.preprocessor(strain)
        strain = self.arch(strain)
        return strain
