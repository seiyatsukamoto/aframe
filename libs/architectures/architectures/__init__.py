from .base import Architecture
from .supervised import (
    SupervisedArchitecture,
    SupervisedFrequencyDomainResNet,
    SupervisedMultiModalResNet,
    SupervisedSpectrogramDomainResNet,
    SupervisedTimeDomainResNet,
    SupervisedTimeSpectrogramResNet,
)

from .resnet_autoencoder import (
    ResNet1D_autoencoder,
    ResNet2D_autoencoder,
    MLP)