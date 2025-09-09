from .base import Architecture
from .supervised import (
    SupervisedArchitecture,
    SupervisedFrequencyDomainResNet,
    SupervisedSpectrogramDomainResNet,
    SupervisedTimeDomainResNet
)

from .bandpass import (
    Bandpass,
    SupervisedFrequencyDomainResNetClasses
)

from .stackedresnets import MultimodalMultiband

