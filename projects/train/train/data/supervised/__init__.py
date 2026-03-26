from .time_frequency_domain import (
    FrequencyDomainSupervisedAframeDataset,
    SpectrogramDomainSupervisedAframeDataset,
    TimeSpectrogramDomainSupervisedAframeDataset,
)
from .multimodal import MultiModalSupervisedAframeDataset
from .supervised import SupervisedAframeDataset
from .time_domain import TimeDomainSupervisedAframeDataset
from .time_domain_autoencoder import TimeDomainAutoencoderAframeDataset
from .spectrogram_autoencoder import SpectrogramAutoencoderAframeDataset
from .autoencoder_MLP import MLPAframeDataset
