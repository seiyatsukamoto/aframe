from typing import Optional
import torch
from architectures.supervised import SupervisedArchitecture
from ml4gw.nn.resnet.resnet_1d import NormLayer, ResNet1D

class MultiModalPsd(SupervisedArchitecture):
    """
    MultiModal embedding network that embeds time, frequency, and PSD data.
    """

    def __init__(
        self,
        num_ifos: int,
        low_freq_classes: int,
        high_freq_classes: int,
        freq_classes: int,
        low_freq_layers: list[int],
        high_freq_layers: list[int],
        freq_layers: list[int],
        freq_kernel_size=3,
        zero_init_residual=True,
        groups=1,
        width_per_group=64,
        norm_layer: Optional[NormLayer] = None,
        **kwargs
    ):
        super().__init__()

        self.low_resnet = ResNet1D(
            in_channels=num_ifos,
            layers=low_freq_layers,
            classes=low_freq_classes,
            kernel_size=3,
        )

        self.high_resnet = ResNet1D(
            in_channels=num_ifos,
            layers=high_freq_layers,
            classes=high_freq_classes,
            kernel_size=3,
        )

        self.freq_psd_resnet = ResNet1D(
            in_channels=int(num_ifos * 3),
            layers=freq_layers,
            classes=freq_classes,
            kernel_size=freq_kernel_size,
        )

        self.classifier = torch.nn.Linear(
            low_freq_classes + high_freq_classes + freq_classes, 1
        )

    def forward(self, X_low, X_high, X_fft):
        low_out = self.low_resnet(X_low)
        high_out = self.high_resnet(X_high)
        freq_out = self.freq_psd_resnet(X_fft)
        x = torch.cat([low_out, high_out, freq_out], dim=-1)
        return self.classifier(x)

