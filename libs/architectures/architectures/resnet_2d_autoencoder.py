from collections.abc import Callable
from typing import Literal
import torch
import torch.nn as nn
from torch import Tensor
from ml4gw.nn.norm import GroupNorm2DGetter, NormLayer
from architectures.autoencoder_parts.resnet_2d_encoder import ResNet2D_encoder
from architectures.autoencoder_parts.resnet_2d_decoder import ResNet2D_decoder
from architectures.supervised import SupervisedArchitecture
import torch.nn.functional as F

def convN(
    in_planes: int,
    out_planes: int,
    kernel_size: int = 3,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    if not kernel_size % 2:
        raise ValueError("Can't use even sized kernels")

    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=dilation * int(kernel_size // 2),
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )

class ResNet2D_autoencoder(SupervisedArchitecture):
    def __init__(
        self,
        in_channels: int,
        layers: list[int],
        kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: list[Literal["stride", "dilation"]] | None = None,
        norm_layer: NormLayer | None = None,
        latent_size: int = 128,
        inplanes: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = ResNet2D_encoder(in_channels = in_channels,
                                        layers = layers,
                                        kernel_size = kernel_size,
                                        zero_init_residual = zero_init_residual,
                                        groups = groups,
                                        width_per_group = width_per_group,
                                        stride_type = stride_type,
                                        norm_layer = norm_layer,
                                        inplanes=inplanes)
        if latent_size < 0:
            latent_size = inplanes * 2 ** (len(layers)-1)
        self.compress = convN(inplanes * 2 ** (len(layers)-1), latent_size, kernel_size)
        self.decompress = convN(latent_size, inplanes * 2 ** (len(layers)-1), kernel_size)
        self.decoder = ResNet2D_decoder(in_channels = in_channels,
                                        layers = layers,
                                        kernel_size = kernel_size,
                                        zero_init_residual = zero_init_residual,
                                        groups = groups,
                                        width_per_group = width_per_group,
                                        stride_type = stride_type,
                                        norm_layer = norm_layer, 
                                        inplanes = inplanes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d | nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.compress(x)
        x = self.decompress(x)
        x = self.decoder(x)
        x = F.sigmoid(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)