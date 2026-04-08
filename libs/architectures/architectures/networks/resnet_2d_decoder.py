"""
In large part lifted from
https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
but with arbitrary kernel sizes
"""

from collections.abc import Callable
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from ml4gw.nn.norm import GroupNorm2DGetter, NormLayer


def convN(
    in_planes: int,
    out_planes: int,
    kernel_size: int = 3,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    """2d convolution with padding"""
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

class UpBasicBlock(nn.Module):
    """Defines the structure of the blocks used to build the ResNet"""

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64"
            )
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock"
            )
        if inplanes != planes:
            self.up = nn.ConvTranspose2d(inplanes, planes, kernel_size=2, stride=2)
        self.conv1 = convN(planes, planes, kernel_size)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = convN(planes, planes, kernel_size)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        if hasattr(self, "up"):
            x = self.up(x)
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)

        return x


class ResNet2D_decoder(nn.Module):
    block = UpBasicBlock

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
        inplanes: int = 64,
    ) -> None:
        super().__init__()
        self.inplanes = inplanes * 2 ** (len(layers)-1)
        self._norm_layer = norm_layer or GroupNorm2DGetter()
        self.dilation = 1
        if stride_type is None:
            stride_type = ["stride"] * (len(layers) - 1)
        if len(stride_type) != (len(layers) - 1):
            raise ValueError(
                f"'stride_type' should be None or a {len(layers) - 1}-element "
                f"tuple, got {stride_type}"
            )
        self.groups = groups
        self.base_width = width_per_group
        it = zip(layers[:0:-1], stride_type, strict=True) #read in reverse and omit first element
        residual_layers = []
        for i, (num_blocks, stride) in enumerate(it):
            block_size = inplanes * 2 ** (len(layers)-i-1)
            if i == 0:
                layer = self._make_initial_layer(block_size,
                                                 num_blocks,
                                                 kernel_size,
                                                 stride=2,
                                                 stride_type=stride)
            else:
                layer = self._make_layer(block_size,
                                         num_blocks,
                                         kernel_size,
                                         stride=2,
                                         stride_type=stride)
            residual_layers.append(layer)
        residual_layers.append(self._make_layer(inplanes, layers[0], kernel_size))
        self.residual_layers = nn.ModuleList(residual_layers)
        self.conv1 = nn.ConvTranspose2d(in_channels=inplanes,
                                        out_channels=inplanes, 
                                        kernel_size=7,
                                        stride=2,
                                        padding=3,
                                        output_padding=1,
                                        bias=False,
                                       )
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(in_channels=inplanes,
                                        out_channels=in_channels, 
                                        kernel_size=7,
                                        stride=2,
                                        padding=3,
                                        output_padding=1,
                                        bias=False,
                                       )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d | nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, UpBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    def _make_initial_layer(
        self,
        planes: int,
        blocks: int,
        kernel_size: int = 3,
        stride: int = 1,
        stride_type: Literal["stride", "dilation"] = "stride",
    ) -> nn.Sequential:
        block = self.block
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if stride_type == "dilation":
            self.dilation *= stride
            stride = 1
        elif stride_type != "stride":
            raise ValueError("Unknown stride type {stride}")

        #if stride != 1 or self.inplanes != planes * block.expansion: TO DO
        #    downsample = nn.Sequential(
        #        conv1(self.inplanes, planes * block.expansion, stride),
        #        norm_layer(planes * block.expansion),
        #    )

        layers = []
        self.inplanes = planes * block.expansion
        for _ in range(blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    kernel_size,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)
    
    def _make_layer(
        self,
        planes: int,
        blocks: int,
        kernel_size: int = 3,
        stride: int = 1,
        stride_type: Literal["stride", "dilation"] = "stride",
    ) -> nn.Sequential:
        block = self.block
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if stride_type == "dilation":
            self.dilation *= stride
            stride = 1
        elif stride_type != "stride":
            raise ValueError("Unknown stride type {stride}")

        #if stride != 1 or self.inplanes != planes * block.expansion: TO DO
        #    downsample = nn.Sequential(
        #        conv1(self.inplanes, planes * block.expansion, stride),
        #        norm_layer(planes * block.expansion),
        #    )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                kernel_size,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    kernel_size,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        for layer in self.residual_layers:
            x = layer(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)