"""
In large part lifted from
https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
but with 1d convolutions and arbitrary kernel sizes, and a
default norm layer that makes more sense for most GW applications
where training-time statistics are entirely arbitrary due to
simulations.
"""

from collections.abc import Callable
from typing import Literal, Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import math
from architectures.supervised import SupervisedArchitecture

from ml4gw.nn.norm import GroupNorm1DGetter, NormLayer

class IntermediateMixing(SupervisedArchitecture):
    def __init__(
        self,
        classes: list[int],
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
        layers: list[list[int]],
        post_mix_layers = list[int],
        t_input = int,
        f_input = int,
        kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[list[Literal["stride", "dilation"]]] = None,
        norm_layer: Optional[NormLayer] = None,
    ) -> None:
        super().__init__()
        self.num_bands = len(layers)
        resnets = [ResNet1D_no_fc(num_ifos,
                            layers=layers[band],
                            kernel_size=kernel_size,
                            zero_init_residual=zero_init_residual,
                            groups=groups,
                            width_per_group=width_per_group,
                            stride_type=stride_type,
                            norm_layer=norm_layer,
                           ) for band in range(self.num_bands-1)]
        resnets.append(ResNet1D_no_fc(num_ifos * 3,
                                layers=layers[-1],
                                kernel_size=kernel_size,
                                zero_init_residual=zero_init_residual,
                                groups=groups,
                                width_per_group=width_per_group,
                                stride_type=stride_type,
                                norm_layer=norm_layer,
                               ))
        self.resnets = torch.nn.ModuleList(resnets)
        x1 = self.resnets[0](torch.zeros(1, num_ifos, t_input)).shape[-1]
        x2 = self.resnets[-1](torch.zeros(1, 3*num_ifos, f_input)).shape[-1]
        diff = x1-x2
        self.padding = (math.floor(diff/2), math.ceil(diff/2))
        self.post_mix_resnet = ResNet1D_post_mix(
            in_channels = self.num_bands*(2**(5+len(layers[0]))),
            inplanes = self.num_bands*(2**(5+len(layers[0])))//2,
            layers=post_mix_layers,
            classes=classes,
            kernel_size=kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer
        )
        self.fc = torch.nn.Linear(classes, 1)
    
    def forward(self, X):
        X_f = self.resnets[-1](X[-1])
        X_f = F.pad(X_f, self.padding)
        X = tuple(self.resnets[band](X[band]) for band in range(self.num_bands-1))
        X += (X_f,)
        X = torch.cat(X, 1)
        X = self.post_mix_resnet(X)
        return self.fc(X)

def convN(
    in_planes: int,
    out_planes: int,
    kernel_size: int = 3,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv1d:
    """1d convolution with padding"""
    if not kernel_size % 2:
        raise ValueError("Can't use even sized kernels")

    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=dilation * int(kernel_size // 2),
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    """Kernel-size 1 convolution"""
    return nn.Conv1d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


class BasicBlock(nn.Module):
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
            norm_layer = nn.BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64"
            )
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock"
            )

        # Both self.conv1 and self.downsample layers
        # downsample the input when stride != 1
        self.conv1 = convN(inplanes, planes, kernel_size, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = convN(planes, planes, kernel_size)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet1D_no_fc(nn.Module):
    """1D ResNet architecture

    Simple extension of ResNet to 1D convolutions with
    arbitrary kernel sizes to support the longer timeseries
    used in BBH detection.

    Args:
        in_channels:
            The number of channels in input tensor.
        layers:
            A list representing the number of residual
            blocks to include in each "layer" of the
            network. Total layers (e.g. 50 in ResNet50)
            is ``2 + sum(layers) * factor``, where factor
            is ``2`` for vanilla ``ResNet`` and ``3`` for
            ``BottleneckResNet``.
        kernel_size:
            The size of the convolutional kernel to
            use in all residual layers. _NOT_ the size
            of the input kernel to the network, which
            is determined at run-time.
        zero_init_residual:
            Flag indicating whether to initialize the
            weights of the batch-norm layer in each block
            to 0 so that residuals are initialized as
            identities. Can improve training results.
        groups:
            Number of convolutional groups to use in all
            layers. Grouped convolutions induce local
            connections between feature maps at subsequent
            layers rather than global. Generally won't
            need this to be >1, and wil raise an error if
            >1 when using vanilla ``ResNet``.
        width_per_group:
            Base width of each of the feature map groups,
            which is scaled up by the typical expansion
            factor at each layer of the network. Meaningless
            for vanilla ``ResNet``.
        stride_type:
            Whether to achieve downsampling on the time axis
            by strided or dilated convolutions for each layer.
            If left as ``None``, strided convolutions will be
            used at each layer. Otherwise, ``stride_type`` should
            be one element shorter than ``layers`` and indicate either
            ``stride`` or ``dilation`` for each layer after the first.
    """

    block = BasicBlock

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
    ) -> None:
        super().__init__()

        self.inplanes = 64
        self.dilation = 1

        # default to using InstanceNorm if no
        # norm layer is provided explicitly
        self._norm_layer = norm_layer or GroupNorm1DGetter()

        # TODO: should we support passing a single string
        # for simplicity here?
        if stride_type is None:
            # each element in the tuple indicates if we should replace
            # the stride with a dilated convolution instead
            stride_type = ["stride"] * (len(layers) - 1)
        if len(stride_type) != (len(layers) - 1):
            raise ValueError(
                f"'stride_type' should be None or a {len(layers) - 1}-element "
                f"tuple, got {stride_type}"
            )

        self.groups = groups
        self.base_width = width_per_group

        # start with a basic conv-bn-relu-maxpool block
        # to reduce the dimensionality before the heavy
        # lifting starts
        self.conv1 = nn.Conv1d(
            in_channels,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # now create layers of residual blocks where each
        # layer uses the same number of feature maps for
        # all its blocks (some power of 2 times 64).
        # Don't downsample along the time axis in the first
        # layer, but downsample in all the rest (either by
        # striding or dilating depending on the stride_type
        # argument)
        residual_layers = [self._make_layer(64, layers[0], kernel_size)]
        it = zip(layers[1:], stride_type, strict=True)
        for i, (num_blocks, stride) in enumerate(it):
            block_size = 64 * 2 ** (i + 1)
            layer = self._make_layer(
                block_size,
                num_blocks,
                kernel_size,
                stride=2,
                stride_type=stride,
            )
            residual_layers.append(layer)
        self.residual_layers = nn.ModuleList(residual_layers)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

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

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

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
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for layer in self.residual_layers:
            x = layer(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

class ResNet1D_post_mix(nn.Module):
    block = BasicBlock
    def __init__(
        self,
        in_channels: int,
        inplanes: int,
        layers: list[int],
        classes: int,
        kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: list[Literal["stride", "dilation"]] | None = None,
        norm_layer: NormLayer | None = None,
    ) -> None:
        super().__init__()

        self.inplanes = inplanes
        self.dilation = 1

        self._norm_layer = norm_layer or GroupNorm1DGetter()
        if stride_type is None:
            stride_type = ["stride"] * (len(layers) - 1)
        if len(stride_type) != (len(layers) - 1):
            raise ValueError(
                f"'stride_type' should be None or a {len(layers) - 1}-element "
                f"tuple, got {stride_type}"
            )

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv1d(
            in_channels,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        residual_layers = [self._make_layer(self.inplanes, layers[0], kernel_size)]
        it = zip(layers[1:], stride_type, strict=True)
        for i, (num_blocks, stride) in enumerate(it):
            block_size = self.inplanes * 2 ** (i + 1)
            layer = self._make_layer(
                block_size,
                num_blocks,
                kernel_size,
                stride=2,
                stride_type=stride,
            )
            residual_layers.append(layer)
        self.residual_layers = nn.ModuleList(residual_layers)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(block_size * self.block.expansion, classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

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

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for layer in self.residual_layers:
            x = layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)