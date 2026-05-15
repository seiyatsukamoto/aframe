from collections.abc import Callable
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor
from architectures.supervised import SupervisedArchitecture

from ml4gw.nn.norm import GroupNorm1DGetter, NormLayer
from ml4gw.nn.resnet.resnet_1d import convN, conv1, BasicBlock, Bottleneck

class gating(nn.Module):
    def __init__(self,
                 in_channels, 
                 out_channels, 
                 num_experts,
                 norm_layer,
                 kernel_size: int = 3,):
        super().__init__()
        self.gate = nn.Sequential(conv1(in_channels, out_channels), 
                                  norm_layer(out_channels),
                                  nn.ReLU(),
                                  convN(out_channels, out_channels, kernel_size),
                                  norm_layer(out_channels),
                                  nn.ReLU(),
                                  conv1(out_channels, num_experts), 
                                  nn.AdaptiveAvgPool1d(1))
    
    def forward(self, x: Tensor) -> Tensor:
        return self.gate(x)
    
class MOE_ResNet1D(SupervisedArchitecture):
    """ 
    Multibranch Resnet with top k gating  
    Args:
        k:
            Number of experts used in foward pass
        num_experts:
            Number of total experts
        initial_layers:
            The shape of the layers before branching
        experts_layers:
            The shape of the experts 
    """
    block = BasicBlock

    def __init__(
        self,
        k: int,
        num_experts: int,
        initial_layers: list[int],
        expert_layers: list[int], 
        in_channels: int,
        kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: list[Literal["stride", "dilation"]] | None = None,
        norm_layer: NormLayer | None = None,
    ) -> None:
        super().__init__()
        
        self.top_k = k
        self.inplanes = 64
        self.dilation = 1

        self._norm_layer = norm_layer or GroupNorm1DGetter()
        if stride_type is None:
            stride_type_initial = ["stride"] * (len(initial_layers) - 1)
            stride_type_expert = ["stride"] * len(expert_layers)

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
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        residual_layers = [self._make_layer(64, initial_layers[0], kernel_size)]
        it = zip(initial_layers[1:], stride_type_initial, strict=True)
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
        
        self.gating = gating(64 * 2 ** (len(initial_layers)-1), 64 * 2 ** (len(initial_layers)-2), num_experts, self._norm_layer)
        experts = []
        for _ in range(num_experts):
            expert_i = []
            self.inplanes = 64 * 2 ** (len(initial_layers) - 1)
            for i, (num_blocks, stride) in enumerate(zip(expert_layers, stride_type_expert, strict=True)):
                block_size = 64 * 2 ** (i + len(initial_layers))
                layer = self._make_layer(
                    block_size,
                    num_blocks,
                    kernel_size,
                    stride=2,
                    stride_type=stride,
                )
                expert_i.append(layer)
            experts.append(nn.ModuleList(expert_i))
        self.experts = nn.ModuleList(experts)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        fc = []
        for _ in range(num_experts):
             fc.append(nn.Linear(block_size * self.block.expansion, 1))
        
        self.fc = nn.ModuleList(fc)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim = 1)
        
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
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
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
        x = self.maxpool(x)
        for layer in self.residual_layers:
            x = layer(x)

        gate_scores = self.gating(x).squeeze(-1)
        topk_probs, topk_indices = torch.topk(gate_scores, self.top_k, dim=1)
        topk_probs = self.softmax(topk_probs)
        
        batch_size = x.shape[0]
        out = torch.zeros(batch_size, 1, device=x.device)
        for expert_idx, (expert_layers, fc) in enumerate(zip(self.experts, self.fc)):
            mask = (topk_indices == expert_idx).any(dim=1)
            if not mask.any():
                continue

            expert_x = x[mask]
            for layer in expert_layers:
                expert_x = layer(expert_x)
            
            expert_x = self.avgpool(expert_x).squeeze(-1)
            expert_x = fc(expert_x)
            expert_weights = torch.zeros(batch_size, device=x.device)
            for k_idx in range(self.top_k):
                slot_mask = (topk_indices[:, k_idx] == expert_idx)
                expert_weights[slot_mask] = topk_probs[slot_mask, k_idx]

            weighted = expert_x * expert_weights[mask].unsqueeze(-1)
            out[mask] += weighted
        
        out = self.sigmoid(out)
        return out, self.softmax(gate_scores)
    
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
