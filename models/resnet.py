"""
CIFAR-style ResNet-18 (BasicBlock) from scratch
- 3x3 stem, stride=1
- NO maxpool
- NO pretrained weights
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SEBlock(nn.Module):
    # Squeeze-and-Excitation block
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ResNetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, se_reduction: int = 16) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SEBlock(planes * self.expansion, reduction=se_reduction)

        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes * self.expansion, stride=stride),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = F.relu(out, inplace=True)
        return out


class ResNet(nn.Module):
    """
    CIFAR-style stem:
      - conv3x3 stride=1
      - no maxpool

    For 128x128 input:
      - after stem: 128x128
      - layer2 stride2 -> 64x64
      - layer3 stride2 -> 32x32
      - layer4 stride2 -> 16x16
      - avgpool -> 1x1
    """
    def __init__(self, block: type, layers: list[int], num_classes: int = 11,
                 base_width: int = 64, dropout: float = 0.5) -> None:
        super().__init__()
        self.inplanes = base_width

        # CIFAR-style stem
        self.conv1 = nn.Conv2d(3, base_width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)

        # Stages (same 2-2-2-2 for ResNet18)
        self.layer1 = self._make_layer(block, base_width,     layers[0], stride=1)
        self.layer2 = self._make_layer(block, base_width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_width * 8, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(base_width * 8 * block.expansion, num_classes)

        # Kaiming init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, block: type, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # head
        x = self.avgpool(x).flatten(1)
        x = self.drop(x)
        x = self.fc(x)
        return x


def resnet18(num_classes: int = 11, dropout: float = 0.5) -> ResNet:
    return ResNet(ResNetBasicBlock, [2, 2, 2, 2], num_classes=num_classes, base_width=64, dropout=dropout)
