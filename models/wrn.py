"""
SE-WideResNet-28-8 with Stochastic Depth (DropPath).

- Pre-activation residual blocks: BN -> ReLU -> Conv
- Squeeze-and-Excitation channel attention
- DropPath (stochastic depth) with linearly increasing rate
- ResNet-style stem for 224x224 input: 7x7 conv stride 2 + maxpool -> 56x56
- NO pretrained weights, Kaiming initialization from scratch
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """
    Stochastic Depth: randomly drop the entire residual branch during training.
    """
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device))
        return x * mask / keep_prob


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation: global channel attention.
    """
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, mid, bias=False)
        self.fc2 = nn.Linear(mid, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = x.view(b, c, -1).mean(dim=2)
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(w))))
        return x * w.view(b, c, 1, 1)


class WideResBlock(nn.Module):
    """
    Wide Residual Block (pre-activation):
    BN -> ReLU -> Conv -> BN -> ReLU -> Dropout -> Conv -> SE -> DropPath + Shortcut
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.se = SEBlock(out_ch, se_reduction)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(x), inplace=True)
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=True)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.se(out)
        out = self.drop_path(out)
        return out + shortcut


class WideResNet(nn.Module):
    """
    SE-WideResNet for Food-11 classification.

    Architecture (depth=28, widen_factor=8):
      - Stem: 7x7 conv stride 2 -> BN -> ReLU -> MaxPool stride 2  (224 -> 56)
      - Stage 1: 4 blocks, 128 channels, stride 1  (56x56)
      - Stage 2: 4 blocks, 256 channels, stride 2  (28x28)
      - Stage 3: 4 blocks, 512 channels, stride 2  (14x14)
      - BN -> ReLU -> Global AvgPool -> Dropout -> FC(512 -> num_classes)
    """
    def __init__(
        self,
        depth: int = 28,
        widen_factor: int = 8,
        num_classes: int = 11,
        dropout: float = 0.3,
        drop_path_rate: float = 0.1,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        assert (depth - 4) % 6 == 0, "WRN depth must be 6n+4"
        n_blocks = (depth - 4) // 6  # 4 blocks per stage for depth=28

        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        # widen_factor=8 -> channels = [16, 128, 256, 512]

        # Stem: downsample 224 -> 56
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 7, stride=2, padding=3, bias=False),  # 224->112
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),  # 112->56
        )

        # Linearly increasing drop path rate
        total_blocks = 3 * n_blocks
        dp_rates = [drop_path_rate * i / max(total_blocks - 1, 1) for i in range(total_blocks)]

        self.stage1 = self._make_stage(
            channels[0], channels[1], n_blocks, stride=1,
            dropout=dropout, dp_rates=dp_rates[0:n_blocks],
            se_reduction=se_reduction,
        )  # 56x56
        self.stage2 = self._make_stage(
            channels[1], channels[2], n_blocks, stride=2,
            dropout=dropout, dp_rates=dp_rates[n_blocks:2*n_blocks],
            se_reduction=se_reduction,
        )  # 28x28
        self.stage3 = self._make_stage(
            channels[2], channels[3], n_blocks, stride=2,
            dropout=dropout, dp_rates=dp_rates[2*n_blocks:],
            se_reduction=se_reduction,
        )  # 14x14

        self.bn_final = nn.BatchNorm2d(channels[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(channels[3], num_classes)

        self._initialize_weights()

    def _make_stage(
        self,
        in_ch: int,
        out_ch: int,
        n_blocks: int,
        stride: int,
        dropout: float,
        dp_rates: list[float],
        se_reduction: int,
    ) -> nn.Sequential:
        layers = [WideResBlock(
            in_ch, out_ch, stride=stride, dropout=dropout,
            drop_path=dp_rates[0], se_reduction=se_reduction,
        )]
        for i in range(1, n_blocks):
            layers.append(WideResBlock(
                out_ch, out_ch, stride=1, dropout=dropout,
                drop_path=dp_rates[i], se_reduction=se_reduction,
            ))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)          # [B, 16, 56, 56]
        out = self.stage1(out)      # [B, 128, 56, 56]
        out = self.stage2(out)      # [B, 256, 28, 28]
        out = self.stage3(out)      # [B, 512, 14, 14]
        out = F.relu(self.bn_final(out), inplace=True)
        out = self.avgpool(out)     # [B, 512, 1, 1]
        out = torch.flatten(out, 1) # [B, 512]
        out = self.drop(out)
        out = self.fc(out)          # [B, num_classes]
        return out


def wrn28_8(num_classes: int = 11, dropout: float = 0.3, drop_path_rate: float = 0.1) -> WideResNet:
    return WideResNet(
        depth=28, widen_factor=8, num_classes=num_classes,
        dropout=dropout, drop_path_rate=drop_path_rate,
    )
