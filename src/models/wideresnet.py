"""WideResNet backbone (WRN-28-2 default), CIFAR stem."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride: int, dropout: float = 0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, 1, 1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, 1, stride, bias=False))

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.dropout(self.relu2(self.bn2(out))))
        out += self.shortcut(x)
        return out


class NetworkBlock(nn.Module):
    def __init__(self, n_layers: int, in_planes: int, out_planes: int, stride: int, dropout: float):
        super().__init__()
        layers = [BasicBlock(in_planes, out_planes, stride, dropout)]
        for _ in range(1, n_layers):
            layers.append(BasicBlock(out_planes, out_planes, 1, dropout))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth: int = 28, widen_factor: int = 2, dropout: float = 0.0, num_classes: int = 10):
        super().__init__()
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        k = widen_factor
        n1, n2, n3 = 16, 16 * k, 32 * k
        self.conv1 = nn.Conv2d(3, n1, 3, 1, 1, bias=False)
        self.block1 = NetworkBlock(n, n1, n2, 1, dropout)
        self.block2 = NetworkBlock(n, n2, n3, 2, dropout)
        self.block3 = NetworkBlock(n, n3, 64 * k, 2, dropout)
        self.bn1 = nn.BatchNorm2d(64 * k)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(64 * k, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def build_model(cfg: dict) -> nn.Module:
    m = cfg.get("model", {})
    return WideResNet(depth=m.get("depth", 28), widen_factor=m.get("widen_factor", 2),
                      dropout=m.get("dropout", 0.0), num_classes=10)
