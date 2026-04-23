"""
CNN model architectures for CIFAR-10 classification (T7).

Five models are defined here so they can be imported both from the
Jupyter notebook and from unit tests without duplicating code.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# T7.1 – ResNet (ResNet-18 style for CIFAR-10)
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Basic residual block with two 3×3 convolutions and a skip connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        # Shortcut: 1×1 conv to match dimensions when stride > 1 or channels differ
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    """
    ResNet for CIFAR-10.

    Architecture (T7.1):
      Conv2d → BN →
      layer1 (2× ResidualBlock, 64  ch, stride 1) →
      layer2 (2× ResidualBlock, 128 ch, stride 2) →
      layer3 (2× ResidualBlock, 256 ch, stride 2) →
      layer4 (2× ResidualBlock, 512 ch, stride 2) →
      AdaptiveAvgPool → Linear(512, num_classes)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64,  64,  num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64,  128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)

        self.fc = nn.Linear(512, num_classes)

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)


# ---------------------------------------------------------------------------
# T7.2 – CNN 1
# ---------------------------------------------------------------------------

class CNN1(nn.Module):
    """
    CNN 1 (T7.2).

    Architecture:
      features1 : Conv2d(3→32) → ReLU → MaxPool(2)   [32×32 → 16×16]
      features2 : Conv2d(32→64) → ReLU → MaxPool(2)  [16×16 → 8×8]
      fc1       : Linear(64*8*8, 512) → ReLU
      fc2       : Linear(512, num_classes)
      out       : Softmax(dim=1)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc1     = nn.Linear(64 * 8 * 8, 512)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features1(x)
        out = self.features2(out)
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        return self.softmax(self.fc2(out))


# ---------------------------------------------------------------------------
# T7.3 – CNN 2
# ---------------------------------------------------------------------------

class CNN2(nn.Module):
    """
    CNN 2 (T7.3).

    Architecture:
      features1 : Conv2d(3→32) → ReLU → MaxPool(2)   [32×32 → 16×16]
      features2 : Conv2d(32→64) → ReLU → MaxPool(2)  [16×16 → 8×8]
      fc        : Linear(64*8*8, num_classes)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(64 * 8 * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features1(x)
        out = self.features2(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)


# ---------------------------------------------------------------------------
# T7.4 – CNN 3
# ---------------------------------------------------------------------------

class CNN3(nn.Module):
    """
    CNN 3 (T7.4).

    Architecture:
      features1 : Conv2d(3→32) → BN → ReLU → MaxPool(2)   [32×32 → 16×16]
      features2 : Conv2d(32→64) → BN → ReLU → MaxPool(2)  [16×16 → 8×8]
      fc1       : Linear(64*8*8, 512)
      dropout   : Dropout(0.5)
      fc2       : Linear(512, 128)
      fc3       : Linear(128, num_classes)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc1     = nn.Linear(64 * 8 * 8, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(512, 128)
        self.fc3     = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features1(x)
        out = self.features2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        return self.fc3(out)


# ---------------------------------------------------------------------------
# T7.5 – CNN 4
# ---------------------------------------------------------------------------

class CNN4(nn.Module):
    """
    CNN 4 (T7.5).

    Architecture:
      features : Conv2d(3→32) → BN → ReLU → MaxPool(2) → Dropout(0.25)
                 [32×32 → 16×16]
      fc1      : Linear(32*16*16, 256)
      fc2      : Linear(256, num_classes)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.fc1 = nn.Linear(32 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        return self.fc2(out)
