from torch import nn
from utils import profile_block


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.conv2 = nn.Conv2d(
            in_channels // 2,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return residual + x


class ResidualStage(nn.Module):
    def __init__(
        self, residual_blocks_count, in_channels, output_channels, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                output_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.1, inplace=True),
            *[ResidualBlock(output_channels) for _ in range(residual_blocks_count)]
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class DarkNet53(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.c1 = ResidualStage(1, 32, 64)
        self.c2 = ResidualStage(1, 64, 128)
        self.c3 = ResidualStage(2, 128, 256)
        self.c4 = ResidualStage(2, 256, 512)
        self.c5 = ResidualStage(1, 512, 1024)

    def forward(self, x):
        x = self.stem(x)
        x = self.c1(x)
        x = self.c2(x)
        c3 = self.c3(x)
        c4 = self.c4(c3)
        c5 = self.c5(c4)
        return c3, c4, c5
