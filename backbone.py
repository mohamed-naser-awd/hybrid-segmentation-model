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
        self, residual_blocks_count, in_channels, output_channels, stride=2, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                output_channels,
                kernel_size=3,
                stride=stride,
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


class DarkNet(nn.Module):
    def __init__(self, net_type: str = "18", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        stage_block_mapper = {
            "18": {"4": 2, "5": 1},
            "21": {"4": 2, "5": 2},
            "24": {"4": 2, "5": 3},
            "53": {"2": 2, "3": 8, "4": 8, "5": 4},  # DarkNet-53 (YOLOv3-style)
        }

        self.c1 = ResidualStage(1, 32, 64, stride=1)
        self.c2 = ResidualStage(stage_block_mapper[net_type].get("2", 1), 64, 128, stride=1)
        self.c3 = ResidualStage(stage_block_mapper[net_type].get("3", 2), 128, 256)
        self.c4 = ResidualStage(stage_block_mapper[net_type]["4"], 256, 512)
        self.c5 = ResidualStage(stage_block_mapper[net_type]["5"], 512, 1024)

    def forward(self, x):
        c1 = profile_block("c1", self.c1, x)
        c2 = profile_block("c2", self.c2, c1)
        c3 = profile_block("c3", self.c3, c2)
        c4 = profile_block("c4", self.c4, c3)
        c5 = profile_block("c5", self.c5, c4)
        return c1, c2, c3, c4, c5
