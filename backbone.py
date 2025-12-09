from torch import nn
from utils import profile_block


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.dropout = nn.Dropout2d(0.15)
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
        x = self.dropout(x)
        return residual + x


class ResidualStage(nn.Module):
    def __init__(
        self, residual_blocks_count, in_channels, output_channels, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.layers = nn.Sequential(
            nn.Dropout(0.1),
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


class DarkNet(nn.Module):
    def __init__(self, net_type: str = "18", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
        )

        stage_block_mapper = {
            "18": {"4": 2, "5": 1},
            "21": {"4": 2, "5": 2},
            "24": {"4": 2, "5": 3},
        }

        self.c1 = ResidualStage(1, 32, 64)
        self.c2 = ResidualStage(1, 64, 128)
        self.c3 = ResidualStage(2, 128, 256)
        self.c4 = ResidualStage(stage_block_mapper[net_type]["4"], 256, 512)
        self.c5 = ResidualStage(stage_block_mapper[net_type]["5"], 512, 1024)

    def forward(self, x):
        x = profile_block("stem", self.stem, x)
        x = profile_block("c1", self.c1, x)
        x = profile_block("c2", self.c2, x)
        c3 = profile_block("c3", self.c3, x)
        c4 = profile_block("c4", self.c4, c3)
        c5 = profile_block("c5", self.c5, c4)
        return c3, c4, c5
