from torch import nn
from utils import profile_block


class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=0,
        stride=1,
        is_transposed=False,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        conv_class = nn.ConvTranspose2d if is_transposed else nn.Conv2d
        self.conv = conv_class(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
            stride=stride,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv1 = ConvBlock(in_channels, in_channels // 2, kernel_size=1)
        self.conv2 = ConvBlock(in_channels // 2, in_channels, padding=1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return residual + x


class ResidualStage(nn.Module):
    def __init__(
        self, residual_blocks_count, in_channels, output_channels, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.layers = nn.Sequential(
            ConvBlock(in_channels, output_channels, kernel_size=3, stride=2, padding=1),
            *[ResidualBlock(output_channels) for _ in range(residual_blocks_count)]
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class DarkNet(nn.Module):
    def __init__(self, net_type: str = "18", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.stem = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3, padding=1),
        )

        stage_block_mapper = {
            "18": {"2": 1, "3": 2, "4": 2, "5": 1},
            "21": {"2": 1, "3": 2, "4": 2, "5": 2},
            "24": {"2": 1, "3": 2, "4": 2, "5": 3},
            "34": {"2": 2, "3": 3, "4": 4, "5": 2},
            "50": {"2": 2, "3": 4, "4": 6, "5": 2},
            "53": {"2": 2, "3": 8, "4": 8, "5": 4},
        }

        c1_blocks_count = stage_block_mapper[net_type].get("1", 1)
        c2_blocks_count = stage_block_mapper[net_type].get("2", 1)
        c3_blocks_count = stage_block_mapper[net_type].get("3", 2)
        c4_blocks_count = stage_block_mapper[net_type].get("4", 2)
        c5_blocks_count = stage_block_mapper[net_type].get("5", 2)

        self.c1 = ResidualStage(c1_blocks_count, 32, 64)
        self.c2 = ResidualStage(c2_blocks_count, 64, 128)
        self.c3 = ResidualStage(c3_blocks_count, 128, 256)
        self.c4 = ResidualStage(c4_blocks_count, 256, 512)
        self.c5 = ResidualStage(c5_blocks_count, 512, 1024)

    def forward(self, image):
        x = profile_block("stem", self.stem, image)
        c1 = profile_block("c1", self.c1, x)
        c2 = profile_block("c2", self.c2, c1)
        c3 = profile_block("c3", self.c3, c2)
        c4 = profile_block("c4", self.c4, c3)
        c5 = profile_block("c5", self.c5, c4)
        return c1, c2, c3, c4, c5
