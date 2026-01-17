import torch
import torch.nn as nn
from utils import profile_block


class FastConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        padding = padding if kernel_size == 3 else 0

        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            stride=stride,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels)
        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = self.relu(self.bn(self.conv(x)))
        return self.relu_out(self.bn_out(self.conv_out(hx)))


class FastNestBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(FastConvBlock(3, 32), FastConvBlock(32, 32))

        self.stage1 = nn.Sequential(
            FastConvBlock(32, 32),
            FastConvBlock(32, 64, stride=2),
        )

        self.stage2 = nn.Sequential(
            FastConvBlock(64, 64),
            FastConvBlock(64, 128, stride=2),
        )

        self.stage3 = nn.Sequential(
            FastConvBlock(128, 128),
            FastConvBlock(128, 256, stride=2),
        )

        self.stage4 = nn.Sequential(
            FastConvBlock(256, 256),
            FastConvBlock(256, 512, stride=2),
        )

        self.downsample1 = FastConvBlock(32, 64, kernel_size=1, stride=2)
        self.downsample2 = FastConvBlock(64, 128, kernel_size=1, stride=2)
        self.downsample3 = FastConvBlock(128, 256, kernel_size=1, stride=2)
        self.downsample4 = FastConvBlock(256, 512, kernel_size=1, stride=2)

    def forward(self, x):
        stem_out = self.stem(x)
        stem_down = self.downsample1(stem_out)
        s1 = self.stage1(stem_out)

        s1 = s1 + stem_down
        s2 = self.stage2(s1) + self.downsample2(s1)
        s3 = self.stage3(s2) + self.downsample3(s2)
        s4 = self.stage4(s3) + self.downsample4(s3)
        return stem_out, s1, s2, s3, s4


class FastNestDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsampler = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.s4_translator = FastConvBlock(512, 256)
        self.s3_translator = FastConvBlock(256, 128)
        self.s2_translator = FastConvBlock(128, 64)
        self.s1_translator = FastConvBlock(64, 32)

        self.hx4_translator = FastConvBlock(512, 256)
        self.hx3_translator = FastConvBlock(256, 128)
        self.hx2_translator = FastConvBlock(128, 64)
        self.hx1_translator = FastConvBlock(64, 32)

    def forward(self, x):
        s0, s1, s2, s3, s4 = x  # 32, 64, 128, 256, 512

        upsampled_x = self.upsampler(s4)
        hx = self.s4_translator(upsampled_x)
        hx = torch.cat([hx, s3], dim=1)
        hx = self.hx4_translator(hx)

        upsampled_x = self.upsampler(hx)
        hx = self.s3_translator(upsampled_x)
        hx = torch.cat([hx, s2], dim=1)
        hx = self.hx3_translator(hx)

        upsampled_x = self.upsampler(hx)
        hx = self.s2_translator(upsampled_x)
        hx = torch.cat([hx, s1], dim=1)
        hx = self.hx2_translator(hx)

        upsampled_x = self.upsampler(hx)
        hx = self.s1_translator(upsampled_x)
        hx = torch.cat([hx, s0], dim=1)
        hx = self.hx1_translator(hx)

        return hx


class FastNestDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # Use addition instead of concatenation
        self.dec4 = FastConvBlock(512, 256)
        self.dec3 = FastConvBlock(256, 128)
        self.dec2 = FastConvBlock(128, 64)
        self.dec1 = FastConvBlock(64, 32)

        # Skip connection projectors (match channels for addition)
        self.skip3 = nn.Conv2d(256, 256, 1, bias=False)
        self.skip2 = nn.Conv2d(128, 128, 1, bias=False)
        self.skip1 = nn.Conv2d(64, 64, 1, bias=False)
        self.skip0 = nn.Conv2d(32, 32, 1, bias=False)

    def forward(self, x):
        s0, s1, s2, s3, s4 = x

        # Use addition instead of cat (much faster, less memory)
        hx = self.dec4(self.up(s4)) + self.skip3(s3)
        hx = self.dec3(self.up(hx)) + self.skip2(s2)
        hx = self.dec2(self.up(hx)) + self.skip1(s1)
        hx = self.dec1(self.up(hx)) + self.skip0(s0)

        return hx


class UltraFastNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = FastNestBackbone()
        self.decoder = FastNestDecoder()
        self.head = nn.Sequential(
            FastConvBlock(32, 32),
            FastConvBlock(32, 16),
            FastConvBlock(16, 8),
            nn.Conv2d(8, 1, kernel_size=1),
        )

    def forward(self, x):
        s0, s1, s2, s3, s4 = profile_block("backbone", self.backbone, x)
        x = profile_block("decoder", self.decoder, (s0, s1, s2, s3, s4))
        return profile_block("head", self.head, x)
