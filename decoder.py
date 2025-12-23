import torch
import torch.nn as nn
import torch.nn.functional as F


def center_crop(enc: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Crop enc (B,C,H,W) to match spatial size of ref (B,*,h,w) around center.
    """
    _, _, H, W = enc.shape
    _, _, h, w = ref.shape
    if (H, W) == (h, w):
        return enc
    top = (H - h) // 2
    left = (W - w) // 2
    return enc[:, :, top : top + h, left : left + w]


class ConvBlock(nn.Module):
    """
    Classic U-Net conv block: (Conv3x3 -> ReLU) * 2
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetDecoderClassic(nn.Module):
    """
    Decoder for classic U-Net with channel plan:
        enc:  c1=64, c2=128, c3=256, c4=512, c5=1024 (bottleneck input)
        dec:  1024->512, (concat with 512 => 1024)->512,
              512->256,  (concat with 256 => 512 )->256,
              256->128,  (concat with 128 => 256 )->128,
              128->64,   (concat with 64  => 128 )->64

    Returns the last feature map (64 channels). You can add a 1x1 head outside.
    """

    def __init__(self, use_transpose: bool = True):
        super().__init__()

        self.up5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        # After concat, classic U-Net uses double conv to reduce channels:
        self.dec4 = ConvBlock(512 + 512, 512)  # (up5=512) + (c4=512) => 1024 -> 512
        self.dec3 = ConvBlock(256 + 256, 256)  # (up4=256) + (c3=256) => 512  -> 256
        self.dec2 = ConvBlock(128 + 128, 128)  # (up3=128) + (c2=128) => 256  -> 128
        self.dec1 = ConvBlock(64 + 64, 64)  # (up2=64 ) + (c1=64 ) => 128  -> 64

    def forward(
        self,
        feats: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ):
        """
        feats = (c1, c2, c3, c4, c5)
        c1 highest resolution, c5 bottleneck (lowest resolution).
        """
        c1, c2, c3, c4, c5 = feats

        x = self.up5(c5)  # 1024 -> 512 (spatial x2)
        c4c = center_crop(c4, x)  # if shapes mismatch (valid-conv style)
        x = torch.cat([x, c4c], dim=1)  # 512+512=1024
        x = self.dec4(x)  # 1024 -> 512

        x = self.up4(x)  # 512 -> 256
        c3c = center_crop(c3, x)
        x = torch.cat([x, c3c], dim=1)  # 256+256=512
        x = self.dec3(x)  # 512 -> 256

        x = self.up3(x)  # 256 -> 128
        c2c = center_crop(c2, x)
        x = torch.cat([x, c2c], dim=1)  # 128+128=256
        x = self.dec2(x)  # 256 -> 128

        x = self.up2(x)  # 128 -> 64
        c1c = center_crop(c1, x)
        x = torch.cat([x, c1c], dim=1)  # 64+64=128
        x = self.dec1(x)  # 128 -> 64

        return self.up1(x)


class UNetHead(nn.Module):
    """
    Segmentation head (logits). For binary mask use out_ch=1.
    """

    def __init__(self, in_ch: int = 64, out_ch: int = 1):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
