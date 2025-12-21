from torch import nn, concat


class Decoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.c1_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.c2_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.c3_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.c4_conv = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.c5_upsample = nn.ConvTranspose2d(
            1024, 512, kernel_size=4, stride=2, padding=1
        )
        self.c4_upsample = nn.ConvTranspose2d(
            512, 256, kernel_size=4, stride=2, padding=1
        )
        self.c3_upsample = nn.ConvTranspose2d(
            256, 128, kernel_size=4, stride=2, padding=1
        )
        self.c2_upsample = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )
        self.c1_upsample = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.projection = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        c1, c2, c3, c4, c5 = x

        c5 = self.bottleneck(c5)

        c5_up = self.c5_upsample(c5)
        c4_u = concat((c5_up, c4), dim=1)
        c4 = self.c4_conv(c4_u)  # 512 channels

        c4_up = self.c4_upsample(c4)
        c3_u = concat((c4_up, c3), dim=1)  # 512 channels
        c3 = self.c3_conv(c3_u)  # 256 channels

        c3_up = self.c3_upsample(c3)  # 128 channels
        c2_u = concat((c3_up, c2), dim=1)  # 256 channels
        c2 = self.c2_conv(c2_u)  # 128 channels

        c2_up = self.c2_upsample(c2)  # 64 channels
        c1_u = concat((c2_up, c1), dim=1)  # 128 channels
        c1 = self.c1_conv(c1_u)  # 64 channels

        c1_up = self.c1_upsample(c1)  # 32 channels
        output = self.projection(c1_up)
        return output
