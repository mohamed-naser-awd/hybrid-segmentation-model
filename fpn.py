from torch import nn, concat


class FPN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.c3_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.c4_conv = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.c5_conv = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        c3, c4, c5 = x
        p5 = self.c5_conv(c5)
        p5_up = self.upsample(p5)
        c4_u = concat((p5_up, c4), dim=1)
        p4 = self.c4_conv(c4_u)
        p4_up = self.upsample(p4)
        c3_u = concat((p4_up, c3), dim=1)
        p3 = self.c3_conv(c3_u)
        return p3, p4, p5
