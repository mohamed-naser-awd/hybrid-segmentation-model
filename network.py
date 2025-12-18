from backbone import DarkNet
from fpn import FPN
from torch import nn, Tensor, concat
from torch.nn import functional as F
from utils import profile_block


class HybirdSegmentationAlgorithm(nn.Module):
    def __init__(
        self,
        num_classes: int,
        net_type: str = "21",
        *args,
        d_model: int = 192,
        query_count: int = 1,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        stem_out_channels = 32
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stem_out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.backbone = DarkNet(net_type=net_type)
        self.fpn = FPN(d_model=d_model)
        self.mask_head = nn.Sequential(
            nn.Conv2d(d_model + stem_out_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, image: Tensor):
        x = profile_block("stem", self.stem, image)
        c1, c2, c3, c4, c5 = profile_block("backbone", self.backbone, x)
        p1 = profile_block("fpn", self.fpn, (c1, c2, c3, c4, c5))
        p1_up = F.interpolate(p1, size=image.size()[2:], mode="nearest")
        x = concat((p1_up, x), dim=1)
        masks = self.mask_head(x)
        return masks
