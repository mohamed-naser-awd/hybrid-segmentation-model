from backbone import DarkNet
from decoder import Decoder
from torch import nn, Tensor
from torch.nn import functional as F
from utils import profile_block


class HybirdSegmentationAlgorithm(nn.Module):
    def __init__(
        self,
        num_classes: int,
        net_type: str = "21",
        *args,
        query_count: int = 1,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.backbone = DarkNet(net_type=net_type)
        self.decoder = Decoder()
        self.mask_head = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Relu(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Relu(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Relu(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, image: Tensor):
        c1, c2, c3, c4, c5 = profile_block("backbone", self.backbone, image)
        output = profile_block("decoder", self.decoder, (c1, c2, c3, c4, c5))
        masks = self.mask_head(output)
        return masks
