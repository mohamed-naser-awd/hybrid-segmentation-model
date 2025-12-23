from backbone import DarkNet, ConvBlock
from decoder import UNetDecoderClassic
from torch import nn, Tensor
from utils import profile_block


class SemanticSegmentationModel(nn.Module):
    """
    Human Semantic Segmentation Model
    """

    def __init__(self, net_type: str = "21", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.backbone = DarkNet(net_type=net_type)
        self.decoder = UNetDecoderClassic()
        self.mask_head = nn.Sequential(
            ConvBlock(32, 16, kernel_size=3, padding=1),
            ConvBlock(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, image: Tensor):
        c1, c2, c3, c4, c5 = profile_block("backbone", self.backbone, image)
        output = profile_block("decoder", self.decoder, (c1, c2, c3, c4, c5))
        masks = self.mask_head(output)
        return masks
