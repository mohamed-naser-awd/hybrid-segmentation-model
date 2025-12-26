from backbone import ConvBlock
from decoder import UNetDecoderClassic
from torch import nn, Tensor
from torch.nn import functional as F
from utils import profile_block


import timm
import torch


class SemanticSegmentationModel(nn.Module):
    """
    Human Semantic Segmentation Model
    """

    def __init__(self, *args, d_model=512, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.backbone = timm.create_model(
            "darknet53",
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3, 4, 5),
        )
        self.backbone_training_disabled = False

        self.decoder = UNetDecoderClassic()
        self.mask_projection = nn.Sequential(
            ConvBlock(32, 16, kernel_size=1),
            ConvBlock(16, 8, kernel_size=1),
            nn.Conv2d(8, 1, kernel_size=1),
        )

    def forward(self, image: Tensor):
        if self.backbone_training_disabled:
            with torch.no_grad():
                backbone_output = profile_block("backbone", self.backbone, image)
        else:
            backbone_output = profile_block("backbone", self.backbone, image)

        decoder_output = profile_block("decoder", self.decoder, backbone_output)
        mask = profile_block("mask_projection", self.mask_projection, decoder_output)
        return mask

    def flatten_tensor(self, tensor: Tensor):
        return tensor.flatten(1)

    def disable_backbone_training(self):
        self.backbone_training_disabled = True
        self.backbone.eval()

        for p in self.backbone.parameters(recurse=True):
            p.requires_grad = False
