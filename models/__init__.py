# from .unet import UNET
from .fast import UltraFastNet
from .nas import SearchableUltraFastNet
from .teacher import BiRefNetTeacher
import torch

from segmentation_models_pytorch import Unet as BaseUnet
from utils import get_device


class UNET(BaseUnet):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("encoder_name", "resnet34")
        kwargs.setdefault("encoder_weights", None)
        kwargs.setdefault("in_channels", 3)
        kwargs.setdefault("out_channels", 1)

        super().__init__(*args, **kwargs)

        self.device = get_device()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)

    def forward(self, x):
        x = (x - self.mean) / self.std
        return super().forward(x)
