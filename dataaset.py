import torch
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import numpy as np


class PetSegmentationDataset(Dataset):
    """
    داتاسيت بسيطة فوق Oxford-IIIT Pet:
    - بتحول الماسك لتريناري → بيناري (pet vs background)
    - بتعمل resize للـ image والـ mask لـ 640x640 عشان تناسب الموديل بتاعك
    """

    def __init__(self, root: str, image_size: int = 640, split: str = "trainval"):
        super().__init__()
        self.image_size = image_size

        # target_types="segmentation" عشان يديك الماسك مش الكلاس
        self.base_dataset = OxfordIIITPet(
            root=root,
            split=split,
            target_types="segmentation",
            download=True,
        )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, seg = self.base_dataset[idx]   # image: PIL, seg: PIL mask (trimap)

        # --- resize image ---
        image = TF.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=InterpolationMode.BILINEAR,
        )
        image = TF.to_tensor(image)  # (3, H, W), [0,1]

        # --- convert & resize mask ---
        seg = TF.resize(
            seg,
            (self.image_size, self.image_size),
            interpolation=InterpolationMode.NEAREST,
        )
        seg_np = np.array(seg)

        # في الداتاسيت دي الماسك trimap (background / pet / border):contentReference[oaicite:1]{index=1}
        # هنخلي أي حاجة مش background = 1
        # غالبًا background = 1, pet = 2, border = 3
        mask_np = (seg_np != 1).astype("float32")

        mask = torch.from_numpy(mask_np)  # (H, W), float32 {0,1}

        return image, mask
