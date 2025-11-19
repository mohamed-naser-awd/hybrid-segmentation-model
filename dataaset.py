import torch
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import numpy as np
import os
from PIL import Image


class P3M10kDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=640):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size
        self.images = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
        try:
            mask = Image.open(os.path.join(self.mask_dir, img_name.replace("jpg", "png"))).convert("L")
        except FileNotFoundError:
            mask = Image.open(os.path.join(self.mask_dir, img_name)).convert("L")

        img = TF.resize(img, (self.size, self.size), interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (self.size, self.size), interpolation=InterpolationMode.NEAREST)

        img = TF.to_tensor(img)
        mask = torch.from_numpy(np.array(mask)).long()  # لو multi-class

        return img, mask


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
