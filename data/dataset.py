import torch
from torch.utils.data import Dataset
import numpy as np
from data.augumentation import color_augment, horizontal_flip, random_crop


class P3MMemmapDataset(Dataset):
    def __init__(
        self, mmap_path, mask_mmap_path, N=None, training=True, height=640, width=640
    ):
        self.mmap_path = mmap_path
        self.mask_mmap_path = mask_mmap_path
        self.N = N
        self.training = training
        self.height = height
        self.width = width

        self.imgs = None
        self.masks = None

    def _init_memmap(self):
        if self.imgs is None:
            self.imgs = np.memmap(
                self.mmap_path,
                dtype="float16",
                mode="r",
                shape=(self.N, 3, self.height, self.width),
            )
            self.masks = np.memmap(
                self.mask_mmap_path,
                dtype="float16",
                mode="r",
                shape=(self.N, 1, 640, 640),
            )

    def __getitem__(self, idx):
        self._init_memmap()

        img = torch.from_numpy(self.imgs[idx].copy()).float()
        mask = torch.from_numpy(self.masks[idx].copy()).float()

        if self.training:
            img = color_augment(img)
            img, mask = horizontal_flip(img, mask)
            img, mask = random_crop(img, mask)

        return img, mask

    def __len__(self):
        return self.N
