import os
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import torch
from torch.functional import F


def make_negative_mask_from_image() -> torch.Tensor:
    """
    img: Tensor [C, H, W]
    return: Tensor [1, H, W] كلها zeros
    """
    H, W = 640, 640
    mask = torch.zeros((1, H, W), dtype=torch.float16)
    return mask


def pad_to_size(x: torch.Tensor, size: int = 640, pad_value: float = 0.0):
    """
    x: [C,H,W] أو [H,W]
    ترجع Tensor padded إلى (size,size) حول المركز.
    لو أكبر من size في أي بعد -> هتعمل crop مركزي (برضه بدون interpolation).
    """
    if x.dim() == 2:
        x = x.unsqueeze(0)  # [1,H,W]
        squeeze_back = True
    else:
        squeeze_back = False

    _, h, w = x.shape

    # لو أكبر: crop مركزي
    if h > size:
        top = (h - size) // 2
        x = x[:, top : top + size, :]
        h = size
    if w > size:
        left = (w - size) // 2
        x = x[:, :, left : left + size]
        w = size

    # padding
    pad_h = size - h
    pad_w = size - w

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)

    if squeeze_back:
        x = x.squeeze(0)  # [H,W]

    return x


def parse_image(img_path, size=640, channels=3):
    if channels == 3:
        img = Image.open(img_path).convert("RGB")
    elif channels == 1:
        img = Image.open(img_path).convert("L")
    else:
        raise ValueError(f"Invalid number of channels: {channels}")

    img = TF.to_tensor(img)

    if channels == 1:
        img = (img > 0.5).float()

    img = pad_to_size(img, size=size, pad_value=0.0)
    return img


def create_mammap(images, mmap_path, channels=3):
    N = len(images)
    H = W = 640
    C = channels

    mmap_arr = np.memmap(
        mmap_path,
        dtype=np.float16,
        mode="w+",
        shape=(N, C, H, W),
    )

    for i, img_path in enumerate(images):
        tensor = parse_image(img_path, size=640, channels=channels)

        mmap_arr[i] = tensor.numpy()

        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{N}")
            mmap_arr.flush()  # تأمين الكتابة على الديسك

    mmap_arr.flush()
    print("Done, memmap saved at:", mmap_path, f"for {N} images")


def parse_folder(folders: list[str], file_name, channels=3):
    image_path_list = []

    for folder in folders:
        image_path_list.extend(
            [os.path.join(folder, img_name) for img_name in os.listdir(folder)]
        )

    images = sorted(image_path_list)
    mmap_path = f"dataset/{file_name}.mmap"
    create_mammap(images, mmap_path, channels)


if __name__ == "__main__":

    # parse_folder(
    #     [
    #         "dataset/P3M-10k/train/mask",
    #         "dataset/supervisely_person_clean_2667_img/supervisely_person_clean_2667_img/masks",
    #         "dataset/places/masks",
    #         "dataset/oxford-pet/images",
    #     ],
    #     "train_640_fp16_masks",
    #     channels=1,
    # )

    # parse_folder(
    #     [
    #         "dataset/P3M-10k/train/blurred_image",
    #         "dataset/supervisely_person_clean_2667_img/supervisely_person_clean_2667_img/images",
    #         "dataset/places/train",
    #         "dataset/oxford-pet/images",
    #     ],
    #     "train_640_fp16_images",
    #     channels=3,
    # )

    parse_folder(
        ["dataset/P3M-10k/validation/P3M-500-P/blurred_image"],
        "val_640_fp16_images",
        channels=3,
    )
    parse_folder(
        ["dataset/P3M-10k/validation/P3M-500-P/mask"],
        "val_640_fp16_masks",
        channels=1,
    )
