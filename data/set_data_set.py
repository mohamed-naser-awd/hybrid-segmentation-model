import pathlib
BASE_DIR = pathlib.Path(__file__).parent.parent

import sys
sys.path.insert(0, str(BASE_DIR))

import os
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import torch
from tqdm import tqdm
from models import BiRefNetTeacher


# Option 1: BiRefNet (SOTA for high-resolution segmentation)
# pip install birefnet

# Option 2: InSPyReNet
# pip install transparent-background

# Option 3: DIS (Dichotomous Image Segmentation)
# Great for fine details


def create_memmap_with_teacher(
    images: list[str],
    mmap_path: str,
    teacher,
    size: int = 640,
):
    """Generate soft labels using teacher model and save to memmap"""
    N = len(images)
    H = W = size
    C = 1

    mmap_arr = np.memmap(
        mmap_path,
        dtype=np.float16,
        mode="w+",
        shape=(N, C, H, W),
    )

    for i, img_path in enumerate(tqdm(images, desc="Generating soft labels")):
        try:
            img = Image.open(img_path).convert("RGB")
            soft_mask = teacher.predict_soft_mask(img, size=size)
            mmap_arr[i] = soft_mask.numpy()
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            mmap_arr[i] = np.zeros((C, H, W), dtype=np.float16)

        if (i + 1) % 100 == 0:
            mmap_arr.flush()

    mmap_arr.flush()
    print(f"Done! Soft labels saved at: {mmap_path} for {N} images")


def create_memmap_images(
    images: list[str], mmap_path: str, channels: int = 3, size: int = 640
):
    """Process and save images to memmap"""
    N = len(images)
    H = W = size
    C = channels

    mmap_arr = np.memmap(
        mmap_path,
        dtype=np.float16,
        mode="w+",
        shape=(N, C, H, W),
    )

    for i, img_path in enumerate(tqdm(images, desc="Processing images")):
        try:
            if channels == 3:
                img = Image.open(img_path).convert("RGB")
            else:
                img = Image.open(img_path).convert("L")

            img = img.resize((size, size), Image.BILINEAR)
            img_tensor = TF.to_tensor(img)
            mmap_arr[i] = img_tensor.numpy().astype(np.float16)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            mmap_arr[i] = np.zeros((C, H, W), dtype=np.float16)

        if (i + 1) % 100 == 0:
            mmap_arr.flush()

    mmap_arr.flush()
    print(f"Done! Images saved at: {mmap_path} for {N} images")


def get_image_paths(folders: list[str]) -> list[str]:
    """Collect and sort all image paths from folders"""
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_paths = []

    for folder in folders:
        for img_name in os.listdir(folder):
            if os.path.splitext(img_name)[1].lower() in image_extensions:
                image_paths.append(os.path.join(folder, img_name))

    return sorted(image_paths)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    SIZE = 640

    # ============================================================
    # Choose your teacher model (uncomment one):
    # ============================================================

    teacher = BiRefNetTeacher(device=device)

    positive_folders = [
        BASE_DIR / "dataset" / "P3M-10k" / "train" / "blurred_image",
        BASE_DIR / "dataset" / "supervisely_person_clean_2667_img" / "supervisely_person_clean_2667_img" / "images",
    ]

    positive_images = get_image_paths(positive_folders)
    print(f"Found {len(positive_images)} positive images")

    # Generate soft labels
    create_memmap_with_teacher(
        images=positive_images,
        mmap_path=BASE_DIR / "dataset" / "train_640_fp16_soft_masks.mmap",
        teacher=teacher,
        size=SIZE,
    )

    # Save images
    create_memmap_images(
        images=positive_images,
        mmap_path=BASE_DIR / "dataset" / "train_640_fp16_images.mmap",
        channels=3,
        size=SIZE,
    )

    print("\n✓ Dataset processing complete!")
    print(f"  Positive: {len(positive_images)} samples")
