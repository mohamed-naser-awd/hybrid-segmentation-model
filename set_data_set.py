import os
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


def parse_image(img_path, size=640, channels=3):
    if channels == 3:
        img = Image.open(img_path).convert("RGB")
    elif channels == 1:
        img = Image.open(img_path).convert("L")
    else:
        raise ValueError(f"Invalid number of channels: {channels}")

    img = TF.resize(img, (size, size), interpolation=InterpolationMode.BILINEAR)
    img = TF.to_tensor(img)
    return img


def parse_folder(img_dir, file_name, channels=3):
    images = sorted(os.listdir(img_dir))
    N = len(images)
    H = W = 640
    C = channels

    # نعمل ملف memmap على الهارد
    mmap_path = f"dataset/{file_name}.mmap"
    mmap_arr = np.memmap(
        mmap_path,
        dtype=np.float16,
        mode="w+",
        shape=(N, C, H, W),
    )

    for i, img_name in enumerate(images):
        img_path = os.path.join(img_dir, img_name)
        tensor = parse_image(img_path, size=640, channels=channels)

        # نحول numpy ونكتبه في السلايس بتاعه
        mmap_arr[i] = tensor.numpy()

        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{N}")
            mmap_arr.flush()  # تأمين الكتابة على الديسك

    mmap_arr.flush()
    print("Done, memmap saved at:", mmap_path)


if __name__ == "__main__":
    parse_folder(
        "P3M-10k/validation/P3M-500-P/blurred_image", "p3m_val_blurred_640_fp16"
    )
    parse_folder("P3M-10k/validation/P3M-500-P/mask", "p3m_val_blurred_640_masks_fp16", channels=1)
    parse_folder("P3M-10k/train/blurred_image", "p3m_train_blurred_640_fp16")
    parse_folder("P3M-10k/train/mask", "p3m_train_blurred_640_masks_fp16", channels=1)
