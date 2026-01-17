import time
import torch
import logging
import torch.nn.functional as F
import torch
import torchvision
import cv2


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def pad_to_valid_size(
    img,
    valid_sizes=(640, 1024, 1280),
    pad_value=255,
    center=False,
):
    """
    img: np.ndarray (H, W, C)
    valid_sizes: الأحجام المسموحة (مربعة)
    pad_value: لون البادينج (255 = أبيض)
    center: لو True يوزع البادينج على الجنبين بدل ما يكون كله يمين وتحت
    """

    h, w = img.shape[:2]
    max_side = max(h, w)

    # اختار أصغر valid size >= max_side
    target = min([s for s in valid_sizes if s >= max_side], default=max(valid_sizes))

    pad_h = target - h
    pad_w = target - w

    if center:
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
    else:
        top, left = 0, 0
        bottom, right = pad_h, pad_w

    img_padded = cv2.copyMakeBorder(
        img,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value),
    )

    return img_padded


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    logging.info("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    logging.info("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    logging.info(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    logging.info(f"Dice score: {dice_score/len(loader)}")
    model.train()


def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()


def center_crop(
    x: torch.Tensor,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """
    Center-crop a tensor of shape (B, C, H, W) or (C, H, W).

    Args:
        x: input tensor
        target_h: target height
        target_w: target width

    Returns:
        Cropped tensor
    """
    if x.dim() == 4:
        _, _, h, w = x.shape
        y1 = (h - target_h) // 2
        x1 = (w - target_w) // 2
        result = x[:, :, y1 : y1 + target_h, x1 : x1 + target_w]
        logging.info(result.shape, "result shape")
        return result

    elif x.dim() == 3:
        _, h, w = x.shape
        y1 = (h - target_h) // 2
        x1 = (w - target_w) // 2
        return x[:, y1 : y1 + target_h, x1 : x1 + target_w]

    else:
        raise ValueError("Expected tensor of shape (B,C,H,W) or (C,H,W)")


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


def upsample_like(x, target):
    return F.interpolate(x, size=target.size()[2:], mode="bilinear")


def profile_block(name, func, *args, extra_info: str = "", **kwargs):
    out = func(*args, **kwargs)
    return out

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = func(*args, **kwargs)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    logging.info(f"{name} Time: {t1 - t0:.6f} seconds, {extra_info}")
    return out
