import time
import torch
import logging
import torch.nn.functional as F


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
        result = x[:, :, y1:y1 + target_h, x1:x1 + target_w]
        print(result.shape, "result shape")
        return result

    elif x.dim() == 3:
        _, h, w = x.shape
        y1 = (h - target_h) // 2
        x1 = (w - target_w) // 2
        return x[:, y1:y1 + target_h, x1:x1 + target_w]

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


def profile_block(name, func, *args, **kwargs):
    out = func(*args, **kwargs)
    return out

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = func(*args, **kwargs)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    logging.info(f"{name} Time: {t1 - t0:.6f} seconds")
    return out
