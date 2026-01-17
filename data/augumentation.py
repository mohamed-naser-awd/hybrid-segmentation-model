import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import random


def color_augment(img):
    """
    img: Tensor [C, H, W]
    """
    brightness = random.uniform(0.7, 1.3)
    contrast = random.uniform(0.7, 1.3)
    saturation = random.uniform(0.8, 1.2)
    
    img = TF.adjust_brightness(img, brightness)
    img = TF.adjust_contrast(img, contrast)
    img = TF.adjust_saturation(img, saturation)
    return img


def horizontal_flip(img, mask):
    """
    img: Tensor [C, H, W]
    mask: Tensor [1, H, W]
    """
    if random.random() > 0.5:
        img = torch.flip(img, dims=[-1])
        mask = torch.flip(mask, dims=[-1])
    return img, mask


def random_crop(img, mask, scale=(0.7, 1.0)):
    """
    img: Tensor [C, H, W]
    mask: Tensor [1, H, W]
    """
    _, h, w = img.shape
    
    crop_scale = random.uniform(scale[0], scale[1])
    crop_h = int(h * crop_scale)
    crop_w = int(w * crop_scale)
    
    top = random.randint(0, h - crop_h)
    left = random.randint(0, w - crop_w)
    
    img = img[:, top:top+crop_h, left:left+crop_w]
    mask = mask[:, top:top+crop_h, left:left+crop_w]
    
    # resize back
    img = F.interpolate(img.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0)
    mask = F.interpolate(mask.unsqueeze(0), size=(h, w), mode='nearest').squeeze(0)
    
    return img, mask