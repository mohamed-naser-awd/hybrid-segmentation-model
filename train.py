import torch
from torch import nn
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import numpy as np
from network import HybirdSegmentationAlgorithm


def load_single_sample(image_path: str, mask_path: str, image_size: int = 640):
    # ---- load image ----
    img = Image.open(image_path).convert("RGB")
    img = TF.resize(
        img,
        (image_size, image_size),
        interpolation=InterpolationMode.BILINEAR,
    )
    img_tensor = TF.to_tensor(img)  # (3, H, W) في الرينج [0,1]

    # ---- load mask ----
    mask = Image.open(mask_path).convert("L")
    mask = TF.resize(
        mask,
        (image_size, image_size),
        interpolation=InterpolationMode.NEAREST,
    )
    mask_np = np.array(mask)
    # أي بيكسل مش 0 نخليه 1
    mask_np = (mask_np > 0).astype("float32")
    mask_tensor = torch.from_numpy(mask_np)  # (H, W), قيم 0/1

    return img_tensor, mask_tensor


def train_on_single_image(
    image_path: str,
    mask_path: str,
    num_steps: int = 2000,
    lr: float = 1e-4,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 1) حمّل الصورة + الماسك ----
    image, mask = load_single_sample(image_path, mask_path, image_size=640)
    image = image.to(device)  # (3, 640, 640)
    mask = mask.to(device)    # (640, 640)

    # خليه batch = 1
    image = image.unsqueeze(0)  # (1, 3, 640, 640)
    mask = mask.unsqueeze(0)    # (1, 640, 640)

    # ---- 2) جهّز الموديل ----
    num_classes = 1  # foreground واحدة
    model = HybirdSegmentationAlgorithm(num_classes=num_classes, d_model=384).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    cls_criterion = nn.CrossEntropyLoss()
    mask_criterion = nn.BCEWithLogitsLoss()

    # ---- 3) أوفرفيت على نفس العينة ----
    for step in range(1, num_steps + 1):
        model.train()

        outputs = model(image)
        pred_logits = outputs["pred_logits"]   # (1, Q, num_classes+1) = (1, 100, 2)
        pred_masks = outputs["pred_masks"]     # (1, Q, H, W) = (1, 100, 640, 640)

        B, Q, C1 = pred_logits.shape
        _, _, H, W = pred_masks.shape

        # ===== targets =====
        # 1) الكلاسات
        target_classes = torch.full(
            (B, Q),
            fill_value=num_classes,  # index بتاع background (الـ +1)
            dtype=torch.long,
            device=device,
        )
        # نخلي query 0 يمثل الـ object بتاعنا
        target_classes[:, 0] = 0

        # 2) الماسكات
        target_masks = torch.zeros(
            (B, Q, H, W),
            dtype=torch.float32,
            device=device,
        )
        # نحط ground truth mask في query 0
        target_masks[:, 0, :, :] = mask  # (1, 640, 640) يتنسخ جوه (1, 1, 640, 640)

        # ===== losses =====
        cls_loss = cls_criterion(
            pred_logits.view(B * Q, C1),
            target_classes.view(B * Q),
        )
        mask_loss = mask_criterion(pred_masks, target_masks)

        loss = cls_loss + mask_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Step {step}/{num_steps} - total_loss={loss.item():.4f} "
                f"(cls={cls_loss.item():.4f}, mask={mask_loss.item():.4f})")

    # بعد ما يخلص التدريب نرجّع الموديل عشان تقدر تستخدمه/تسيفه
    return model


if __name__ == "__main__":
    model = train_on_single_image(
        image_path="export/image.png",
        mask_path="export/mask.png",
        num_steps=100,
        lr=1e-4,
    )
    # مثال تحفظه
    torch.save(model.state_dict(), "hybrid_seg_single_overfit.pt")

