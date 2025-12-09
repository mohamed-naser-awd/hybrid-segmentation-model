import os
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from network import HybirdSegmentationAlgorithm
from dataaset import P3MMemmapDataset
from utils import profile_block


def forward_step(
    model,
    imgs,
    masks,
    cls_criterion,
    mask_criterion,
    num_classes,
    device,
):
    """
    يعمل:
    - تجهيز الداتا (imgs, masks)
    - forward للموديل
    - حساب ال losses
    ويرجع: loss, cls_loss, mask_loss
    """

    imgs = imgs.to(device, non_blocking=True)  # (B, 3, H, W)

    # ----- تجهيز الماسكات -----
    # شكل الماسك من الداتاسيت ممكن يكون:
    # (B, H, W) أو (B, 1, H, W) أو حتى (B, C, H, W)
    if masks.dim() == 4:
        if masks.size(1) == 1:
            masks = masks.squeeze(1)  # (B, H, W)
        else:
            # لو multi-channel ناخد أول قناة (أو ممكن نعمل mean)
            masks = masks[:, 0, :, :]  # (B, H, W)

    masks = masks.to(device, non_blocking=True)

    # ===== Forward =====
    pred_logits, pred_masks = model(imgs)
    # pred_logits: (B, Q, C1) , C1 = num_classes + 1
    # pred_masks : (B, Q, H, W)

    B, Q, C1 = pred_logits.shape
    _, Qm, H, W = pred_masks.shape
    assert Q == Qm, "Mismatch between Q of logits and masks!"

    # ===== Targets لكل Batch =====
    # 1) الكلاسات للـ queries
    # index الأخير = background (no-object)
    target_classes = torch.full(
        (B, Q),
        fill_value=num_classes,  # background index
        dtype=torch.long,
        device=device,
    )
    # نخلي query 0 هو الـ object الحقيقي (الوش)
    target_classes[:, 0] = 0

    # 2) الماسكات
    target_masks = torch.zeros(
        (B, Q, H, W),
        dtype=torch.float32,
        device=device,
    )
    target_masks[:, 0, :, :] = masks  # الماسك الحقيقي في query 0

    # ===== losses =====
    cls_loss = cls_criterion(
        pred_logits.view(B * Q, C1),
        target_classes.view(B * Q),
    )

    mask_loss = mask_criterion(pred_masks, target_masks)

    loss = cls_loss + mask_loss  # ممكن تعمل weights لو حابب

    return loss, cls_loss, mask_loss


def train_p3m10k(
    model: HybirdSegmentationAlgorithm,
    train_img_dir: str = "P3M-10k/train/blurred_image",
    train_mask_dir: str = "P3M-10k/train/mask",
    val_img_dir: str = "P3M-10k/validation/P3M-500-P/blurred_image",
    val_mask_dir: str = "P3M-10k/validation/P3M-500-P/mask",
    image_size: int = 640,
    num_epochs: int = 50,
    batch_size: int = 20,
    lr: float = 1e-4,
    num_workers: int = 4,
    save_path: str = "hybrid_seg_p3m10k.pt",
):
    # ==========================
    # 0) الجهاز
    # ==========================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = model.to(device)

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass

    # ==========================
    # 1) الـ Dataset & DataLoader
    # ==========================
    train_dataset = P3MMemmapDataset(
        mmap_path="dataset/p3m_train_blurred_640_fp16.mmap",
        mask_mmap_path="dataset/p3m_train_blurred_640_masks_fp16.mmap",
        N=9421,
    )

    val_dataset = P3MMemmapDataset(
        mmap_path="dataset/p3m_val_blurred_640_fp16.mmap",
        mask_mmap_path="dataset/p3m_val_blurred_640_masks_fp16.mmap",
        N=500,
    )

    # خليتها False عشان موضوع الـ RAM اللي عندك
    pin = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    # ==========================
    # 2) الموديل + اللوس + الأوبتيميزر + AMP
    # ==========================
    # عندنا كلاس واحد (foreground) + background كـ "no-object"
    num_classes = 1

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # classification: CrossEntropy على الـ queries
    cls_criterion = nn.CrossEntropyLoss()
    # masks: BCEWithLogits على الماسكات
    mask_criterion = nn.BCEWithLogitsLoss()

    scaler = GradScaler()

    best_val_loss = float("inf")

    # ==========================
    # 3) لوب التدريب
    # ==========================
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        # -------- Train --------
        model.train()
        running_train_loss = 0.0
        running_train_cls = 0.0
        running_train_mask = 0.0

        print(f"Training on {len(train_loader)} batches")

        for step, (imgs, masks) in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)

            # ----- Forward + loss (AMP) -----
            with autocast():
                loss, cls_loss, mask_loss = profile_block(
                    "forward step",
                    forward_step,
                    model,
                    imgs,
                    masks,
                    cls_criterion,
                    mask_criterion,
                    num_classes,
                    device,
                )

            # ----- Backward + optimizer step (scaled) -----
            profile_block("backward", scaler.scale(loss).backward)
            profile_block("optimizer step", scaler.step, optimizer)
            scaler.update()

            running_train_loss += loss.item()
            running_train_cls += cls_loss.item()
            running_train_mask += mask_loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        avg_train_cls = running_train_cls / len(train_loader)
        avg_train_mask = running_train_mask / len(train_loader)

        end_time = time.time()

        print(
            f"[Epoch {epoch}/{num_epochs}] "
            f"Train: total={avg_train_loss:.4f}, "
            f"cls={avg_train_cls:.4f}, mask={avg_train_mask:.4f}, "
            f"Time: {end_time - start_time:.6f} seconds"
        )

        # -------- Validation --------
        model.eval()
        val_loss = 0.0
        val_cls_loss = 0.0
        val_mask_loss = 0.0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)

                if masks.dim() == 4:
                    if masks.size(1) == 1:
                        masks = masks.squeeze(1)
                    else:
                        masks = masks[:, 0, :, :]

                masks = masks.to(device).float()
                if masks.max() > 1.0:
                    masks = (masks > 0).float()

                pred_logits, pred_masks = model(imgs)

                B, Q, C1 = pred_logits.shape
                _, Qm, H, W = pred_masks.shape
                assert Q == Qm

                target_classes = torch.full(
                    (B, Q),
                    fill_value=num_classes,
                    dtype=torch.long,
                    device=device,
                )
                target_classes[:, 0] = 0

                target_masks = torch.zeros(
                    (B, Q, H, W),
                    dtype=torch.float32,
                    device=device,
                )
                target_masks[:, 0, :, :] = masks

                cls_l = cls_criterion(
                    pred_logits.view(B * Q, C1),
                    target_classes.view(B * Q),
                )
                mask_l = mask_criterion(pred_masks, target_masks)
                l = cls_l + mask_l

                val_loss += l.item()
                val_cls_loss += cls_l.item()
                val_mask_loss += mask_l.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_cls = val_cls_loss / len(val_loader)
        avg_val_mask = val_mask_loss / len(val_loader)

        print(
            f"[Epoch {epoch}/{num_epochs}] "
            f"Val:   total={avg_val_loss:.4f}, "
            f"cls={avg_val_cls:.4f}, mask={avg_val_mask:.4f}"
        )

        # -------- حفظ أحسن موديل --------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(
                f"--> Saved best model so far to '{save_path}' "
                f"(val_loss={best_val_loss:.4f})"
            )

    print("Training finished!")
    return model


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    model = HybirdSegmentationAlgorithm(num_classes=1, net_type="18")
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    train_p3m10k(model, save_path="hybrid_seg_p3m10k_dark18.pt")
