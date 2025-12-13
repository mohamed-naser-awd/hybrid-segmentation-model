import logging

logging.basicConfig(
    filename="app.logs",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.info("Training started")


import torch
from torch import nn
from torch.utils.data import DataLoader

from network import HybirdSegmentationAlgorithm
from dataset import P3MMemmapDataset
from utils import profile_block
import time
from torch.cuda.amp import autocast, GradScaler

# ==========================
# Dice Loss + وزنها
# ==========================

DICE_WEIGHT = 1.0  # تقدر تجرب 0.5 أو 2.0 وتشوف تأثيرها


def dice_loss(pred, target, eps=1e-6):
    """
    pred: logits من الموديل (B, Q, H, W)
    target: ماسكات حقيقية (B, Q, H, W) بقيم 0/1 أو 0..1
    """
    # نحول من logits إلى احتمالات
    pred = torch.sigmoid(pred)

    # نفرد الـ batch والـ channels في بعد واحد
    pred = pred.flatten(1)
    target = target.flatten(1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)

    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def train_step(
    model,
    imgs,
    masks,
    optimizer,
    cls_criterion,
    mask_criterion,
    num_classes,
    device,
    scaler: GradScaler,
):
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

    optimizer.zero_grad(set_to_none=True)

    # ===== forward + loss تحت autocast =====
    with autocast(dtype=torch.float16, enabled=(device == "cuda")):
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
        # نخلي query 0 هو الـ object الحقيقي (الوش/الشخص)
        target_classes[:, 0] = 0

        # 2) الماسكات
        target_masks = torch.zeros(
            (B, Q, H, W),
            dtype=torch.float32,
            device=device,
        )
        target_masks[:, 0, :, :] = masks  # الماسك الحقيقي في query 0

        # ===== losses =====
        # loss للكلاسات (queries)
        cls_loss = cls_criterion(
            pred_logits.view(B * Q, C1),
            target_classes.view(B * Q),
        )

        # BCE + Dice على الماسكات
        bce_loss = mask_criterion(pred_masks, target_masks)
        d_loss = dice_loss(pred_masks, target_masks)
        mask_loss = bce_loss + DICE_WEIGHT * d_loss

        loss = cls_loss + mask_loss

    # ===== backward + optimizer مع GradScaler =====
    def backward_fn():
        scaler.scale(loss).backward()

    profile_block("backward", backward_fn)

    def step_fn():
        scaler.step(optimizer)
        scaler.update()

    profile_block("optimizer step", step_fn)

    return loss, cls_loss, mask_loss


def train_p3m10k(
    model: HybirdSegmentationAlgorithm,
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
    logging.info(f"Using device: {device}")
    model = model.to(device)

    # ==========================
    # 1) الـ Dataset & DataLoader
    # ==========================
    train_dataset = P3MMemmapDataset(
        mmap_path="dataset/train_640_fp16_images.mmap",
        mask_mmap_path="dataset/train_640_fp16_masks.mmap",
        N=12088,
    )

    val_dataset = P3MMemmapDataset(
        mmap_path="dataset/val_640_fp16_images.mmap",
        mask_mmap_path="dataset/val_640_fp16_masks.mmap",
        N=500,
    )

    pin = True if device == "cuda" else False

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
    # 2) الموديل + اللوس + الأوبتيميزر + GradScaler
    # ==========================
    # عندنا كلاس واحد (foreground) + background كـ "no-object"
    num_classes = 1

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # classification: CrossEntropy على الـ queries
    cls_criterion = nn.CrossEntropyLoss()
    # masks: BCEWithLogits على الماسكات (هنضيف له Dice)
    mask_criterion = nn.BCEWithLogitsLoss()

    scaler = GradScaler(enabled=(device == "cuda"))

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

        logging.info(f"Training on {len(train_loader)} batches")
        for step, (imgs, masks) in enumerate(train_loader, start=1):
            loss, cls_loss, mask_loss = profile_block(
                "train step",
                train_step,
                model,
                imgs,
                masks,
                optimizer,
                cls_criterion,
                mask_criterion,
                num_classes,
                device,
                scaler,
            )

            if step % 10 == 0:
                logging.info(f"Step {step} completed out of {len(train_loader)}")

            running_train_loss += loss.item()
            running_train_cls += cls_loss.item()
            running_train_mask += mask_loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        avg_train_cls = running_train_cls / len(train_loader)
        avg_train_mask = running_train_mask / len(train_loader)

        end_time = time.time()

        logging.info(
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
                imgs = imgs.to(device, non_blocking=True)

                if masks.dim() == 4:
                    if masks.size(1) == 1:
                        masks = masks.squeeze(1)
                    else:
                        masks = masks[:, 0, :, :]
                masks = masks.to(device, non_blocking=True)

                # validation برضه تحت autocast عشان السرعة (من غير GradScaler)
                with autocast(dtype=torch.float16, enabled=(device == "cuda")):
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

                    # BCE + Dice في الـ validation برضه
                    bce_l = mask_criterion(pred_masks, target_masks)
                    d_l = dice_loss(pred_masks, target_masks)
                    mask_l = bce_l + DICE_WEIGHT * d_l

                    l = cls_l + mask_l

                val_loss += l.item()
                val_cls_loss += cls_l.item()
                val_mask_loss += mask_l.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_cls = val_cls_loss / len(val_loader)
        avg_val_mask = val_mask_loss / len(val_loader)

        logging.info(
            f"[Epoch {epoch}/{num_epochs}] "
            f"Val:   total={avg_val_loss:.4f}, "
            f"cls={avg_val_cls:.4f}, mask={avg_val_mask:.4f}"
        )

        # -------- حفظ أحسن موديل --------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"exported_models/{epoch}-{save_path}")
            logging.info(
                f"--> Saved best model so far to '{save_path}' "
                f"(val_loss={best_val_loss:.4f})"
            )

    logging.info("Training finished!")
    return model


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    model = HybirdSegmentationAlgorithm(num_classes=1, net_type="21")
    train_p3m10k(model, save_path="dark21.pt")
