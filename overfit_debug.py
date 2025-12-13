import logging
import time
import random
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler

from network import HybirdSegmentationAlgorithm
from dataset import P3MMemmapDataset
from utils import profile_block

# ==========================
# Logging
# ==========================
logging.basicConfig(
    filename="overfit.logs",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Overfit debug started")

# ==========================
# Dice Loss + وزنها
# ==========================
DICE_WEIGHT = 1.0

def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    pred = pred.flatten(1)
    target = target.flatten(1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()

@torch.no_grad()
def hard_dice_iou_from_logits(mask_logits, mask_target, thr=0.5, eps=1e-6):
    """
    mask_logits: (B, Q, H, W) logits
    mask_target: (B, Q, H, W) float 0/1
    نحسب dice/iou على query 0 فقط (object)
    """
    prob = torch.sigmoid(mask_logits[:, 0])           # (B,H,W)
    pred = (prob > thr).float()
    tgt = (mask_target[:, 0] > 0.5).float()

    inter = (pred * tgt).sum(dim=(1,2))
    union = pred.sum(dim=(1,2)) + tgt.sum(dim=(1,2))
    dice = (2 * inter + eps) / (union + eps)

    iou = inter / (pred.sum(dim=(1,2)) + tgt.sum(dim=(1,2)) - inter + eps)
    return dice.mean().item(), iou.mean().item()

def seed_everything(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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
    imgs = imgs.to(device, non_blocking=True)

    if masks.dim() == 4:
        if masks.size(1) == 1:
            masks = masks.squeeze(1)
        else:
            masks = masks[:, 0, :, :]
    masks = masks.to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)

    with autocast(dtype=torch.float16, enabled=(device == "cuda")):
        pred_logits, pred_masks = model(imgs)  # (B,Q,C1), (B,Q,H,W)

        B, Q, C1 = pred_logits.shape
        _, Qm, H, W = pred_masks.shape
        assert Q == Qm, "Mismatch between Q of logits and masks!"

        target_classes = torch.full(
            (B, Q),
            fill_value=num_classes,   # background index
            dtype=torch.long,
            device=device,
        )
        target_classes[:, 0] = 0  # query 0 = object

        target_masks = torch.zeros((B, Q, H, W), dtype=torch.float32, device=device)
        target_masks[:, 0, :, :] = masks

        cls_loss = cls_criterion(
            pred_logits.view(B * Q, C1),
            target_classes.view(B * Q),
        )

        bce_loss = mask_criterion(pred_masks, target_masks)
        d_loss = dice_loss(pred_masks, target_masks)
        mask_loss = bce_loss + DICE_WEIGHT * d_loss

        loss = cls_loss + mask_loss

    def backward_fn():
        scaler.scale(loss).backward()

    profile_block("backward", backward_fn)

    def step_fn():
        scaler.step(optimizer)
        scaler.update()

    profile_block("optimizer step", step_fn)

    return loss, cls_loss, mask_loss

def overfit_p3m_memmap(
    model: HybirdSegmentationAlgorithm,
    overfit_n: int = 16,
    num_epochs: int = 300,
    batch_size: int = 8,
    lr: float = 3e-4,
    num_workers: int = 0,
    seed: int = 123,
):
    seed_everything(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    model = model.to(device)

    # نفس داتاسيت التدريب بتاعتك
    full_train = P3MMemmapDataset(
        mmap_path="dataset/train_640_fp16_images.mmap",
        mask_mmap_path="dataset/train_640_fp16_masks.mmap",
        N=13003,
    )

    # اختار أول N (أو ممكن تختار random indices)
    indices = list(range(overfit_n))
    tiny_ds = Subset(full_train, indices)

    pin = True if device == "cuda" else False
    loader = DataLoader(
        tiny_ds,
        batch_size=batch_size,
        shuffle=True,          # مهم عشان يلف عليهم بترتيب مختلف
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True if overfit_n >= batch_size else False,
    )

    num_classes = 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    cls_criterion = nn.CrossEntropyLoss()
    mask_criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=(device == "cuda"))

    logging.info(f"Overfitting on N={overfit_n} samples, epochs={num_epochs}, bs={batch_size}, lr={lr}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        t0 = time.time()

        total_loss = 0.0
        total_cls = 0.0
        total_mask = 0.0
        total_dice = 0.0
        total_iou = 0.0
        n_batches = 0

        for imgs, masks in loader:
            loss, cls_loss, mask_loss = train_step(
                model, imgs, masks,
                optimizer, cls_criterion, mask_criterion,
                num_classes, device, scaler
            )

            total_loss += loss.item()
            total_cls += cls_loss.item()
            total_mask += mask_loss.item()

            # metric على نفس الباتش (للتشخيص فقط)
            with torch.no_grad():
                imgs_d = imgs.to(device, non_blocking=True)
                if masks.dim() == 4:
                    if masks.size(1) == 1:
                        masks_d = masks.squeeze(1)
                    else:
                        masks_d = masks[:, 0, :, :]
                else:
                    masks_d = masks
                masks_d = masks_d.to(device, non_blocking=True)

                with autocast(dtype=torch.float16, enabled=(device == "cuda")):
                    pred_logits, pred_masks = model(imgs_d)
                    B, Q, C1 = pred_logits.shape
                    _, _, H, W = pred_masks.shape
                    target_masks = torch.zeros((B, Q, H, W), dtype=torch.float32, device=device)
                    target_masks[:, 0] = masks_d

                dice, iou = hard_dice_iou_from_logits(pred_masks, target_masks, thr=0.5)
                total_dice += dice
                total_iou += iou

            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        avg_cls = total_cls / max(1, n_batches)
        avg_mask = total_mask / max(1, n_batches)
        avg_dice = total_dice / max(1, n_batches)
        avg_iou = total_iou / max(1, n_batches)

        dt = time.time() - t0
        msg = (
            f"[Epoch {epoch}/{num_epochs}] "
            f"loss={avg_loss:.4f} (cls={avg_cls:.4f}, mask={avg_mask:.4f}) "
            f"dice@0.5={avg_dice:.4f} iou@0.5={avg_iou:.4f} "
            f"time={dt:.2f}s"
        )
        print(msg)
        logging.info(msg)

        # وقف بدري لو وصلنا حفظ واضح
        if avg_dice > 0.98 and avg_iou > 0.95:
            logging.info("Early stop: reached near-perfect overfit.")
            break

    return model

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    model = HybirdSegmentationAlgorithm(num_classes=1, net_type="21")
    overfit_p3m_memmap(
        model,
        overfit_n=16,
        num_epochs=400,
        batch_size=8,
        lr=3e-4,
        num_workers=0,
        seed=123,
    )
