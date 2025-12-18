import os
import logging
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from network import HybirdSegmentationAlgorithm
from dataset import P3MMemmapDataset

# ==========================
# Device
# ==========================
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# Logging
# ==========================
logging.basicConfig(
    filename="app.logs",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ==========================
# Config
# ==========================
@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 10
    lr: float = 2e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    pin_memory: bool = True
    log_every: int = 100
    ckpt_dir: str = "checkpoints"
    use_amp: bool = True
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    dice_smooth: float = 1.0


# ==========================
# Losses & Metrics (LOGITS -> sigmoid inside loss/metrics)
# ==========================
def dice_loss_from_probs(
    probs: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    probs = probs.flatten(1)
    targets = targets.flatten(1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth + 1e-7)
    return 1.0 - dice.mean()


@torch.no_grad()
def dice_score_from_probs(
    probs: torch.Tensor,
    targets: torch.Tensor,
    thresh: float = 0.5,
) -> float:
    preds = (probs >= thresh).float()

    preds = preds.flatten(1)
    targets = targets.flatten(1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + 1.0) / (union + 1.0 + 1e-7)
    return float(dice.mean().item())


def compute_loss(
    mask_logits: torch.Tensor,
    target_masks: torch.Tensor,
    bce_logits_crit: nn.Module,
    cfg: TrainConfig,
):
    # ensure shape [B,1,H,W]
    if mask_logits.dim() == 3:
        mask_logits = mask_logits.unsqueeze(1)
    if target_masks.dim() == 3:
        target_masks = target_masks.unsqueeze(1)

    target_masks = target_masks.float()

    # BCE on logits (stable)
    bce = bce_logits_crit(mask_logits, target_masks)

    # Dice on probs
    probs = torch.sigmoid(mask_logits)
    dice = dice_loss_from_probs(probs, target_masks, smooth=cfg.dice_smooth)

    total = cfg.bce_weight * bce + cfg.dice_weight * dice
    return total, float(bce.item()), float(dice.item()), probs


# ==========================
# Optim / Checkpoint
# ==========================
def build_optimizer(model: nn.Module, cfg: TrainConfig):
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    cfg: TrainConfig,
):
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    path = os.path.join(cfg.ckpt_dir, f"epoch_{epoch:04d}.pt")
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
        },
        path,
    )
    logging.info(f"Checkpoint saved: {path}")


# ==========================
# Train / Val Steps
# ==========================
def train_batch(model, batch, optimizer, scaler, bce_logits_crit, cfg: TrainConfig):
    model.train()
    images, target_masks = batch
    images = images.to(device, non_blocking=True)
    target_masks = target_masks.to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)

    with autocast(dtype=torch.float16, enabled=(cfg.use_amp and device == "cuda")):
        mask_logits = model(images)  # [B,1,H,W] logits
        loss, bce, dice, probs = compute_loss(mask_logits, target_masks, bce_logits_crit, cfg)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    with torch.no_grad():
        if probs.dim() == 3:
            probs = probs.unsqueeze(1)
        if target_masks.dim() == 3:
            target_masks = target_masks.unsqueeze(1)
        dice_s = dice_score_from_probs(probs, target_masks)

    return float(loss.item()), bce, dice, dice_s


@torch.no_grad()
def val_batch(model, batch, bce_logits_crit, cfg: TrainConfig):
    model.eval()
    images, target_masks = batch
    images = images.to(device, non_blocking=True)
    target_masks = target_masks.to(device, non_blocking=True)

    with autocast(dtype=torch.float16, enabled=(cfg.use_amp and device == "cuda")):
        mask_logits = model(images)
        loss, bce, dice, probs = compute_loss(mask_logits, target_masks, bce_logits_crit, cfg)

    if probs.dim() == 3:
        probs = probs.unsqueeze(1)
    if target_masks.dim() == 3:
        target_masks = target_masks.unsqueeze(1)

    dice_s = dice_score_from_probs(probs, target_masks)
    return float(loss.item()), bce, dice, dice_s


# ==========================
# Epochs
# ==========================
def train_epoch(model, loader, optimizer, scaler, cfg: TrainConfig, epoch: int):
    bce_logits_crit = nn.BCEWithLogitsLoss()

    loss_sum = bce_sum = dice_sum = dice_s_sum = 0.0

    for i, batch in enumerate(loader, 1):
        loss, bce, dice, dice_s = train_batch(
            model, batch, optimizer, scaler, bce_logits_crit, cfg
        )

        loss_sum += loss
        bce_sum += bce
        dice_sum += dice
        dice_s_sum += dice_s

        if i % cfg.log_every == 0:
            msg = (
                f"[TRAIN E{epoch:03d} | {i:04d}] "
                f"loss={loss:.4f} bce={bce:.4f} "
                f"diceL={dice:.4f} dice@0.5={dice_s:.4f}"
            )
            logging.info(msg)

    n = len(loader)
    return (loss_sum / n, bce_sum / n, dice_sum / n, dice_s_sum / n)


def val_epoch(model, loader, cfg: TrainConfig, epoch: int):
    bce_logits_crit = nn.BCEWithLogitsLoss()

    loss_sum = bce_sum = dice_sum = dice_s_sum = 0.0

    for batch in loader:
        loss, bce, dice, dice_s = val_batch(model, batch, bce_logits_crit, cfg)
        loss_sum += loss
        bce_sum += bce
        dice_sum += dice
        dice_s_sum += dice_s

    n = len(loader)
    msg = (
        f"[VAL   E{epoch:03d}] "
        f"loss={loss_sum/n:.4f} bce={bce_sum/n:.4f} "
        f"diceL={dice_sum/n:.4f} dice@0.5={dice_s_sum/n:.4f}"
    )
    print(msg)
    logging.info(msg)

    return dice_s_sum / n


# ==========================
# Datasets
# ==========================
def build_train_dataset():
    return P3MMemmapDataset(
        mmap_path="dataset/train_640_fp16_images.mmap",
        mask_mmap_path="dataset/train_640_fp16_masks.mmap",
        N=20392,
    )


def build_val_dataset():
    return P3MMemmapDataset(
        mmap_path="dataset/val_640_fp16_images.mmap",
        mask_mmap_path="dataset/val_640_fp16_masks.mmap",
        N=500,
    )


# ==========================
# Main
# ==========================
def train():
    cfg = TrainConfig()

    model = HybirdSegmentationAlgorithm(
        num_classes=1,
        query_count=1,  # مهم: semantic mask واحدة
        d_model=192,
    ).to(device)

    train_loader = DataLoader(
        build_train_dataset(),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.pin_memory and device == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        build_val_dataset(),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.pin_memory and device == "cuda"),
    )

    optimizer = build_optimizer(model, cfg)
    scaler = GradScaler(enabled=(cfg.use_amp and device == "cuda"))

    best_dice = 0.0

    for epoch in range(1, cfg.epochs + 1):
        logging.info(f"Epoch {epoch} started")
        train_epoch(model, train_loader, optimizer, scaler, cfg, epoch)
        logging.info(f"Epoch {epoch} training completed")

        logging.info(f"Epoch {epoch} validation started")
        val_dice = val_epoch(model, val_loader, cfg, epoch)
        logging.info(f"Epoch {epoch} validation completed")

        # if val_dice > best_dice:
        best_dice = val_dice
        save_checkpoint(model, optimizer, scaler, epoch, cfg)
        logging.info(f"New BEST dice@0.5 = {best_dice:.4f}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    train()
