import os
import logging
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from network import SemanticSegmentationModel
from dataset import P3MMemmapDataset, MixedSegmentationDataset
from utils import profile_block


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
    epochs: int = 30
    batch_size: int = 50
    lr: float = 2e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    pin_memory: bool = True
    log_every: int = 20
    ckpt_dir: str = "checkpoints"
    disable_backbone_training: bool = True

    # FP32 only
    use_amp: bool = False

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
    # flatten per-sample
    probs = probs.flatten(1)
    targets = targets.flatten(1)

    # samples that actually contain foreground
    has_fg = targets.sum(dim=1) > 0

    # لو كل الباتش فاضي، نرجع صفر (ما نكافئش التصفير)
    if not has_fg.any():
        return probs.new_tensor(0.0)

    probs = probs[has_fg]
    targets = targets[has_fg]

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

    pred_sum = preds.sum(dim=1)
    target_sum = targets.sum(dim=1)

    intersection = (preds * targets).sum(dim=1)
    union = pred_sum + target_sum

    dice = torch.zeros_like(union)

    # case 1: GT empty & pred empty -> perfect
    both_empty = (target_sum == 0) & (pred_sum == 0)
    dice[both_empty] = 1.0

    # case 2: GT empty & pred non-empty -> wrong
    gt_empty_pred_non = (target_sum == 0) & (pred_sum > 0)
    dice[gt_empty_pred_non] = 0.0

    # case 3: normal case
    normal = target_sum > 0
    dice[normal] = (2.0 * intersection[normal] + 1.0) / (union[normal] + 1.0 + 1e-7)

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

    # target_masks already binarized & float before calling compute_loss
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
    if cfg.disable_backbone_training:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    else:
        trainable_params = model.parameters()
    optimizer = torch.optim.AdamW(
        trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    return optimizer


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
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
            # no scaler in FP32 mode
        },
        path,
    )
    logging.info(f"Checkpoint saved: {path}")


# ==========================
# Train / Val Steps (FP32 only)
# ==========================
def train_batch(model, batch, optimizer, bce_logits_crit, cfg: TrainConfig):
    model.train()
    images, target_masks = batch

    # Force FP32
    images = images.to(device, non_blocking=True).float()

    # Force masks to {0,1} float32
    target_masks = target_masks.to(device, non_blocking=True)
    target_masks = (target_masks > 0).float()

    optimizer.zero_grad(set_to_none=True)

    mask_logits = profile_block("model", model, images)  # logits FP32
    loss, bce, dice, probs = profile_block("compute_loss", compute_loss, mask_logits, target_masks, bce_logits_crit, cfg)

    profile_block("backward", loss.backward)
    profile_block("optimizer", optimizer.step)

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

    images = images.to(device, non_blocking=True).float()
    target_masks = (target_masks.to(device, non_blocking=True) > 0).float()

    mask_logits = model(images)
    loss, bce, dice, probs = compute_loss(
        mask_logits, target_masks, bce_logits_crit, cfg
    )

    if probs.dim() == 3:
        probs = probs.unsqueeze(1)
    if target_masks.dim() == 3:
        target_masks = target_masks.unsqueeze(1)

    dice_s = dice_score_from_probs(probs, target_masks)
    return float(loss.item()), bce, dice, dice_s


# ==========================
# Epochs
# ==========================
def train_epoch(model, loader, optimizer, cfg: TrainConfig, epoch: int):
    bce_logits_crit = nn.BCEWithLogitsLoss()

    loss_sum = bce_sum = dice_sum = dice_s_sum = 0.0

    for i, batch in enumerate(loader, 1):
        loss, bce, dice, dice_s = profile_block("train_batch", train_batch, model, batch, optimizer, bce_logits_crit, cfg)

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
    positive_dataset_size = 12088
    negative_dataset_size = 8304

    positive_dataset = P3MMemmapDataset(
        "dataset/train_640_fp16_images.mmap",
        "dataset/train_640_fp16_masks.mmap",
        positive_dataset_size,
    )

    negative_dataset = P3MMemmapDataset(
        "dataset/train_640_fp16_negative_images.mmap",
        "dataset/train_640_fp16_negative_masks.mmap",
        negative_dataset_size,
    )

    return MixedSegmentationDataset(
        pos_dataset=positive_dataset,
        neg_dataset=negative_dataset,
        pos_ratio=0.9,  # عدلها براحتك (0.5 / 0.7 / ...)
        length=20392,  # نفس الرقم اللي كنت مستعمله
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

    model = SemanticSegmentationModel().to(device).float()  # ⬅️ force FP32 model
    if cfg.disable_backbone_training:
        model.disable_backbone_training()

    # model_path = "checkpoints/epoch_0006.pt"
    # state_dict = torch.load(model_path, map_location=device)
    # model.load_state_dict(state_dict["model"])

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
    # optimizer.load_state_dict(state_dict["optimizer"])

    best_dice = 0.0
    epoch_start = 0

    for epoch in range(epoch_start + 1, cfg.epochs + 1):
        logging.info(f"Epoch {epoch} started")
        train_epoch(model, train_loader, optimizer, cfg, epoch)
        logging.info(f"Epoch {epoch} training completed")

        logging.info(f"Epoch {epoch} validation started")
        val_dice = val_epoch(model, val_loader, cfg, epoch)
        logging.info(f"Epoch {epoch} validation completed")

        best_dice = val_dice
        save_checkpoint(model, optimizer, epoch, cfg)
        logging.info(f"New BEST dice@0.5 = {best_dice:.4f}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    train()
