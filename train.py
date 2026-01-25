"""
Distillation Training Script
============================
Train student model to mimic teacher's soft labels.
No validation needed - just minimize difference from teacher.
"""

import os
import logging
import argparse
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from models import UNET as Model
from data.dataset import P3MMemmapDataset
from utils import profile_block, get_device


device = get_device()

logging.basicConfig(
    filename="distillation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


@dataclass
class DistillationConfig:
    # Training
    epochs: int = 30
    batch_size: int = 12
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # Data
    num_workers: int = 4
    pin_memory: bool = True
    num_samples: int = 12088

    # Optimization
    grad_accum_steps: int = 2

    # Scheduler
    warmup_epochs: int = 2
    min_lr: float = 1e-6

    # Loss weights
    mse_weight: float = 1.0
    kl_weight: float = 1.0

    # Logging & Checkpoints
    log_every: int = 5
    ckpt_dir: str = "checkpoints_distill"
    save_every: int = 1


class WarmupCosineLR:
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]["lr"]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (
                self.max_epochs - self.warmup_epochs
            )
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (
                1 + torch.cos(torch.tensor(progress * 3.14159))
            )
            lr = float(lr)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


def distillation_loss(student_logits, teacher_probs, cfg):
    if student_logits.dim() == 3:
        student_logits = student_logits.unsqueeze(1)
    if teacher_probs.dim() == 3:
        teacher_probs = teacher_probs.unsqueeze(1)

    # Ensure float32
    student_logits = student_logits.float()
    teacher_probs = teacher_probs.float()

    student_probs = torch.sigmoid(student_logits)

    # MSE Loss
    mse = nn.functional.mse_loss(student_probs, teacher_probs)

    # KL Divergence Loss
    eps = 1e-6
    s = student_probs.clamp(eps, 1 - eps)
    t = teacher_probs.clamp(eps, 1 - eps)

    kl = t * torch.log(t / s) + (1 - t) * torch.log((1 - t) / (1 - s))
    kl = kl.mean()

    total_loss = cfg.mse_weight * mse + cfg.kl_weight * kl

    return total_loss, mse.item(), kl.item(), student_probs


@torch.no_grad()
def compute_similarity(student_probs, teacher_probs, thresh=0.5):
    mae = (student_probs - teacher_probs).abs().mean().item()

    student_bin = (student_probs >= thresh).float()
    teacher_bin = (teacher_probs >= thresh).float()

    intersection = (student_bin * teacher_bin).sum()
    union = student_bin.sum() + teacher_bin.sum()

    dice_sim = (2 * intersection / (union + 1e-7)).item() if union > 0 else 1.0

    return mae, dice_sim


def train_batch(model, batch, optimizer, cfg, step):
    images, teacher_masks = batch
    images = images.to(device, non_blocking=True).float()
    teacher_masks = teacher_masks.to(device, non_blocking=True).float()

    student_logits = profile_block("train_batch_model_forward", model, images)
    loss, mse, kl, student_probs = distillation_loss(student_logits, teacher_masks, cfg)
    loss_scaled = loss / cfg.grad_accum_steps

    profile_block("train_batch_loss_scaled", loss_scaled.backward)

    if (step + 1) % cfg.grad_accum_steps == 0:
        profile_block("train_batch_optimizer_step", optimizer.step)
        optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        if teacher_masks.dim() == 3:
            teacher_masks = teacher_masks.unsqueeze(1)
        mae, dice_sim = compute_similarity(student_probs, teacher_masks)

    return loss.item(), mse, kl, mae, dice_sim


def train_epoch(model, loader, optimizer, cfg, epoch):
    model.train()

    loss_sum = mse_sum = kl_sum = mae_sum = dice_sum = 0.0

    number_of_batches = len(loader)
    for i, batch in enumerate(loader):
        loss, mse, kl, mae, dice_sim = train_batch(model, batch, optimizer, cfg, i)

        loss_sum += loss
        mse_sum += mse
        kl_sum += kl
        mae_sum += mae
        dice_sum += dice_sim

        if (i + 1) % cfg.log_every == 0:
            msg = (
                f"[E{epoch:03d} | {i+1:04d}/{number_of_batches}] "
                f"loss={loss:.4f} mse={mse:.4f} kl={kl:.4f} "
                f"mae={mae:.4f} dice_sim={dice_sim:.4f}"
            )
            logging.info(msg)

    n = len(loader)
    return loss_sum / n, mse_sum / n, kl_sum / n, mae_sum / n, dice_sum / n


def save_checkpoint(model, optimizer, epoch, metrics, cfg):
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    path = os.path.join(cfg.ckpt_dir, f"distill_epoch_{epoch:03d}.pt")

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )
    logging.info(f"Checkpoint saved: {path}")
    logging.info(f"Saved: {path}")


def load_checkpoint(model, optimizer, epoch, cfg):
    """Load checkpoint from specific epoch."""
    path = os.path.join(cfg.ckpt_dir, f"distill_epoch_{epoch:03d}.pt")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    loaded_epoch = checkpoint["epoch"]
    metrics = checkpoint.get("metrics", {})

    logging.info(f"Loaded checkpoint from epoch {loaded_epoch}")
    logging.info(f"Previous metrics: {metrics}")
    logging.info(f"Resumed from checkpoint: {path}")

    return loaded_epoch


def build_dataset(cfg):
    return P3MMemmapDataset(
        mmap_path="dataset/train_640_fp16_images.mmap",
        mask_mmap_path="dataset/train_640_fp16_soft_masks.mmap",
        N=cfg.num_samples,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Distillation Training")
    parser.add_argument(
        "--resume",
        type=int,
        default=None,
        help="Resume from epoch number (e.g., --resume 10)",
    )
    return parser.parse_args()


def train(resume_epoch=None):
    cfg = DistillationConfig()

    logging.info("\n" + "=" * 60)
    logging.info("Distillation Training (FP32)")
    logging.info("=" * 60)
    logging.info(
        f"Batch size: {cfg.batch_size} (effective: {cfg.batch_size * cfg.grad_accum_steps})"
    )
    logging.info(f"Learning rate: {cfg.lr}")
    logging.info(f"Epochs: {cfg.epochs}")
    logging.info(f"Loss: MSE (w={cfg.mse_weight}) + KL (w={cfg.kl_weight})")
    if resume_epoch:
        logging.info(f"Resuming from: epoch {resume_epoch}")
    logging.info("=" * 60 + "\n")

    model = Model().to(device)

    params = sum(p.numel() for p in model.parameters())
    logging.info(f"Parameters: {params:,} ({params/1e6:.1f}M)")

    train_loader = DataLoader(
        build_dataset(cfg),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = WarmupCosineLR(optimizer, cfg.warmup_epochs, cfg.epochs, cfg.min_lr)

    # Resume from checkpoint if specified
    start_epoch = 1
    if resume_epoch is not None:
        start_epoch = load_checkpoint(model, optimizer, resume_epoch, cfg) + 1
        logging.info(f"Will start from epoch {start_epoch}")

    logging.info(f"Dataset: {cfg.num_samples} samples")
    logging.info(f"Batches per epoch: {len(train_loader)}")
    logging.info("\nStarting distillation...\n")

    for epoch in range(start_epoch, cfg.epochs + 1):
        lr = scheduler.step(epoch - 1)
        logging.info(f"\n[Epoch {epoch}/{cfg.epochs}] LR: {lr:.6f}")

        loss, mse, kl, mae, dice_sim = profile_block(
            "train_epoch", train_epoch, model, train_loader, optimizer, cfg, epoch
        )

        logging.info(
            f"[Epoch {epoch} Summary] "
            f"loss={loss:.4f} mse={mse:.4f} kl={kl:.4f} "
            f"mae={mae:.4f} dice_sim={dice_sim:.4f}"
        )

        logging.info(
            f"Epoch {epoch}: loss={loss:.4f} mse={mse:.4f} "
            f"kl={kl:.4f} mae={mae:.4f} dice_sim={dice_sim:.4f}"
        )

        if epoch % cfg.save_every == 0 or epoch == cfg.epochs:
            metrics = {
                "loss": loss,
                "mse": mse,
                "kl": kl,
                "mae": mae,
                "dice_sim": dice_sim,
            }
            save_checkpoint(model, optimizer, epoch, metrics, cfg)

    final_path = os.path.join(cfg.ckpt_dir, "final_model.pt")
    torch.save(model.state_dict(), final_path)
    logging.info(f"\nFinal model saved: {final_path}")

    logging.info("\n" + "=" * 60)
    logging.info("Distillation complete!")
    logging.info("=" * 60 + "\n")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    torch.multiprocessing.set_start_method("spawn", force=True)
    args = parse_args()
    train(resume_epoch=args.resume)
