import os
import random
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler

# نفس imports بتاعتك
from network import HybirdSegmentationAlgorithm
from dataset import P3MMemmapDataset

# ==========================
# Device + Reproducibility
# ==========================
device = "cuda" if torch.cuda.is_available() else "cpu"

def seed_everything(seed: int = 123):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# ==========================
# Config (Overfit)
# ==========================
@dataclass
class OverfitConfig:
    # خليه كبير عشان يسمح بالـ overfit
    epochs: int = 500
    batch_size: int = 10           # = عدد الصور نفسها
    lr: float = 2e-4               # ممكن ترفعه لـ 1e-3 لو التدريب بطيء
    weight_decay: float = 0.0      # عادةً بنقفله للـ overfit
    num_workers: int = 0
    pin_memory: bool = True

    use_amp: bool = True

    # loss weights
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    dice_smooth: float = 1.0

    # debug
    subset_size: int = 10
    save_every: int = 10           # كل كام epoch يحفظ صور مقارنة
    out_dir: str = "overfit_debug"
    thresh: float = 0.1

# ==========================
# Losses & Metrics (LOGITS -> sigmoid inside)
# ==========================
def dice_loss_from_probs(probs: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth + 1e-7)
    return 1.0 - dice.mean()

@torch.no_grad()
def dice_score_from_probs(probs: torch.Tensor, targets: torch.Tensor, thresh: float = 0.5) -> float:
    preds = (probs >= thresh).float()
    preds = preds.flatten(1)
    targets = targets.flatten(1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + 1.0) / (union + 1.0 + 1e-7)
    return float(dice.mean().item())

def compute_loss(mask_logits: torch.Tensor, target_masks: torch.Tensor, bce_logits_crit: nn.Module, cfg: OverfitConfig):
    # ensure shape [B,1,H,W]
    if mask_logits.dim() == 3:
        mask_logits = mask_logits.unsqueeze(1)
    if target_masks.dim() == 3:
        target_masks = target_masks.unsqueeze(1)

    target_masks = (target_masks > 0).float()

    bce = bce_logits_crit(mask_logits, target_masks)
    probs = torch.sigmoid(mask_logits)
    dice = dice_loss_from_probs(probs, target_masks, smooth=cfg.dice_smooth)

    total = cfg.bce_weight * bce + cfg.dice_weight * dice
    return total, float(bce.item()), float(dice.item()), probs

# ==========================
# Dataset: نفس memmap dataset بتاعك لكن Subset(10)
# ==========================
def build_full_train_dataset():
    return P3MMemmapDataset(
        mmap_path="dataset/train_640_fp16_images.mmap",
        mask_mmap_path="dataset/train_640_fp16_masks.mmap",
        N=20392,
    )

# ==========================
# Debug image saving
# ==========================
@torch.no_grad()
def save_debug_batch(images, targets, probs, epoch: int, cfg: OverfitConfig):
    """
    Saves a small grid of:
      - input image
      - GT mask
      - predicted prob map
      - predicted binary mask
    Each saved as separate PNG grids to avoid confusion.
    """
    os.makedirs(cfg.out_dir, exist_ok=True)

    # take up to 4 samples for visualization
    k = min(4, images.size(0))
    imgs = images[:k].detach().float().cpu()
    tgts = targets[:k].detach().float().cpu()
    prb = probs[:k].detach().float().cpu()
    binm = (prb >= cfg.thresh).float()

    # normalize image if it looks like 0..255
    if imgs.max() > 1.5:
        imgs = imgs / 255.0

    # make sure masks are 1-channel
    if tgts.dim() == 3:
        tgts = tgts.unsqueeze(1)
    if prb.dim() == 3:
        prb = prb.unsqueeze(1)
    if binm.dim() == 3:
        binm = binm.unsqueeze(1)

    try:
        import torchvision
        torchvision.utils.save_image(imgs, os.path.join(cfg.out_dir, f"e{epoch:04d}_img.png"))
        torchvision.utils.save_image(tgts, os.path.join(cfg.out_dir, f"e{epoch:04d}_gt.png"))
        torchvision.utils.save_image(prb,  os.path.join(cfg.out_dir, f"e{epoch:04d}_prob.png"))
        torchvision.utils.save_image(binm, os.path.join(cfg.out_dir, f"e{epoch:04d}_bin.png"))
    except Exception as e:
        print(f"[WARN] Could not save debug images (torchvision missing?): {e}")

# ==========================
# Main Overfit Loop
# ==========================
def main():
    seed_everything(123)
    cfg = OverfitConfig()

    # Model
    model = HybirdSegmentationAlgorithm(
        num_classes=1,
        query_count=1,
    ).to(device)

    # Optional: load your checkpoint (زي كودك)
    ckpt_path = "checkpoints/epoch_0008.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        print(f"[INFO] Loaded checkpoint: {ckpt_path} (epoch={ckpt.get('epoch', 'N/A')})")
    else:
        print(f"[INFO] No checkpoint found at {ckpt_path}. Training from scratch.")

    # If you have dropout inside model and want pure overfit:
    # try to disable it (depends on your implementation)
    if hasattr(model, "dropout") and isinstance(model.dropout, nn.Dropout2d):
        model.dropout.p = 0.0
        print("[INFO] Disabled model.dropout (p=0.0) for overfit test.")

    # Dataset subset
    full_ds = build_full_train_dataset()
    indices = list(range(cfg.subset_size))  # أول 10 صور
    # ممكن بدل أول 10: random sample (لكن خليك ثابت في الأول)
    ds10 = Subset(full_ds, indices)

    loader = DataLoader(
        ds10,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.pin_memory and device == "cuda"),
        drop_last=False,
    )

    # Optim
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=(cfg.use_amp and device == "cuda"))
    bce_logits_crit = nn.BCEWithLogitsLoss()

    best_dice = -1.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        loss_sum = 0.0
        bce_sum = 0.0
        diceL_sum = 0.0
        dice_s_sum = 0.0
        n_batches = 0

        for batch in loader:
            images, target_masks = batch
            images = images.to(device, non_blocking=True)
            target_masks = target_masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(dtype=torch.float16, enabled=(cfg.use_amp and device == "cuda")):
                mask_logits = model(images)
                loss, bce, diceL, probs = compute_loss(mask_logits, target_masks, bce_logits_crit, cfg)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                if probs.dim() == 3:
                    probs = probs.unsqueeze(1)
                if target_masks.dim() == 3:
                    target_masks = target_masks.unsqueeze(1)
                dice_s = dice_score_from_probs(probs, target_masks, thresh=cfg.thresh)

            loss_sum += float(loss.item())
            bce_sum += bce
            diceL_sum += diceL
            dice_s_sum += float(dice_s)
            n_batches += 1

        avg_loss = loss_sum / max(1, n_batches)
        avg_bce = bce_sum / max(1, n_batches)
        avg_diceL = diceL_sum / max(1, n_batches)
        avg_dice = dice_s_sum / max(1, n_batches)

        if avg_dice > best_dice:
            best_dice = avg_dice

        print(f"[OVERFIT E{epoch:04d}] loss={avg_loss:.4f} bce={avg_bce:.4f} diceL={avg_diceL:.4f} dice@{cfg.thresh:.2f}={avg_dice:.4f} (best={best_dice:.4f})")

        # Save debug images occasionally
        if (epoch % cfg.save_every == 0) or (avg_dice > 0.98):
            model.eval()
            with torch.no_grad():
                images, target_masks = next(iter(loader))
                images = images.to(device, non_blocking=True)
                target_masks = target_masks.to(device, non_blocking=True)
                with autocast(dtype=torch.float16, enabled=(cfg.use_amp and device == "cuda")):
                    logits = model(images)
                    probs = torch.sigmoid(logits)
            save_debug_batch(images, target_masks, probs, epoch, cfg)

        # Early stop if perfect-ish
        if avg_dice >= 0.99:
            torch.save(model.state_dict(), "overfit_debug_model.pt")
            print("[INFO] Reached dice >= 0.99 on 10 samples. Overfit test PASSED.")
            break

    print(f"[DONE] best_dice={best_dice:.4f}, debug_dir={cfg.out_dir}")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
