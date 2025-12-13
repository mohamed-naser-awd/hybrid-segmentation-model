import logging
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from network import HybirdSegmentationAlgorithm
from dataset import P3MMemmapDataset

# ==========================
# Logging
# ==========================
logging.basicConfig(
    filename="app.logs",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def train():
    pass


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    train()
