"""
DARTS (Differentiable Architecture Search) for UltraFastNet
============================================================
This script searches for optimal encoder/decoder architectures
for real-time human segmentation using gradient-based NAS.

Usage:
    python nas_darts.py --data_dir /path/to/dataset --epochs 50

Output:
    - best_architecture.json: The discovered architecture config
    - search_log.csv: Search progress metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from collections import OrderedDict


# =============================================================================
# Search Space Operations
# =============================================================================


class OPS:
    """Available operations in the search space"""

    @staticmethod
    def conv_3x3(C_in: int, C_out: int, stride: int = 1) -> nn.Module:
        """Standard 3x3 convolution"""
        return nn.Sequential(
            nn.Conv2d(C_in, C_out, 3, stride, 1, bias=False),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def conv_1x1(C_in: int, C_out: int, stride: int = 1) -> nn.Module:
        """1x1 pointwise convolution"""
        return nn.Sequential(
            nn.Conv2d(C_in, C_out, 1, stride, 0, bias=False),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_sep_3x3(C_in: int, C_out: int, stride: int = 1) -> nn.Module:
        """Depthwise separable convolution (MobileNet style)"""
        return nn.Sequential(
            # Depthwise
            nn.Conv2d(C_in, C_in, 3, stride, 1, groups=C_in, bias=False),
            nn.BatchNorm2d(C_in),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv2d(C_in, C_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_sep_5x5(C_in: int, C_out: int, stride: int = 1) -> nn.Module:
        """Depthwise separable 5x5 convolution"""
        return nn.Sequential(
            nn.Conv2d(C_in, C_in, 5, stride, 2, groups=C_in, bias=False),
            nn.BatchNorm2d(C_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(C_in, C_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def dilated_conv_3x3(
        C_in: int, C_out: int, stride: int = 1, dilation: int = 2
    ) -> nn.Module:
        """Dilated convolution for larger receptive field"""
        return nn.Sequential(
            nn.Conv2d(C_in, C_out, 3, stride, dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def skip_connect(C_in: int, C_out: int, stride: int = 1) -> nn.Module:
        """Skip connection (identity or projection)"""
        if stride == 1 and C_in == C_out:
            return nn.Identity()
        else:
            return nn.Sequential(
                nn.Conv2d(C_in, C_out, 1, stride, 0, bias=False), nn.BatchNorm2d(C_out)
            )

    @staticmethod
    def avg_pool_3x3(C_in: int, C_out: int, stride: int = 1) -> nn.Module:
        """Average pooling followed by projection"""
        return nn.Sequential(
            nn.AvgPool2d(3, stride, 1, count_include_pad=False),
            nn.Conv2d(C_in, C_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(C_out),
        )

    @staticmethod
    def max_pool_3x3(C_in: int, C_out: int, stride: int = 1) -> nn.Module:
        """Max pooling followed by projection"""
        return nn.Sequential(
            nn.MaxPool2d(3, stride, 1),
            nn.Conv2d(C_in, C_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(C_out),
        )


# Operation registry
PRIMITIVES = [
    "conv_3x3",
    "conv_1x1",
    "depthwise_sep_3x3",
    "depthwise_sep_5x5",
    "dilated_conv_3x3",
    "skip_connect",
    "avg_pool_3x3",
    "max_pool_3x3",
]


def get_op(name: str, C_in: int, C_out: int, stride: int = 1) -> nn.Module:
    """Get operation by name"""
    op_fn = getattr(OPS, name)
    return op_fn(C_in, C_out, stride)


# =============================================================================
# Mixed Operation (Differentiable)
# =============================================================================


class MixedOp(nn.Module):
    """
    Mixed operation: weighted sum of all candidate operations.
    Weights are learned via architecture parameters (alphas).
    """

    def __init__(self, C_in: int, C_out: int, stride: int = 1):
        super().__init__()
        self.ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = get_op(primitive, C_in, C_out, stride)
            self.ops.append(op)

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor
            weights: Softmax weights for each operation
        """
        return sum(w * op(x) for w, op in zip(weights, self.ops))


# =============================================================================
# Searchable Cell
# =============================================================================


class SearchableCell(nn.Module):
    """
    A cell with searchable operations.
    Can be configured as encoder (stride=2) or decoder cell.
    """

    def __init__(
        self,
        C_in: int,
        C_out: int,
        num_nodes: int = 4,
        stride: int = 1,
        is_decoder: bool = False,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.is_decoder = is_decoder

        # Preprocess input
        self.preprocess = nn.Sequential(
            nn.Conv2d(C_in, C_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=True),
        )

        # Mixed operations between nodes
        # Each node receives input from all previous nodes
        self.ops = nn.ModuleList()
        for i in range(num_nodes):
            for j in range(i + 1):  # Connections from node j to node i
                op_stride = stride if j == 0 and not is_decoder else 1
                self.ops.append(MixedOp(C_out, C_out, op_stride))

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(C_out * num_nodes, C_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(C_out),
        )

    def forward(
        self,
        x: torch.Tensor,
        alphas: torch.Tensor,
        skip_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor
            alphas: Architecture parameters [num_edges, num_ops]
            skip_input: Optional skip connection for decoder
        """
        # Preprocess
        s0 = self.preprocess(x)

        # Handle decoder skip connection
        if self.is_decoder and skip_input is not None:
            s0 = s0 + F.interpolate(
                skip_input, size=s0.shape[2:], mode="bilinear", align_corners=False
            )

        # Forward through nodes
        states = [s0]
        offset = 0

        for i in range(self.num_nodes):
            # Gather inputs from all previous nodes
            node_inputs = []
            for j in range(i + 1):
                edge_weights = F.softmax(alphas[offset], dim=-1)
                node_inputs.append(self.ops[offset](states[j], edge_weights))
                offset += 1

            # Sum inputs for this node
            states.append(sum(node_inputs))

        # Concatenate all intermediate states (excluding s0)
        out = torch.cat(states[1:], dim=1)
        out = self.output_proj(out)

        return out


# =============================================================================
# Searchable UltraFastNet
# =============================================================================


class SearchableUltraFastNet(nn.Module):
    """
    UltraFastNet with searchable architecture.
    The encoder and decoder structures are defined by architecture parameters.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        init_channels: int = 32,
        num_cells: int = 4,
        num_nodes: int = 4,
    ):
        super().__init__()
        self.num_cells = num_cells
        self.num_nodes = num_nodes

        # Calculate number of edges per cell
        # Node i receives from nodes 0..i-1, so edges = 0+1+2+...+(num_nodes-1) = num_nodes*(num_nodes+1)/2
        self.num_edges = sum(range(1, num_nodes + 1))

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(init_channels, init_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True),
        )

        # Encoder cells (downsampling)
        self.encoder_cells = nn.ModuleList()
        channels = [init_channels]

        for i in range(num_cells):
            C_in = channels[-1]
            C_out = min(C_in * 2, 256)  # Cap at 256 channels
            cell = SearchableCell(C_in, C_out, num_nodes, stride=2, is_decoder=False)
            self.encoder_cells.append(cell)
            channels.append(C_out)

        # Decoder cells (upsampling)
        self.decoder_cells = nn.ModuleList()
        self.upsample = nn.ModuleList()

        for i in range(num_cells):
            C_in = channels[-(i + 1)]
            C_skip = channels[-(i + 2)]
            C_out = C_skip

            self.upsample.append(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            )
            cell = SearchableCell(C_in, C_out, num_nodes, stride=1, is_decoder=True)
            self.decoder_cells.append(cell)

        # Final head
        self.head = nn.Sequential(
            nn.Conv2d(init_channels, init_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(init_channels, num_classes, 1),
        )

        # Architecture parameters (learnable)
        self._initialize_alphas()

    def _initialize_alphas(self):
        """Initialize architecture parameters"""
        num_ops = len(PRIMITIVES)

        # Encoder alphas: one set per cell
        self.alphas_encoder = nn.ParameterList(
            [
                nn.Parameter(1e-3 * torch.randn(self.num_edges, num_ops))
                for _ in range(self.num_cells)
            ]
        )

        # Decoder alphas: one set per cell
        self.alphas_decoder = nn.ParameterList(
            [
                nn.Parameter(1e-3 * torch.randn(self.num_edges, num_ops))
                for _ in range(self.num_cells)
            ]
        )

    def arch_parameters(self) -> List[nn.Parameter]:
        """Return architecture parameters"""
        return list(self.alphas_encoder) + list(self.alphas_decoder)

    def model_parameters(self) -> List[nn.Parameter]:
        """Return model (weight) parameters"""
        arch_params = set(self.arch_parameters())
        return [p for p in self.parameters() if p not in arch_params]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.stem(x)

        # Encoder
        encoder_features = [x]
        for i, cell in enumerate(self.encoder_cells):
            x = cell(x, self.alphas_encoder[i])
            encoder_features.append(x)

        # Decoder
        for i, cell in enumerate(self.decoder_cells):
            skip = encoder_features[-(i + 2)]
            x = self.upsample[i](x)
            x = cell(x, self.alphas_decoder[i], skip_input=skip)

        # Head
        x = self.head(x)

        return x

    def get_genotype(self) -> Dict:
        """Extract the best architecture from learned alphas"""

        def parse_alphas(alphas_list: nn.ParameterList, prefix: str) -> Dict:
            genotype = {}
            for cell_idx, alphas in enumerate(alphas_list):
                cell_genotype = []
                weights = F.softmax(alphas, dim=-1).detach().cpu().numpy()

                offset = 0
                for i in range(self.num_nodes):
                    # Find best 2 edges for this node
                    edges = []
                    for j in range(i + 1):
                        best_op_idx = np.argmax(weights[offset])
                        best_op = PRIMITIVES[best_op_idx]
                        best_weight = weights[offset, best_op_idx]
                        edges.append((best_op, j, best_weight))
                        offset += 1

                    # Keep top-2 edges by weight
                    edges.sort(key=lambda x: x[2], reverse=True)
                    top_edges = edges[:2] if len(edges) >= 2 else edges
                    cell_genotype.append([(op, src) for op, src, _ in top_edges])

                genotype[f"{prefix}_cell_{cell_idx}"] = cell_genotype

            return genotype

        encoder_genotype = parse_alphas(self.alphas_encoder, "encoder")
        decoder_genotype = parse_alphas(self.alphas_decoder, "decoder")

        return {**encoder_genotype, **decoder_genotype}


# =============================================================================
# DARTS Trainer
# =============================================================================


class DARTSTrainer:
    """
    DARTS training procedure:
    1. Update weights w by gradient descent on training loss
    2. Update architecture α by gradient descent on validation loss
    """

    def __init__(
        self,
        model: SearchableUltraFastNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lr_model: float = 0.025,
        lr_arch: float = 3e-4,
        weight_decay: float = 3e-4,
        arch_weight_decay: float = 1e-3,
        epochs: int = 50,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs

        # Model optimizer (weights)
        self.optimizer_w = torch.optim.SGD(
            model.model_parameters(),
            lr=lr_model,
            momentum=0.9,
            weight_decay=weight_decay,
        )

        # Architecture optimizer (alphas)
        self.optimizer_alpha = torch.optim.Adam(
            model.arch_parameters(),
            lr=lr_arch,
            betas=(0.5, 0.999),
            weight_decay=arch_weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_w, epochs, eta_min=0.001
        )

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # Logging
        self.history = []

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        train_loss = 0.0
        val_loss = 0.0

        val_iter = iter(self.val_loader)

        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Step 1: Update architecture parameters on validation data
            try:
                val_images, val_masks = next(val_iter)
            except StopIteration:
                val_iter = iter(self.val_loader)
                val_images, val_masks = next(val_iter)

            val_images = val_images.to(self.device)
            val_masks = val_masks.to(self.device)

            self.optimizer_alpha.zero_grad()
            val_pred = self.model(val_images)
            loss_arch = self.criterion(val_pred, val_masks)
            loss_arch.backward()
            self.optimizer_alpha.step()
            val_loss += loss_arch.item()

            # Step 2: Update model weights on training data
            self.optimizer_w.zero_grad()
            pred = self.model(images)
            loss_w = self.criterion(pred, masks)
            loss_w.backward()
            self.optimizer_w.step()
            train_loss += loss_w.item()

            if batch_idx % 20 == 0:
                print(
                    f"  Batch {batch_idx}/{len(self.train_loader)} | "
                    f"Train Loss: {loss_w.item():.4f} | "
                    f"Val Loss: {loss_arch.item():.4f}"
                )

        self.scheduler.step()

        n_train = len(self.train_loader)
        n_val = len(self.train_loader)  # Same number of val batches

        return train_loss / n_train, val_loss / n_val

    def search(self) -> Dict:
        """Run the full architecture search"""
        print("=" * 60)
        print("Starting DARTS Architecture Search")
        print("=" * 60)

        best_val_loss = float("inf")
        best_genotype = None

        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print("-" * 40)

            start_time = time.time()
            train_loss, val_loss = self.train_epoch(epoch)
            epoch_time = time.time() - start_time

            # Get current genotype
            genotype = self.model.get_genotype()

            # Log progress
            log_entry = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "time": epoch_time,
                "lr": self.scheduler.get_last_lr()[0],
            }
            self.history.append(log_entry)

            print(
                f"Epoch {epoch + 1} Complete | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Track best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_genotype = genotype
                print(f"  *** New best architecture found! ***")

            # Print architecture weights periodically
            if (epoch + 1) % 10 == 0:
                self._print_arch_weights()

        print("\n" + "=" * 60)
        print("Search Complete!")
        print("=" * 60)

        return best_genotype

    def _print_arch_weights(self):
        """Print the learned architecture weights"""
        print("\nEncoder Architecture Weights:")
        for i, alphas in enumerate(self.model.alphas_encoder):
            weights = F.softmax(alphas, dim=-1).detach().cpu().numpy()
            top_ops = []
            for edge_idx in range(weights.shape[0]):
                best_idx = np.argmax(weights[edge_idx])
                top_ops.append(
                    f"{PRIMITIVES[best_idx]}({weights[edge_idx, best_idx]:.2f})"
                )
            print(f"  Cell {i}: {', '.join(top_ops[:4])}...")

        print("\nDecoder Architecture Weights:")
        for i, alphas in enumerate(self.model.alphas_decoder):
            weights = F.softmax(alphas, dim=-1).detach().cpu().numpy()
            top_ops = []
            for edge_idx in range(weights.shape[0]):
                best_idx = np.argmax(weights[edge_idx])
                top_ops.append(
                    f"{PRIMITIVES[best_idx]}({weights[edge_idx, best_idx]:.2f})"
                )
            print(f"  Cell {i}: {', '.join(top_ops[:4])}...")


# =============================================================================
# Dummy Dataset (Replace with your actual dataset)
# =============================================================================


class DummySegmentationDataset(Dataset):
    """Dummy dataset for testing. Replace with your actual dataset."""

    def __init__(self, num_samples: int = 1000, img_size: int = 256):
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Random image and mask
        image = torch.randn(3, self.img_size, self.img_size)
        mask = torch.randint(0, 2, (1, self.img_size, self.img_size)).float()
        return image, mask


# =============================================================================
# Architecture Export
# =============================================================================


def export_architecture(genotype: Dict, save_path: str):
    """Export discovered architecture to JSON"""
    # Convert to serializable format
    export_dict = {
        "genotype": genotype,
        "primitives": PRIMITIVES,
        "metadata": {
            "search_method": "DARTS",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }

    with open(save_path, "w") as f:
        json.dump(export_dict, f, indent=2)

    print(f"\nArchitecture saved to: {save_path}")


def generate_model_code(genotype: Dict, save_path: str):
    """Generate Python code for the discovered architecture"""

    code = '''"""
Auto-generated UltraFastNet Architecture
Generated by DARTS NAS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscoveredUltraFastNet(nn.Module):
    """
    UltraFastNet with architecture discovered by DARTS.
    """
    
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        
        # TODO: Implement the discovered architecture
        # See genotype in best_architecture.json
        
        # This is a placeholder - you need to implement
        # based on the discovered genotype
        pass
    
    def forward(self, x):
        # TODO: Implement forward pass
        pass


# Discovered Genotype:
'''

    code += f"GENOTYPE = {json.dumps(genotype, indent=4)}\n"

    with open(save_path, "w") as f:
        f.write(code)

    print(f"Model code template saved to: {save_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="DARTS NAS for UltraFastNet")
    parser.add_argument(
        "--data_dir", type=str, default=None, help="Path to dataset directory"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of search epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument("--img_size", type=int, default=256, help="Input image size")
    parser.add_argument(
        "--init_channels", type=int, default=32, help="Initial number of channels"
    )
    parser.add_argument(
        "--num_cells", type=int, default=4, help="Number of cells in encoder/decoder"
    )
    parser.add_argument(
        "--num_nodes", type=int, default=4, help="Number of nodes per cell"
    )
    parser.add_argument(
        "--lr_model", type=float, default=0.025, help="Learning rate for model weights"
    )
    parser.add_argument(
        "--lr_arch",
        type=float,
        default=3e-4,
        help="Learning rate for architecture params",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./nas_output",
        help="Output directory for results",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create datasets
    # TODO: Replace with your actual dataset
    print("\n[WARNING] Using dummy dataset. Replace with your actual data!")
    train_dataset = DummySegmentationDataset(num_samples=500, img_size=args.img_size)
    val_dataset = DummySegmentationDataset(num_samples=100, img_size=args.img_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Create model
    model = SearchableUltraFastNet(
        in_channels=3,
        num_classes=1,
        init_channels=args.init_channels,
        num_cells=args.num_cells,
        num_nodes=args.num_nodes,
    )

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    arch_params = sum(p.numel() for p in model.arch_parameters())
    print(f"\nModel Parameters: {total_params:,}")
    print(f"Architecture Parameters: {arch_params:,}")
    print(f"Weight Parameters: {total_params - arch_params:,}")

    # Create trainer
    trainer = DARTSTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr_model=args.lr_model,
        lr_arch=args.lr_arch,
        epochs=args.epochs,
    )

    # Run search
    best_genotype = trainer.search()

    # Save results
    export_architecture(best_genotype, output_dir / "best_architecture.json")
    generate_model_code(best_genotype, output_dir / "discovered_model.py")

    # Save search history
    import csv

    history_path = output_dir / "search_log.csv"
    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "val_loss", "time", "lr"]
        )
        writer.writeheader()
        writer.writerows(trainer.history)
    print(f"Search history saved to: {history_path}")

    # Print final genotype
    print("\n" + "=" * 60)
    print("DISCOVERED ARCHITECTURE")
    print("=" * 60)
    for key, value in best_genotype.items():
        print(f"\n{key}:")
        for node_idx, edges in enumerate(value):
            print(f"  Node {node_idx}: {edges}")


if __name__ == "__main__":
    main()
