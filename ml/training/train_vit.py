""""Train a CPU-safe 3D-CNN + Vision Transformer hybrid
for hyperspectral crop classification.

Pipeline:
    Hyperspectral cube
        ↓
    3D-CNN feature extraction
        ↓
    Spatial feature tokens
        ↓
    Vision Transformer
        ↓
    101-class classification

Verify first:
    py -3.12 ml/training/train_vit.py --verify-only

Train:
    py -3.12 ml/training/train_vit.py --epochs 15 --batch-size 4
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from ml.training.train_3dcnn import (
    HyperspectralDataset,
    read_samples,
    stratified_split,
)


# ============================================================
# 3D-CNN + VISION TRANSFORMER HYBRID
# ============================================================

class CNNViTHybrid(nn.Module):
    """
    Hybrid hyperspectral classifier.

    Stage 1:
        3D-CNN extracts spectral-spatial features.

    Stage 2:
        CNN feature map is converted into spatial tokens.

    Stage 3:
        Vision Transformer learns relationships between tokens.

    Stage 4:
        Classification head predicts the crop class.
    """

    def __init__(
        self,
        num_classes: int,
        width: int = 8,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()

        # ----------------------------------------------------
        # 3D-CNN FEATURE EXTRACTOR
        # ----------------------------------------------------

        self.cnn = nn.Sequential(

            # Input:
            # [B, 1, 125, 32, 32]
            nn.Conv3d(
                1,
                width,
                kernel_size=3,
                stride=(2, 1, 1),
                padding=1,
            ),
            nn.BatchNorm3d(width),
            nn.ReLU(inplace=True),

            # Spatial reduction:
            # 32x32 -> 16x16
            nn.MaxPool3d(
                kernel_size=(1, 2, 2)
            ),

            # Spectral + spatial feature extraction
            nn.Conv3d(
                width,
                width * 2,
                kernel_size=3,
                stride=(2, 1, 1),
                padding=1,
            ),
            nn.BatchNorm3d(width * 2),
            nn.ReLU(inplace=True),

            # Spatial:
            # 16x16 -> 8x8
            nn.MaxPool3d(
                kernel_size=(1, 2, 2)
            ),

            # Deeper spectral-spatial features
            nn.Conv3d(
                width * 2,
                width * 4,
                kernel_size=3,
                stride=(2, 1, 1),
                padding=1,
            ),
            nn.BatchNorm3d(width * 4),
            nn.ReLU(inplace=True),
        )

        cnn_channels = width * 4

        # ----------------------------------------------------
        # CNN FEATURE MAP -> TRANSFORMER TOKENS
        # ----------------------------------------------------

        # The CNN output has approximately:
        #
        # [B, width*4, spectral, 8, 8]
        #
        # We collapse the remaining spectral dimension while
        # converting each spatial region into an embedding.

        self.token_projection = nn.Conv3d(
            cnn_channels,
            embed_dim,
            kernel_size=(16, 2, 2),
            stride=(16, 2, 2),
        )

        # After projection:
        #
        # [B, embed_dim, 1, 4, 4]
        #
        # Therefore:
        #
        # 4 x 4 = 16 spatial tokens

        num_tokens = 16

        # ----------------------------------------------------
        # CLASS TOKEN
        # ----------------------------------------------------

        self.class_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        # ----------------------------------------------------
        # POSITION EMBEDDING
        # ----------------------------------------------------

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                num_tokens + 1,
                embed_dim
            )
        )

        # ----------------------------------------------------
        # VISION TRANSFORMER
        # ----------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.15,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # ----------------------------------------------------
        # FINAL CLASSIFIER
        # ----------------------------------------------------

        self.norm = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(0.30)

        self.classifier = nn.Linear(
            embed_dim,
            num_classes,
        )

        # ----------------------------------------------------
        # INITIALIZATION
        # ----------------------------------------------------

        nn.init.trunc_normal_(
            self.class_token,
            std=0.02,
        )

        nn.init.trunc_normal_(
            self.position_embedding,
            std=0.02,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:

        # ----------------------------------------------------
        # STEP 1: 3D-CNN
        # ----------------------------------------------------

        x = self.cnn(x)

        # ----------------------------------------------------
        # STEP 2: CNN FEATURES -> TOKENS
        # ----------------------------------------------------

        x = self.token_projection(x)

        # Expected:
        # [B, embed_dim, 1, 4, 4]

        # Remove spectral dimension
        x = x.squeeze(2)

        # Convert:
        # [B, embed_dim, 4, 4]
        #
        # into:
        # [B, 16, embed_dim]

        x = x.flatten(2).transpose(1, 2)

        # ----------------------------------------------------
        # STEP 3: ADD CLASS TOKEN
        # ----------------------------------------------------

        class_token = self.class_token.expand(
            x.size(0),
            -1,
            -1,
        )

        x = torch.cat(
            (class_token, x),
            dim=1,
        )

        # ----------------------------------------------------
        # STEP 4: POSITION INFORMATION
        # ----------------------------------------------------

        x = x + self.position_embedding

        # ----------------------------------------------------
        # STEP 5: VISION TRANSFORMER
        # ----------------------------------------------------

        x = self.transformer(x)

        # ----------------------------------------------------
        # STEP 6: CLASS TOKEN
        # ----------------------------------------------------

        x = self.norm(x[:, 0])

        x = self.dropout(x)

        # ----------------------------------------------------
        # STEP 7: CLASSIFICATION
        # ----------------------------------------------------

        return self.classifier(x)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(
                    set_to_none=True
                )

            logits = model(images)

            loss = criterion(
                logits,
                labels,
            )

            if training:

                loss.backward()

                optimizer.step()

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            total_correct += (
                logits.argmax(1) == labels
            ).sum().item()

            total_examples += labels.size(0)

    return (
        total_loss / total_examples,
        total_correct / total_examples,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "train_final.csv",
    )

    parser.add_argument(
        "--cubes-dir",
        type=Path,
        default=PROJECT_ROOT / "ot" / "ot",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "ml"
            / "experiments"
            / "cnn_vit_hybrid"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--spatial-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--embed-dim",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not 0 < args.validation_fraction < 1:
        raise ValueError(
            "--validation-fraction must be between 0 and 1."
        )

    # --------------------------------------------------------
    # REPRODUCIBILITY
    # --------------------------------------------------------

    random.seed(args.seed)

    np.random.seed(args.seed)

    torch.manual_seed(args.seed)

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    samples = read_samples(
        args.csv,
        args.cubes_dir,
    )

    num_classes = (
        max(
            sample.label
            for sample in samples
        )
        + 1
    )

    # --------------------------------------------------------
    # STRATIFIED SPLIT
    # --------------------------------------------------------

    train_samples, validation_samples = (
        stratified_split(
            samples,
            args.validation_fraction,
            args.seed,
        )
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_data = HyperspectralDataset(
        train_samples,
        args.cubes_dir,
        args.spatial_size,
        augment=True,
    )

    validation_data = HyperspectralDataset(
        validation_samples,
        args.cubes_dir,
        args.spatial_size,
        augment=False,
    )

    # --------------------------------------------------------
    # DATA LOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = CNNViTHybrid(
        num_classes=num_classes,
        width=args.width,
        embed_dim=args.embed_dim,
        num_heads=4,
        num_layers=2,
    ).to(device)

    # --------------------------------------------------------
    # VERIFY FORWARD PASS
    # --------------------------------------------------------

    images, labels = next(
        iter(train_loader)
    )

    with torch.no_grad():

        logits = model(
            images.to(device)
        )

    print(
        f"Device: {device}; "
        f"valid samples: {len(samples)}; "
        f"classes: {num_classes}"
    )

    print(
        f"Split: "
        f"train={len(train_data)}, "
        f"validation={len(validation_data)}"
    )

    print(
        f"Batch input: {tuple(images.shape)}; "
        f"logits: {tuple(logits.shape)}; "
        f"parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    print(
        f"Labels: {labels.tolist()}"
    )

    if args.verify_only:

        print(
            "Verification passed: "
            "3D-CNN + ViT forward pass is working."
        )

        return

    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-3,
    )

    # --------------------------------------------------------
    # LEARNING RATE SCHEDULER
    # --------------------------------------------------------

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )
    )

    # --------------------------------------------------------
    # TRAINING VARIABLES
    # --------------------------------------------------------

    best_accuracy = -1.0

    best_loss = float("inf")

    stale_epochs = 0

    history = []

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        started = time.perf_counter()

        # Training
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        # Validation
        validation_loss, validation_accuracy = (
            run_epoch(
                model,
                validation_loader,
                criterion,
                None,
                device,
            )
        )

        # Learning-rate update
        scheduler.step(validation_loss)

        elapsed = (
            time.perf_counter()
            - started
        )

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "learning_rate": (
                optimizer.param_groups[0]["lr"]
            ),
            "seconds": elapsed,
        }

        history.append(metrics)

        print(
            "Epoch "
            f"{epoch:02d}/{args.epochs}: "
            f"train acc={train_accuracy:.3%}; "
            f"val acc={validation_accuracy:.3%}; "
            f"val loss={validation_loss:.4f}; "
            f"{elapsed:.1f}s"
        )

        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        improved = (
            validation_accuracy
            > best_accuracy
            or (
                validation_accuracy
                == best_accuracy
                and validation_loss
                < best_loss
            )
        )

        if improved:

            best_accuracy = (
                validation_accuracy
            )

            best_loss = (
                validation_loss
            )

            stale_epochs = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "validation_accuracy": (
                        validation_accuracy
                    ),
                    "validation_loss": (
                        validation_loss
                    ),
                    "num_classes": (
                        num_classes
                    ),
                    "spatial_size": (
                        args.spatial_size
                    ),
                    "width": args.width,
                    "embed_dim": (
                        args.embed_dim
                    ),
                },
                (
                    args.output_dir
                    / "best_model.pt"
                ),
            )

            print(
                f"  New best model: "
                f"{best_accuracy:.3%}"
            )

        else:

            stale_epochs += 1

            if stale_epochs >= 6:

                print(
                    "Early stopping: "
                    "no validation improvement "
                    "for 6 epochs."
                )

                break

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    with (
        args.output_dir
        / "metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            {
                "configuration": vars(args),
                "device": str(device),
                "valid_samples": len(samples),
                "num_classes": num_classes,
                "history": history,
            },
            handle,
            indent=2,
            default=str,
        )

    print()
    print(
        "Training complete."
    )

    print(
        f"Best validation accuracy: "
        f"{best_accuracy:.3%}"
    )

    print(
        f"Results saved to: "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()