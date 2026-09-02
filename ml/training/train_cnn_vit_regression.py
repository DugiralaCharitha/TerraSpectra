"""
Improved 3D-CNN + Vision Transformer Regression

Predicts disease percentage (0-100) from hyperspectral cubes.

Important:
- Original dataset labels are percentages: 0-100
- Model trains on normalized targets: 0-1
- Metrics are reported on the original percentage scale: 0-100
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
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# EXISTING DATASET UTILITIES
# ============================================================

from ml.training.train_3dcnn import (
    HyperspectralDataset,
    read_samples,
    stratified_split,
)


# ============================================================
# REGRESSION DATASET WRAPPER
# ============================================================

class RegressionDataset(Dataset):
    """
    Wraps the existing HyperspectralDataset.

    The original dataset returns labels as:
        0, 1, 2, ..., 100

    This wrapper converts them to:
        0.00, 0.01, 0.02, ..., 1.00

    This is required because the regression model uses Sigmoid().
    """

    def __init__(
        self,
        samples,
        cubes_dir,
        spatial_size,
        augment=False,
    ):
        self.dataset = HyperspectralDataset(
            samples,
            cubes_dir,
            spatial_size,
            augment=augment,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]

        # Original label is 0-100.
        # Convert to normalized 0-1.
        label = label.float() / 100.0

        return image, label


# ============================================================
# MODEL
# ============================================================

class ImprovedCNNViTRegression(nn.Module):

    def __init__(
        self,
        width=8,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
    ):

        super().__init__()

        # ----------------------------------------------------
        # 3D CNN
        # ----------------------------------------------------

        self.cnn = nn.Sequential(

            # Block 1
            nn.Conv3d(
                1,
                width,
                kernel_size=(7, 3, 3),
                stride=(2, 1, 1),
                padding=(3, 1, 1),
                bias=False,
            ),

            nn.BatchNorm3d(width),

            nn.GELU(),

            nn.MaxPool3d(
                kernel_size=(1, 2, 2)
            ),

            # Block 2
            nn.Conv3d(
                width,
                width * 2,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm3d(width * 2),

            nn.GELU(),

            nn.MaxPool3d(
                kernel_size=(2, 2, 2)
            ),

            # Block 3
            nn.Conv3d(
                width * 2,
                width * 4,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm3d(width * 4),

            nn.GELU(),

            # Block 4
            nn.Conv3d(
                width * 4,
                width * 4,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm3d(width * 4),

            nn.GELU(),
        )

        channels = width * 4

        # ----------------------------------------------------
        # SPECTRAL COLLAPSE
        # ----------------------------------------------------

        self.token_projection = nn.Conv3d(
            channels,
            embed_dim,
            kernel_size=(16, 1, 1),
            stride=(16, 1, 1),
        )

        # Expected spatial feature map:
        # 8 x 8 = 64 tokens

        self.num_tokens = 64

        # ----------------------------------------------------
        # CLASS TOKEN
        # ----------------------------------------------------

        self.class_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                embed_dim,
            )
        )

        # ----------------------------------------------------
        # POSITION EMBEDDING
        # ----------------------------------------------------

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                self.num_tokens + 1,
                embed_dim,
            )
        )

        # ----------------------------------------------------
        # TRANSFORMER
        # ----------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # ----------------------------------------------------
        # REGRESSION HEAD
        # ----------------------------------------------------

        self.norm = nn.LayerNorm(embed_dim)

        self.regressor = nn.Sequential(

            nn.Linear(
                embed_dim,
                32,
            ),

            nn.GELU(),

            nn.Dropout(0.20),

            nn.Linear(
                32,
                1,
            ),

            # Output guaranteed to be 0-1.
            nn.Sigmoid(),
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

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        # 3D CNN
        x = self.cnn(x)

        # Spectral projection
        x = self.token_projection(x)

        # Remove spectral dimension
        x = x.squeeze(2)

        # [B, C, 8, 8]
        #
        # ->
        #
        # [B, 64, C]

        x = x.flatten(
            2
        ).transpose(
            1,
            2,
        )

        # Add class token

        class_token = self.class_token.expand(
            x.size(0),
            -1,
            -1,
        )

        x = torch.cat(
            (
                class_token,
                x,
            ),
            dim=1,
        )

        # Add positional information

        x = (
            x
            + self.position_embedding
        )

        # Transformer

        x = self.transformer(x)

        # Class token

        x = self.norm(
            x[:, 0]
        )

        # Regression

        prediction = self.regressor(x)

        return prediction.squeeze(1)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    predictions,
    targets,
):

    predictions = np.asarray(
        predictions,
        dtype=np.float64,
    )

    targets = np.asarray(
        targets,
        dtype=np.float64,
    )

    # Both inputs are normalized 0-1.
    # Convert to percentage for reporting.

    predictions_percentage = predictions * 100.0
    targets_percentage = targets * 100.0

    errors = (
        predictions_percentage
        - targets_percentage
    )

    mae = np.mean(
        np.abs(errors)
    )

    rmse = np.sqrt(
        np.mean(
            errors ** 2
        )
    )

    ss_res = np.sum(
        errors ** 2
    )

    ss_tot = np.sum(
        (
            targets_percentage
            - np.mean(targets_percentage)
        ) ** 2
    )

    if ss_tot > 0:

        r2 = (
            1.0
            - ss_res / ss_tot
        )

    else:

        r2 = 0.0

    return (
        float(mae),
        float(rmse),
        float(r2),
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader,
    device,
):

    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)

            predictions.extend(
                outputs.cpu()
                .numpy()
                .tolist()
            )

            targets.extend(
                labels.cpu()
                .numpy()
                .tolist()
            )

    return calculate_metrics(
        predictions,
        targets,
    )


# ============================================================
# TRAINING EPOCH
# ============================================================

def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):

    model.train()

    total_loss = 0.0
    total_examples = 0

    for images, labels in loader:

        images = images.to(device)

        labels = labels.float().to(device)

        # Clear gradients

        optimizer.zero_grad(
            set_to_none=True
        )

        # Forward

        predictions = model(images)

        # Both predictions and labels are 0-1.

        loss = criterion(
            predictions,
            labels,
        )

        # Backpropagation

        loss.backward()

        # Gradient clipping

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        # Update

        optimizer.step()

        # Accumulate loss

        total_loss += (
            loss.item()
            * labels.size(0)
        )

        total_examples += (
            labels.size(0)
        )

    return (
        total_loss
        / total_examples
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--spatial-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "ml"
            / "experiments"
            / "cnn_vit_regression_v4"
        ),
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
    )

    args = parser.parse_args()

    # ========================================================
    # REPRODUCIBILITY
    # ========================================================

    random.seed(args.seed)

    np.random.seed(args.seed)

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # ========================================================
    # PATHS
    # ========================================================

    csv_path = (
        PROJECT_ROOT
        / "train_final.csv"
    )

    cubes_dir = (
        PROJECT_ROOT
        / "ot"
        / "ot"
    )

    # ========================================================
    # DATASET
    # ========================================================

    print(
        "Loading dataset..."
    )

    samples = read_samples(
        csv_path,
        cubes_dir,
    )

    print(
        f"Samples: {len(samples)}"
    )

    # ========================================================
    # TARGET INFORMATION
    # ========================================================

    raw_labels = np.asarray(
        [
            sample.label
            for sample in samples
        ],
        dtype=np.float32,
    )

    print(
        f"Raw target range: "
        f"{raw_labels.min():.1f} - "
        f"{raw_labels.max():.1f}"
    )

    print(
        f"Raw target mean: "
        f"{raw_labels.mean():.2f}"
    )

    print(
        f"Raw target std: "
        f"{raw_labels.std():.2f}"
    )

    # --------------------------------------------------------
    # Verify raw labels are percentages.
    # --------------------------------------------------------

    if (
        raw_labels.min() < 0
        or raw_labels.max() > 100
    ):

        raise RuntimeError(
            "Unexpected target range. "
            "Expected labels between 0 and 100."
        )

    normalized_labels = (
        raw_labels / 100.0
    )

    print(
        f"Normalized target range: "
        f"{normalized_labels.min():.3f} - "
        f"{normalized_labels.max():.3f}"
    )

    # ========================================================
    # STRATIFIED SPLIT
    # ========================================================

    train_samples, validation_samples = (
        stratified_split(
            samples,
            0.20,
            args.seed,
        )
    )

    print(
        f"Train: "
        f"{len(train_samples)}"
    )

    print(
        f"Validation: "
        f"{len(validation_samples)}"
    )

    # ========================================================
    # REGRESSION DATASETS
    # ========================================================

    train_data = RegressionDataset(
        train_samples,
        cubes_dir,
        args.spatial_size,
        augment=True,
    )

    validation_data = RegressionDataset(
        validation_samples,
        cubes_dir,
        args.spatial_size,
        augment=False,
    )

    # ========================================================
    # DATA LOADERS
    # ========================================================

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

    # ========================================================
    # MODEL
    # ========================================================

    model = ImprovedCNNViTRegression(
        width=8,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
    ).to(device)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters: "
        f"{parameters:,}"
    )

    # ========================================================
    # FORWARD VERIFICATION
    # ========================================================

    images, labels = next(
        iter(train_loader)
    )

    with torch.no_grad():

        output = model(
            images.to(device)
        )

    print(
        f"Input shape: "
        f"{tuple(images.shape)}"
    )

    print(
        f"Output shape: "
        f"{tuple(output.shape)}"
    )

    print(
        f"Example predictions "
        f"(normalized 0-1): "
        f"{output[:5].cpu().numpy()}"
    )

    print(
        f"Example predictions "
        f"(percentage): "
        f"{output[:5].cpu().numpy() * 100.0}"
    )

    print(
        f"Target examples "
        f"(normalized 0-1): "
        f"{labels[:5].numpy()}"
    )

    print(
        f"Target examples "
        f"(percentage): "
        f"{labels[:5].numpy() * 100.0}"
    )

    # --------------------------------------------------------
    # Output range verification
    # --------------------------------------------------------

    if (
        torch.any(output < 0.0)
        or torch.any(output > 1.0)
    ):

        raise RuntimeError(
            "Model output is outside "
            "the expected 0-1 range."
        )

    # --------------------------------------------------------
    # Target range verification
    # --------------------------------------------------------

    if (
        torch.any(labels < 0.0)
        or torch.any(labels > 1.0)
    ):

        raise RuntimeError(
            "Regression targets are outside "
            "the expected 0-1 range."
        )

    print(
        "Output range verification: PASSED"
    )

    print(
        "Target normalization verification: PASSED"
    )

    # ========================================================
    # VERIFY ONLY
    # ========================================================

    if args.verify_only:

        print()

        print(
            "SUCCESS: "
            "Improved 3D-CNN + ViT "
            "forward pass and target "
            "normalization work correctly."
        )

        return

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # LOSS
    # ========================================================

    criterion = nn.SmoothL1Loss(
        beta=0.05
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    # ========================================================
    # LR SCHEDULER
    # ========================================================

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    # ========================================================
    # BEST MODEL
    # ========================================================

    best_mae = float("inf")

    best_rmse = float("inf")

    best_r2 = -float("inf")

    best_epoch = 0

    stale_epochs = 0

    history = []

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        started = time.perf_counter()

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        mae, rmse, r2 = evaluate(
            model,
            validation_loader,
            device,
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(mae)

        elapsed = (
            time.perf_counter()
            - started
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            f"Epoch {epoch:02d}/{args.epochs}: "
            f"Loss={train_loss:.5f} "
            f"MAE={mae:.3f}% "
            f"RMSE={rmse:.3f}% "
            f"R2={r2:.4f} "
            f"LR={current_lr:.2e} "
            f"({elapsed:.1f}s)"
        )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "mae": float(mae),
                "rmse": float(rmse),
                "r2": float(r2),
                "learning_rate": float(current_lr),
                "seconds": float(elapsed),
            }
        )

        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if mae < best_mae:

            best_mae = mae

            best_rmse = rmse

            best_r2 = r2

            best_epoch = epoch

            stale_epochs = 0

            torch.save(
                {
                    "epoch": epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "best_mae":
                        best_mae,

                    "best_rmse":
                        best_rmse,

                    "best_r2":
                        best_r2,

                    "spatial_size":
                        args.spatial_size,

                    "seed":
                        args.seed,

                    "architecture":
                        "Improved 3D-CNN + ViT",

                    "target_scale":
                        "0-1 normalized from original 0-100 percentage",

                    "metric_scale":
                        "0-100 percentage",

                    "parameters":
                        parameters,
                },

                args.output_dir
                / "best_model.pt",
            )

            print(
                f"  New best model saved. "
                f"MAE={best_mae:.3f}%"
            )

        else:

            stale_epochs += 1

        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if stale_epochs >= args.patience:

            print(
                f"Early stopping after "
                f"{stale_epochs} stale epochs."
            )

            break

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics = {

        "model":
            "Improved 3D-CNN + ViT Regression",

        "architecture":
            "3D-CNN spectral-spatial extraction + ViT spatial tokens",

        "device":
            str(device),

        "samples":
            len(samples),

        "train_samples":
            len(train_samples),

        "validation_samples":
            len(validation_samples),

        "spatial_size":
            args.spatial_size,

        "epochs_requested":
            args.epochs,

        "best_epoch":
            best_epoch,

        "best_mae":
            best_mae,

        "best_rmse":
            best_rmse,

        "best_r2":
            best_r2,

        "target_original_scale":
            "0-100 percentage",

        "target_training_scale":
            "0-1 normalized",

        "metric_reporting_scale":
            "0-100 percentage",

        "loss":
            "SmoothL1Loss",

        "optimizer":
            "AdamW",

        "learning_rate":
            args.learning_rate,

        "weight_decay":
            1e-4,

        "seed":
            args.seed,

        "parameters":
            parameters,

        "history":
            history,
    }

    with (
        args.output_dir
        / "metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()

    print(
        "================================"
    )

    print(
        "IMPROVED CNN + ViT COMPLETE"
    )

    print(
        "================================"
    )

    print(
        f"Best MAE:  "
        f"{best_mae:.3f}%"
    )

    print(
        f"Best RMSE: "
        f"{best_rmse:.3f}%"
    )

    print(
        f"Best R2:   "
        f"{best_r2:.4f}"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print()

    print(
        "Results saved to:"
    )

    print(
        args.output_dir
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()