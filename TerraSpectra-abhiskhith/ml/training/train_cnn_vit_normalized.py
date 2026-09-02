"""
CNN + Vision Transformer Regression
Training-set-fitted per-band normalization.

Predicts disease percentage (0-100) from hyperspectral cubes.

Pipeline:

Hyperspectral cube
        ↓
Spatial downsampling
        ↓
Training-set-fitted per-band normalization
        ↓
3D-CNN spectral-spatial feature extraction
        ↓
Spectral collapse
        ↓
Spatial tokens
        ↓
Vision Transformer
        ↓
Normalized regression (0-1)
        ↓
Disease percentage (0-100)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_SHAPE = (128, 128, 125)
NUM_BANDS = 125


# ============================================================
# SAMPLE
# ============================================================

@dataclass(frozen=True)
class Sample:
    filename: str
    label: int


# ============================================================
# READ SAMPLES
# ============================================================

def read_samples(csv_path: Path, cubes_dir: Path):

    samples = []
    skipped = []

    with csv_path.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            filename = row["id"]
            label = int(row["label"])

            path = cubes_dir / filename

            if not path.is_file():

                skipped.append(
                    f"{filename}: missing"
                )

                continue

            try:

                cube = np.load(
                    path,
                    mmap_mode="r"
                )

                if cube.shape != EXPECTED_SHAPE:

                    skipped.append(
                        f"{filename}: {cube.shape}"
                    )

                    continue

            except Exception as error:

                skipped.append(
                    f"{filename}: {error}"
                )

                continue

            samples.append(
                Sample(
                    filename,
                    label
                )
            )

    if not samples:

        raise RuntimeError(
            "No valid hyperspectral cubes found."
        )

    print(
        f"Valid samples: {len(samples)}"
    )

    print(
        f"Skipped samples: {len(skipped)}"
    )

    return samples


# ============================================================
# STRATIFIED SPLIT
# ============================================================

def stratified_split(
    samples,
    validation_fraction=0.20,
    seed=42
):

    grouped = defaultdict(list)

    for sample in samples:

        grouped[sample.label].append(
            sample
        )

    rng = random.Random(seed)

    train = []
    validation = []

    for label_samples in grouped.values():

        label_samples = list(
            label_samples
        )

        rng.shuffle(
            label_samples
        )

        n_val = max(
            1,
            round(
                len(label_samples)
                * validation_fraction
            )
        )

        n_val = min(
            n_val,
            len(label_samples) - 1
        )

        validation.extend(
            label_samples[:n_val]
        )

        train.extend(
            label_samples[n_val:]
        )

    rng.shuffle(train)
    rng.shuffle(validation)

    return train, validation


# ============================================================
# CALCULATE TRAINING-SET NORMALIZATION
# ============================================================

def calculate_band_statistics(
    samples,
    cubes_dir,
    spatial_size
):

    print()
    print(
        "Calculating training-set spectral statistics..."
    )

    factor = 128 // spatial_size

    band_sum = np.zeros(
        NUM_BANDS,
        dtype=np.float64
    )

    band_sum_sq = np.zeros(
        NUM_BANDS,
        dtype=np.float64
    )

    band_count = np.zeros(
        NUM_BANDS,
        dtype=np.int64
    )

    for index, sample in enumerate(samples):

        path = cubes_dir / sample.filename

        cube = np.load(
            path
        ).astype(
            np.float32,
            copy=False
        )

        if cube.shape != EXPECTED_SHAPE:

            raise ValueError(
                f"Invalid cube shape for "
                f"{sample.filename}: "
                f"{cube.shape}"
            )

        # ----------------------------------------------------
        # Spatial downsampling
        # ----------------------------------------------------

        cube = cube.reshape(
            spatial_size,
            factor,
            spatial_size,
            factor,
            NUM_BANDS
        )

        cube = cube.mean(
            axis=(1, 3),
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Calculate statistics per spectral band
        # ----------------------------------------------------

        values = cube.reshape(
            -1,
            NUM_BANDS
        ).astype(
            np.float64
        )

        band_sum += values.sum(
            axis=0
        )

        band_sum_sq += np.square(
            values
        ).sum(
            axis=0
        )

        band_count += values.shape[0]

        if (
            index + 1
        ) % 200 == 0:

            print(
                f"  Processed "
                f"{index + 1}/"
                f"{len(samples)} samples"
            )

    means = (
        band_sum
        / band_count
    )

    variances = (
        band_sum_sq
        / band_count
        - means ** 2
    )

    variances = np.maximum(
        variances,
        1e-8
    )

    stds = np.sqrt(
        variances
    )

    print(
        "Training-set normalization statistics calculated."
    )

    print(
        f"Mean range: "
        f"{means.min():.3f} - "
        f"{means.max():.3f}"
    )

    print(
        f"Std range: "
        f"{stds.min():.3f} - "
        f"{stds.max():.3f}"
    )

    return (
        means.astype(np.float32),
        stds.astype(np.float32)
    )


# ============================================================
# REGRESSION DATASET
# ============================================================

class HyperspectralRegressionDataset(Dataset):

    def __init__(
        self,
        samples,
        cubes_dir,
        spatial_size,
        band_means,
        band_stds,
        augment=False
    ):

        self.samples = list(samples)

        self.cubes_dir = Path(
            cubes_dir
        )

        self.spatial_size = (
            spatial_size
        )

        self.augment = augment

        self.band_means = (
            np.asarray(
                band_means,
                dtype=np.float32
            )
        )

        self.band_stds = (
            np.asarray(
                band_stds,
                dtype=np.float32
            )
        )

        if 128 % spatial_size != 0:

            raise ValueError(
                "spatial_size must divide 128."
            )

        if self.band_means.shape != (
            NUM_BANDS,
        ):

            raise ValueError(
                "Invalid band means shape."
            )

        if self.band_stds.shape != (
            NUM_BANDS,
        ):

            raise ValueError(
                "Invalid band std shape."
            )

        self.factor = (
            128 // spatial_size
        )

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(self, index):

        sample = self.samples[index]

        path = (
            self.cubes_dir
            / sample.filename
        )

        cube = np.load(
            path
        ).astype(
            np.float32,
            copy=False
        )

        if cube.shape != EXPECTED_SHAPE:

            raise ValueError(
                f"Invalid cube shape: "
                f"{cube.shape}"
            )

        # ----------------------------------------------------
        # Spatial downsampling
        # ----------------------------------------------------

        cube = cube.reshape(
            self.spatial_size,
            self.factor,
            self.spatial_size,
            self.factor,
            NUM_BANDS
        )

        cube = cube.mean(
            axis=(1, 3),
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Spatial augmentation
        # ----------------------------------------------------

        if self.augment:

            if np.random.random() < 0.5:

                cube = cube[::-1, :, :]

            if np.random.random() < 0.5:

                cube = cube[:, ::-1, :]

        # ----------------------------------------------------
        # Per-band standardization
        #
        # IMPORTANT:
        # Means/stds were calculated using TRAINING
        # samples only.
        # ----------------------------------------------------

        cube = (
            cube
            - self.band_means.reshape(
                1,
                1,
                NUM_BANDS
            )
        )

        cube = (
            cube
            / self.band_stds.reshape(
                1,
                1,
                NUM_BANDS
            )
        )

        # ----------------------------------------------------
        # H,W,B -> B,H,W
        # ----------------------------------------------------

        cube = cube.transpose(
            2,
            0,
            1
        )

        cube = np.ascontiguousarray(
            cube,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Add channel dimension
        #
        # B,H,W
        #    ↓
        # 1,B,H,W
        # ----------------------------------------------------

        image = torch.from_numpy(
            cube
        ).unsqueeze(0)

        # ----------------------------------------------------
        # Target normalization
        #
        # Original target:
        # 0-100
        #
        # Training target:
        # 0-1
        # ----------------------------------------------------

        target = (
            float(sample.label)
            / 100.0
        )

        label = torch.tensor(
            target,
            dtype=torch.float32
        )

        return image, label


# ============================================================
# IMPROVED CNN + VIT
# ============================================================

class ImprovedCNNViTRegression(nn.Module):

    def __init__(
        self,
        width=8,
        embed_dim=64,
        num_heads=4,
        num_layers=2
    ):

        super().__init__()

        # ----------------------------------------------------
        # 3D CNN
        # ----------------------------------------------------

        self.cnn = nn.Sequential(

            nn.Conv3d(
                1,
                width,
                kernel_size=(7, 3, 3),
                stride=(2, 1, 1),
                padding=(3, 1, 1),
                bias=False
            ),

            nn.BatchNorm3d(
                width
            ),

            nn.GELU(),

            nn.MaxPool3d(
                kernel_size=(1, 2, 2)
            ),

            nn.Conv3d(
                width,
                width * 2,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm3d(
                width * 2
            ),

            nn.GELU(),

            nn.MaxPool3d(
                kernel_size=(2, 2, 2)
            ),

            nn.Conv3d(
                width * 2,
                width * 4,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm3d(
                width * 4
            ),

            nn.GELU(),

            nn.Conv3d(
                width * 4,
                width * 4,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm3d(
                width * 4
            ),

            nn.GELU(),
        )

        channels = width * 4

        # ----------------------------------------------------
        # Spectral projection
        # ----------------------------------------------------

        self.token_projection = nn.Conv3d(
            channels,
            embed_dim,
            kernel_size=(16, 1, 1),
            stride=(16, 1, 1)
        )

        # 8 x 8 spatial tokens
        self.num_tokens = 64

        # ----------------------------------------------------
        # CLASS TOKEN
        # ----------------------------------------------------

        self.class_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                embed_dim
            )
        )

        # ----------------------------------------------------
        # POSITION EMBEDDING
        # ----------------------------------------------------

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                self.num_tokens + 1,
                embed_dim
            )
        )

        # ----------------------------------------------------
        # TRANSFORMER
        # ----------------------------------------------------

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 2,
                dropout=0.10,
                activation="gelu",
                batch_first=True,
                norm_first=False
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers
            )
        )

        # ----------------------------------------------------
        # REGRESSION HEAD
        # ----------------------------------------------------

        self.norm = nn.LayerNorm(
            embed_dim
        )

        self.regressor = nn.Sequential(

            nn.Linear(
                embed_dim,
                32
            ),

            nn.GELU(),

            nn.Dropout(
                0.20
            ),

            nn.Linear(
                32,
                1
            ),

            nn.Sigmoid()
        )

        # ----------------------------------------------------
        # Initialization
        # ----------------------------------------------------

        nn.init.trunc_normal_(
            self.class_token,
            std=0.02
        )

        nn.init.trunc_normal_(
            self.position_embedding,
            std=0.02
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        x = self.cnn(x)

        x = self.token_projection(
            x
        )

        x = x.squeeze(2)

        # [B,C,H,W]
        # ->
        # [B,64,C]

        x = x.flatten(
            2
        ).transpose(
            1,
            2
        )

        class_token = (
            self.class_token.expand(
                x.size(0),
                -1,
                -1
            )
        )

        x = torch.cat(
            (
                class_token,
                x
            ),
            dim=1
        )

        x = (
            x
            + self.position_embedding
        )

        x = self.transformer(
            x
        )

        x = self.norm(
            x[:, 0]
        )

        prediction = self.regressor(
            x
        )

        return prediction.squeeze(1)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    predictions,
    targets
):

    predictions = np.asarray(
        predictions,
        dtype=np.float64
    )

    targets = np.asarray(
        targets,
        dtype=np.float64
    )

    # Convert 0-1 to percentage
    predictions_pct = (
        predictions * 100.0
    )

    targets_pct = (
        targets * 100.0
    )

    errors = (
        predictions_pct
        - targets_pct
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
            targets_pct
            - np.mean(targets_pct)
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
        float(r2)
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader,
    device,
    criterion
):

    model.eval()

    predictions = []
    targets = []

    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            total_examples += (
                labels.size(0)
            )

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

    mae, rmse, r2 = (
        calculate_metrics(
            predictions,
            targets
        )
    )

    average_loss = (
        total_loss
        / total_examples
    )

    return (
        average_loss,
        mae,
        rmse,
        r2
    )


# ============================================================
# TRAINING EPOCH
# ============================================================

def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    total_loss = 0.0
    total_examples = 0

    for images, labels in loader:

        images = images.to(
            device
        )

        labels = labels.to(
            device
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        predictions = model(
            images
        )

        loss = criterion(
            predictions,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

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
        "--csv",
        type=Path,
        default=(
            PROJECT_ROOT
            / "train_final.csv"
        )
    )

    parser.add_argument(
        "--cubes-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "ot"
            / "ot"
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "ml"
            / "experiments"
            / "cnn_vit_band_normalized"
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8
    )

    parser.add_argument(
        "--spatial-size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--verify-only",
        action="store_true"
    )

    args = parser.parse_args()

    # ========================================================
    # REPRODUCIBILITY
    # ========================================================

    random.seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    torch.manual_seed(
        args.seed
    )

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
    # DATASET
    # ========================================================

    print(
        "Loading dataset..."
    )

    samples = read_samples(
        args.csv,
        args.cubes_dir
    )

    print(
        f"Samples: {len(samples)}"
    )

    # ========================================================
    # TARGET INFORMATION
    # ========================================================

    raw_targets = np.asarray(
        [
            sample.label
            for sample in samples
        ],
        dtype=np.float32
    )

    print(
        f"Raw target range: "
        f"{raw_targets.min():.1f} - "
        f"{raw_targets.max():.1f}"
    )

    print(
        f"Raw target mean: "
        f"{raw_targets.mean():.2f}"
    )

    print(
        f"Raw target std: "
        f"{raw_targets.std():.2f}"
    )

    normalized_targets = (
        raw_targets / 100.0
    )

    print(
        f"Normalized target range: "
        f"{normalized_targets.min():.3f} - "
        f"{normalized_targets.max():.3f}"
    )

    # ========================================================
    # SPLIT
    # ========================================================

    train_samples, validation_samples = (
        stratified_split(
            samples,
            args.validation_fraction,
            args.seed
        )
    )

    print(
        f"Train: {len(train_samples)}"
    )

    print(
        f"Validation: {len(validation_samples)}"
    )

    # ========================================================
    # FIT NORMALIZATION ON TRAINING DATA ONLY
    # ========================================================

    band_means, band_stds = (
        calculate_band_statistics(
            train_samples,
            args.cubes_dir,
            args.spatial_size
        )
    )

    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = (
        HyperspectralRegressionDataset(
            train_samples,
            args.cubes_dir,
            args.spatial_size,
            band_means,
            band_stds,
            augment=True
        )
    )

    validation_dataset = (
        HyperspectralRegressionDataset(
            validation_samples,
            args.cubes_dir,
            args.spatial_size,
            band_means,
            band_stds,
            augment=False
        )
    )

    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = ImprovedCNNViTRegression(
        width=8,
        embed_dim=64,
        num_heads=4,
        num_layers=2
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
        f"Input range: "
        f"[{images.min().item():.4f}, "
        f"{images.max().item():.4f}]"
    )

    print(
        "Example predictions "
        f"(normalized 0-1): "
        f"{output[:5].cpu().numpy()}"
    )

    print(
        "Example predictions "
        f"(percentage): "
        f"{output[:5].cpu().numpy() * 100.0}"
    )

    print(
        "Target examples "
        f"(normalized 0-1): "
        f"{labels[:5].numpy()}"
    )

    print(
        "Target examples "
        f"(percentage): "
        f"{labels[:5].numpy() * 100.0}"
    )

    # ========================================================
    # VERIFY INPUT
    # ========================================================

    if not torch.isfinite(
        images
    ).all():

        raise RuntimeError(
            "Input contains NaN or infinity."
        )

    if not torch.isfinite(
        output
    ).all():

        raise RuntimeError(
            "Model output contains NaN or infinity."
        )

    if (
        torch.any(output < 0.0)
        or torch.any(output > 1.0)
    ):

        raise RuntimeError(
            "Model output is outside "
            "the expected 0-1 range."
        )

    if (
        torch.any(labels < 0.0)
        or torch.any(labels > 1.0)
    ):

        raise RuntimeError(
            "Target is outside "
            "the expected 0-1 range."
        )

    print(
        "Input verification: PASSED"
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
            "CNN + ViT with training-set "
            "per-band normalization "
            "forward pass works."
        )

        return

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # SAVE NORMALIZATION STATISTICS
    # ========================================================

    np.savez(
        args.output_dir
        / "band_normalization.npz",
        means=band_means,
        stds=band_stds
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
        weight_decay=1e-4
    )

    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2
        )
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
        args.epochs + 1
    ):

        started = time.perf_counter()

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss, mae, rmse, r2 = evaluate(
            model,
            validation_loader,
            device,
            criterion
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(
            mae
        )

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
            f"ValLoss={val_loss:.5f} "
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
                "train_loss": float(
                    train_loss
                ),
                "validation_loss": float(
                    val_loss
                ),
                "mae": float(
                    mae
                ),
                "rmse": float(
                    rmse
                ),
                "r2": float(
                    r2
                ),
                "learning_rate": float(
                    current_lr
                ),
                "seconds": float(
                    elapsed
                )
            }
        )

        # ====================================================
        # BEST MODEL
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

                    "normalization":
                        "training-set per-band standardization",

                    "target_scale":
                        "0-1 normalized during training",

                    "metric_scale":
                        "0-100 percentage",

                    "parameters":
                        parameters,

                    "band_means":
                        band_means,

                    "band_stds":
                        band_stds
                },

                args.output_dir
                / "best_model.pt"
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

        if (
            stale_epochs
            >= args.patience
        ):

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
            "CNN + ViT Regression",

        "architecture":
            "3D-CNN spectral-spatial extraction + ViT spatial tokens",

        "normalization":
            "training-set per-band standardization",

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

        "target_training_scale":
            "0-1",

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
            history
    }

    with (
        args.output_dir
        / "metrics.json"
    ).open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print(
        "================================"
    )

    print(
        "CNN + ViT BAND NORMALIZATION "
        "COMPLETE"
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
# ENTRY PO0i9INT
# ============================================================

if __name__ == "__main__":
    main()