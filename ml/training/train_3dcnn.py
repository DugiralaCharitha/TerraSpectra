"""
Fast CPU-safe 3D-CNN for hyperspectral crop classification.

Dataset:
    1990 valid hyperspectral cubes
    101 classes
    Original cube: 128 x 128 x 125
    Pooled cube:   16 x 16 x 125

The model remains a genuine 3D-CNN:
    spectral + spatial dimensions are processed together.

Verify:
    py -3.12 -m ml.training.train_3dcnn --verify-only

Train:
    py -3.12 -m ml.training.train_3dcnn
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PROJECT SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_SHAPE = (128, 128, 125)

# Dataset-wide maximum discovered during validation.
NORMALIZATION_DIVISOR = 28906.0


# ============================================================
# SAMPLE
# ============================================================

@dataclass(frozen=True)
class Sample:
    filename: str
    label: int


# ============================================================
# READ DATASET
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

        # Always leave at least one
        # sample for training.
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
# DATASET
# ============================================================

class HyperspectralDataset(Dataset):

    def __init__(
        self,
        samples,
        cubes_dir,
        spatial_size=16,
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

        if 128 % spatial_size != 0:

            raise ValueError(
                "spatial_size must divide 128."
            )

        self.factor = (
            128 // spatial_size
        )


    def __len__(self):

        return len(self.samples)


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
        # Spatial average pooling
        #
        # 128 x 128 x 125
        #          ↓
        # 16 x 16 x 125
        #
        # All 125 spectral bands remain.
        # ----------------------------------------------------

        factor = self.factor

        cube = cube.reshape(
            self.spatial_size,
            factor,
            self.spatial_size,
            factor,
            125
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
        # Normalize
        # ----------------------------------------------------

        cube = (
            cube
            / NORMALIZATION_DIVISOR
        )

        # H,W,B -> B,H,W
        cube = cube.transpose(
            2, 0, 1
        )

        cube = np.ascontiguousarray(
            cube
        )

        # Add channel dimension:
        #
        # B,H,W
        # ↓
        # 1,B,H,W
        image = torch.from_numpy(
            cube
        ).unsqueeze(0)

        label = torch.tensor(
            sample.label,
            dtype=torch.long
        )

        return image, label


# ============================================================
# FAST 3D CNN
# ============================================================

class Fast3DCNN(nn.Module):

    """
    Lightweight spectral-spatial 3D-CNN.

    Input:
        [batch, 1, 125, 16, 16]

    The first convolution reduces:
        spectral dimension
        spatial dimensions

    This makes CPU training considerably faster
    than the previous 1.29M parameter network.
    """

    def __init__(
        self,
        num_classes
    ):

        super().__init__()

        self.features = nn.Sequential(

            # ------------------------------------------------
            # Block 1
            # ------------------------------------------------

            nn.Conv3d(
                1,
                8,
                kernel_size=(7, 3, 3),
                stride=(2, 2, 2),
                padding=(3, 1, 1),
                bias=False
            ),

            nn.BatchNorm3d(8),

            nn.ReLU(inplace=True),

            # ------------------------------------------------
            # Block 2
            # ------------------------------------------------

            nn.Conv3d(
                8,
                16,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm3d(16),

            nn.ReLU(inplace=True),

            nn.MaxPool3d(
                kernel_size=(2, 2, 2)
            ),

            # ------------------------------------------------
            # Block 3
            # ------------------------------------------------

            nn.Conv3d(
                16,
                32,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm3d(32),

            nn.ReLU(inplace=True),

            # ------------------------------------------------
            # Global pooling
            # ------------------------------------------------

            nn.AdaptiveAvgPool3d(
                (1, 1, 1)
            )
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.25),

            nn.Linear(
                32,
                num_classes
            )
        )


    def forward(self, x):

        x = self.features(x)

        return self.classifier(x)


# ============================================================
# TRAIN / VALIDATION EPOCH
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    training = optimizer is not None

    if training:

        model.train()

    else:

        model.eval()

    total_loss = 0.0
    total_correct = 0
    total = 0

    if training:

        context = torch.enable_grad()

    else:

        context = torch.no_grad()

    with context:

        for images, labels in loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            if training:

                optimizer.zero_grad(
                    set_to_none=True
                )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            if training:

                loss.backward()

                optimizer.step()

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            total_correct += (
                predictions == labels
            ).sum().item()

            total += (
                labels.size(0)
            )

    return (
        total_loss / total,
        total_correct / total
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
            / "3dcnn_fast"
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
        default=16
    )

    parser.add_argument(
        "--spatial-size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001
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

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    random.seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    torch.manual_seed(
        args.seed
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    samples = read_samples(
        args.csv,
        args.cubes_dir
    )

    num_classes = (
        max(
            sample.label
            for sample in samples
        )
        + 1
    )

    print(
        f"Classes: {num_classes}"
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train_samples, val_samples = (
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
        f"Validation: {len(val_samples)}"
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = HyperspectralDataset(
        train_samples,
        args.cubes_dir,
        args.spatial_size,
        augment=True
    )

    val_dataset = HyperspectralDataset(
        val_samples,
        args.cubes_dir,
        args.spatial_size,
        augment=False
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = Fast3DCNN(
        num_classes
    ).to(device)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Model parameters: "
        f"{parameters:,}"
    )

    # --------------------------------------------------------
    # Verify forward pass
    # --------------------------------------------------------

    images, labels = next(
        iter(train_loader)
    )

    with torch.no_grad():

        outputs = model(
            images.to(device)
        )

    print(
        f"Batch input: "
        f"{tuple(images.shape)}"
    )

    print(
        f"Logits: "
        f"{tuple(outputs.shape)}"
    )

    print(
        f"Input range: "
        f"[{images.min():.4f}, "
        f"{images.max():.4f}]"
    )

    if args.verify_only:

        print()
        print(
            "SUCCESS: "
            "Fast 3D-CNN verification passed."
        )

        return

    # --------------------------------------------------------
    # Class-weighted loss
    # --------------------------------------------------------

    counts = Counter(
        sample.label
        for sample in train_samples
    )

    weights = []

    for label in range(
        num_classes
    ):

        count = counts.get(
            label,
            1
        )

        weights.append(
            1.0 / count
        )

    weights = torch.tensor(
        weights,
        dtype=torch.float32
    )

    weights = (
        weights
        / weights.mean()
    )

    weights = weights.to(
        device
    )

    criterion = nn.CrossEntropyLoss(
        weight=weights
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2
        )
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_accuracy = -1.0
    best_loss = float("inf")
    stale = 0
    history = []

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for epoch in range(
        1,
        args.epochs + 1
    ):

        start = time.perf_counter()

        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
            None,
            device
        )

        scheduler.step(
            val_loss
        )

        seconds = (
            time.perf_counter()
            - start
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs}: "
            f"train acc={train_acc:.3%}; "
            f"val acc={val_acc:.3%}; "
            f"val loss={val_loss:.4f}; "
            f"{seconds:.1f}s"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "validation_loss": val_loss,
                "validation_accuracy": val_acc,
                "seconds": seconds,
                "learning_rate": (
                    optimizer.param_groups[0]["lr"]
                )
            }
        )

        improved = (
            val_acc > best_accuracy
            or (
                val_acc == best_accuracy
                and val_loss < best_loss
            )
        )

        if improved:

            best_accuracy = val_acc
            best_loss = val_loss
            stale = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "num_classes":
                        num_classes,

                    "spatial_size":
                        args.spatial_size,

                    "normalization_divisor":
                        NORMALIZATION_DIVISOR,

                    "validation_accuracy":
                        val_acc,

                    "validation_loss":
                        val_loss,

                    "epoch":
                        epoch
                },
                args.output_dir
                / "best_model.pt"
            )

            print(
                f"New best model: "
                f"{val_acc:.3%}"
            )

        else:

            stale += 1

            if stale >= args.patience:

                print(
                    "Early stopping."
                )

                break

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    with (
        args.output_dir
        / "metrics.json"
    ).open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "device": str(device),
                "valid_samples": len(samples),
                "classes": num_classes,
                "spatial_size":
                    args.spatial_size,
                "normalization_divisor":
                    NORMALIZATION_DIVISOR,
                "best_validation_accuracy":
                    best_accuracy,
                "history": history
            },
            file,
            indent=2
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
        "Results saved to:"
    )

    print(
        args.output_dir
    )


if __name__ == "__main__":
    main()