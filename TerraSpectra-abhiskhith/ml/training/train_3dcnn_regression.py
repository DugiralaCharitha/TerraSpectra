import argparse
import csv
import json
import random
import time
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

NORMALIZATION_DIVISOR = 4095.0


# ============================================================
# SAMPLE
# ============================================================

@dataclass(frozen=True)
class Sample:
    filename: str
    label: float


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
            label = float(row["label"])

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
# TRAIN / VALIDATION SPLIT
# ============================================================

def train_validation_split(
    samples,
    validation_fraction=0.20,
    seed=42
):

    samples = list(samples)

    rng = random.Random(seed)

    rng.shuffle(samples)

    n_val = round(
        len(samples) * validation_fraction
    )

    n_val = max(
        1,
        min(
            n_val,
            len(samples) - 1
        )
    )

    validation = samples[:n_val]
    train = samples[n_val:]

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

        self.spatial_size = spatial_size

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

        # H,W,B → B,H,W

        cube = cube.transpose(
            2,
            0,
            1
        )

        cube = np.ascontiguousarray(
            cube
        )

        # Add channel dimension
        #
        # B,H,W
        # ↓
        # 1,B,H,W

        image = torch.from_numpy(
            cube
        ).unsqueeze(0)

        # Regression target

        label = torch.tensor(
            sample.label,
            dtype=torch.float32
        )

        return image, label


# ============================================================
# FAST 3D CNN
# ============================================================

class Fast3DCNN(nn.Module):

    def __init__(self):

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

        # One output = disease percentage

        self.regressor = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.25),

            nn.Linear(32, 1)
        )


    def forward(self, x):

        x = self.features(x)

        return self.regressor(x).squeeze(1)


# ============================================================
# REGRESSION METRICS
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

    errors = (
        predictions
        - targets
    )

    mae = np.mean(
        np.abs(errors)
    )

    rmse = np.sqrt(
        np.mean(
            errors ** 2
        )
    )

    target_mean = np.mean(
        targets
    )

    ss_res = np.sum(
        (targets - predictions) ** 2
    )

    ss_tot = np.sum(
        (targets - target_mean) ** 2
    )

    if ss_tot == 0:

        r2 = 0.0

    else:

        r2 = (
            1.0
            - ss_res / ss_tot
        )

    return mae, rmse, r2


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

    predictions = []

    targets = []

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

            predictions.extend(
                outputs.detach()
                .cpu()
                .numpy()
                .tolist()
            )

            targets.extend(
                labels.detach()
                .cpu()
                .numpy()
                .tolist()
            )

    average_loss = (
        total_loss
        / len(loader.dataset)
    )

    mae, rmse, r2 = calculate_metrics(
        predictions,
        targets
    )

    return (
        average_loss,
        mae,
        rmse,
        r2
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
            / "3dcnn_regression"
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

    labels = np.asarray(
        [
            sample.label
            for sample in samples
        ],
        dtype=np.float32
    )

    print(
        f"Target range: "
        f"{labels.min():.1f} - "
        f"{labels.max():.1f}"
    )

    print(
        f"Target mean: "
        f"{labels.mean():.2f}"
    )

    print(
        f"Target std: "
        f"{labels.std():.2f}"
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train_samples, val_samples = (
        train_validation_split(
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

    model = Fast3DCNN().to(
        device
    )

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
        f"Predictions: "
        f"{tuple(outputs.shape)}"
    )

    print(
        f"Input range: "
        f"[{images.min():.4f}, "
        f"{images.max():.4f}]"
    )

    print(
        f"Example predictions: "
        f"{outputs[:5].cpu().numpy()}"
    )

    if args.verify_only:

        print()

        print(
            "SUCCESS: "
            "3D-CNN regression verification passed."
        )

        return

    # --------------------------------------------------------
    # Regression loss
    # --------------------------------------------------------

    criterion = nn.SmoothL1Loss()

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

    best_mae = float("inf")

    best_rmse = float("inf")

    best_r2 = -float("inf")

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

        (
            train_loss,
            train_mae,
            train_rmse,
            train_r2
        ) = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        (
            val_loss,
            val_mae,
            val_rmse,
            val_r2
        ) = run_epoch(
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
            f"train MAE={train_mae:.3f}; "
            f"val MAE={val_mae:.3f}; "
            f"val RMSE={val_rmse:.3f}; "
            f"val R²={val_r2:.4f}; "
            f"{seconds:.1f}s"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mae": train_mae,
                "train_rmse": train_rmse,
                "train_r2": train_r2,
                "validation_loss": val_loss,
                "validation_mae": val_mae,
                "validation_rmse": val_rmse,
                "validation_r2": val_r2,
                "seconds": seconds,
                "learning_rate": (
                    optimizer.param_groups[0]["lr"]
                )
            }
        )

        # Lower MAE = better

        improved = (
            val_mae < best_mae
        )

        if improved:

            best_mae = val_mae

            best_rmse = val_rmse

            best_r2 = val_r2

            stale = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "spatial_size":
                        args.spatial_size,

                    "normalization_divisor":
                        NORMALIZATION_DIVISOR,

                    "validation_mae":
                        val_mae,

                    "validation_rmse":
                        val_rmse,

                    "validation_r2":
                        val_r2,

                    "epoch":
                        epoch
                },
                args.output_dir
                / "best_model.pt"
            )

            print(
                f"New best model: "
                f"MAE={val_mae:.3f}"
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
                "spatial_size":
                    args.spatial_size,
                "normalization_divisor":
                    NORMALIZATION_DIVISOR,
                "target_type":
                    "disease_percentage_regression",
                "target_min":
                    float(labels.min()),
                "target_max":
                    float(labels.max()),
                "best_validation_mae":
                    best_mae,
                "best_validation_rmse":
                    best_rmse,
                "best_validation_r2":
                    best_r2,
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
        f"Best validation MAE: "
        f"{best_mae:.3f}"
    )

    print(
        f"Best validation RMSE: "
        f"{best_rmse:.3f}"
    )

    print(
        f"Best validation R²: "
        f"{best_r2:.4f}"
    )

    print()

    print(
        "Results saved to:"
    )

    print(
        args.output_dir
    )


if __name__ == "__main__":
    main()