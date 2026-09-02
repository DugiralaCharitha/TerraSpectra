from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from ml.training.train_3dcnn import Fast3DCNN


# ============================================================
# PROJECT SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_SHAPE = (128, 128, 125)

NORMALIZATION_DIVISOR = 28906.0


# ============================================================
# SAMPLE
# ============================================================

class Sample:
    def __init__(self, filename: str, label: int):
        self.filename = filename
        self.label = label


# ============================================================
# READ DATASET
# ============================================================

def read_samples(csv_path: Path, cubes_dir: Path):

    samples = []
    skipped = []

    print("Loading dataset...")

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

    print(
        f"Valid samples: {len(samples)}"
    )

    print(
        f"Skipped samples: {len(skipped)}"
    )

    if not samples:

        raise RuntimeError(
            "No valid hyperspectral cubes found."
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

        if len(label_samples) == 1:

            train.extend(
                label_samples
            )

            continue

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
# DATASET
# ============================================================

class HyperspectralDataset(Dataset):

    def __init__(
        self,
        samples,
        cubes_dir,
        spatial_size=16
    ):

        self.samples = list(samples)

        self.cubes_dir = Path(
            cubes_dir
        )

        self.spatial_size = (
            spatial_size
        )

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

        # B,H,W → 1,B,H,W

        image = torch.from_numpy(
            cube
        ).unsqueeze(0)

        label = torch.tensor(
            sample.label,
            dtype=torch.long
        )

        return image, label


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    seed = 42

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    csv_path = (
        PROJECT_ROOT
        / "train_final.csv"
    )

    cubes_dir = (
        PROJECT_ROOT
        / "ot"
        / "ot"
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
        csv_path,
        cubes_dir
    )

    num_classes = (
        max(
            sample.label
            for sample in samples
        )
        + 1
    )

    print(
        f"Total valid samples: "
        f"{len(samples)}"
    )

    print(
        f"Number of classes: "
        f"{num_classes}"
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train_samples, val_samples = (
        stratified_split(
            samples,
            validation_fraction=0.20,
            seed=seed
        )
    )

    print(
        f"Training samples: "
        f"{len(train_samples)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_samples)}"
    )

    # --------------------------------------------------------
    # Dataset objects
    # --------------------------------------------------------

    train_dataset = HyperspectralDataset(
        train_samples,
        cubes_dir,
        spatial_size=16
    )

    val_dataset = HyperspectralDataset(
        val_samples,
        cubes_dir,
        spatial_size=16
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # FAST 3D CNN
    # --------------------------------------------------------

    model = Fast3DCNN(
        num_classes
    ).to(device)

    print(
        f"Model: Fast3DCNN"
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
    # Forward-pass test
    # --------------------------------------------------------

    images, labels = next(
        iter(train_loader)
    )

    print(
        f"Batch input shape: "
        f"{tuple(images.shape)}"
    )

    print(
        f"Labels shape: "
        f"{tuple(labels.shape)}"
    )

    with torch.no_grad():

        outputs = model(
            images.to(device)
        )

    print(
        f"Model output shape: "
        f"{tuple(outputs.shape)}"
    )

    print(
        f"Expected output shape: "
        f"(batch, {num_classes})"
    )

    print(
        f"Input range: "
        f"[{images.min():.4f}, "
        f"{images.max():.4f}]"
    )

    # --------------------------------------------------------
    # Check output dimensions
    # --------------------------------------------------------

    if outputs.shape[1] != num_classes:

        raise RuntimeError(
            "Model output class count does "
            "not match dataset classes."
        )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.01,
        weight_decay=1e-4
    )

    # --------------------------------------------------------
    # Tiny overfit test
    #
    # Use only a small subset.
    # The model should be capable of
    # learning this tiny dataset.
    # --------------------------------------------------------

    overfit_samples = train_samples[:8]

    overfit_dataset = HyperspectralDataset(
        overfit_samples,
        cubes_dir,
        spatial_size=16
    )

    overfit_loader = DataLoader(
        overfit_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )

    print()
    print(
        "Starting tiny overfit test..."
    )

    print(
        f"Overfit samples: "
        f"{len(overfit_samples)}"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    epochs = 100

    best_accuracy = 0.0

    for epoch in range(
        1,
        epochs + 1
    ):

        model.train()

        total_loss = 0.0

        total_correct = 0

        total = 0

        for batch_images, batch_labels in overfit_loader:

            batch_images = batch_images.to(
                device
            )

            batch_labels = batch_labels.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            predictions = model(
                batch_images
            )

            loss = criterion(
                predictions,
                batch_labels
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                * batch_labels.size(0)
            )

            predicted_labels = (
                predictions.argmax(
                    dim=1
                )
            )

            total_correct += (
                predicted_labels
                == batch_labels
            ).sum().item()

            total += (
                batch_labels.size(0)
            )

        accuracy = (
            total_correct
            / total
        )

        average_loss = (
            total_loss
            / total
        )

        best_accuracy = max(
            best_accuracy,
            accuracy
        )

        print(
            f"Epoch {epoch:02d}/{epochs}: "
            f"loss={average_loss:.4f}; "
            f"accuracy={accuracy:.3%}"
        )

        # ----------------------------------------------------
        # Success condition
        # ----------------------------------------------------

        if accuracy >= 0.95:

            print()
            print(
                "SUCCESS: "
                "Fast 3D-CNN passed the "
                "tiny overfit test."
            )

            print(
                f"Accuracy reached: "
                f"{accuracy:.3%}"
            )

            return  

    # --------------------------------------------------------
    # If we reach here
    # --------------------------------------------------------

    print()
    print(
        "Overfit test finished."
    )

    print(
        f"Best accuracy: "
        f"{best_accuracy:.3%}"
    )

    if best_accuracy < 0.95:

        print(
            "The model did not reach "
            "95% accuracy on the tiny "
            "overfit dataset."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()