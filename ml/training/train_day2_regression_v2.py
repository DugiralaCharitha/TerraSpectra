from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = PROJECT_ROOT / "train_final.csv"
CUBES_DIR = PROJECT_ROOT / "ot" / "ot"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "experiments"
    / "day2_regression_v2"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42
EPOCHS = 15
BATCH_SIZE = 16
SPATIAL_SIZE = 32
LEARNING_RATE = 0.0005

NORMALIZATION_DIVISOR = 28906.0


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DATASET
# ============================================================

class HyperspectralDataset(Dataset):

    def __init__(self, dataframe, augment=False):

        self.df = dataframe.reset_index(drop=True)
        self.augment = augment

        self.factor = 128 // SPATIAL_SIZE

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        cube = np.load(
            CUBES_DIR / row["id"]
        ).astype(np.float32)

        # 128 x 128 x 125
        # ->
        # 32 x 32 x 125

        f = self.factor

        cube = cube.reshape(
            SPATIAL_SIZE,
            f,
            SPATIAL_SIZE,
            f,
            125
        )

        cube = cube.mean(
            axis=(1, 3),
            dtype=np.float32
        )

        # Spatial augmentation

        if self.augment:

            if np.random.random() < 0.5:
                cube = cube[::-1, :, :]

            if np.random.random() < 0.5:
                cube = cube[:, ::-1, :]

        cube = np.ascontiguousarray(cube)

        # Normalize

        cube = cube / NORMALIZATION_DIVISOR

        # H,W,B -> B,H,W

        cube = cube.transpose(2, 0, 1)

        cube = np.ascontiguousarray(cube)

        # 1,B,H,W

        image = torch.from_numpy(cube).unsqueeze(0)

        # Disease percentage -> 0-1

        target = float(row["label"]) / 100.0

        target = torch.tensor(
            target,
            dtype=torch.float32
        )

        return image, target


# ============================================================
# MODEL
# ============================================================

class Strong3DCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            # Block 1

            nn.Conv3d(
                1,
                16,
                kernel_size=(7, 3, 3),
                stride=(2, 1, 1),
                padding=(3, 1, 1),
                bias=False
            ),

            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(
                kernel_size=(2, 2, 2)
            ),

            # Block 2

            nn.Conv3d(
                16,
                32,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(
                kernel_size=(2, 2, 2)
            ),

            # Block 3

            nn.Conv3d(
                32,
                64,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            # Block 4

            nn.Conv3d(
                64,
                64,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            # Global pooling

            nn.AdaptiveAvgPool3d(
                (1, 1, 1)
            )
        )

        self.regressor = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.30),

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.20),

            nn.Linear(
                32,
                1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.features(x)

        return self.regressor(x).squeeze(1)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(predictions, targets):

    predictions = np.asarray(
        predictions,
        dtype=np.float64
    ) * 100.0

    targets = np.asarray(
        targets,
        dtype=np.float64
    ) * 100.0

    errors = predictions - targets

    mae = np.mean(
        np.abs(errors)
    )

    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    target_mean = np.mean(targets)

    ss_res = np.sum(
        (targets - predictions) ** 2
    )

    ss_tot = np.sum(
        (targets - target_mean) ** 2
    )

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0
        else 0.0
    )

    return mae, rmse, r2


# ============================================================
# EPOCH
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

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            if training:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0
                )

                optimizer.step()

        total_loss += (
            loss.item()
            * images.size(0)
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

    return average_loss, mae, rmse, r2


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading dataset...")

    df = pd.read_csv(CSV_PATH)

    print(
        f"Samples: {len(df)}"
    )

    print(
        f"Labels: {df['label'].min()} - "
        f"{df['label'].max()}"
    )

    print(
        f"Mean: {df['label'].mean():.2f}"
    )

    print(
        f"Std: {df['label'].std():.2f}"
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    indices = np.arange(len(df))

    rng = np.random.RandomState(SEED)

    rng.shuffle(indices)

    n_val = int(
        len(indices) * 0.20
    )

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    print(
        f"Train: {len(train_df)}"
    )

    print(
        f"Validation: {len(val_df)}"
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = HyperspectralDataset(
        train_df,
        augment=True
    )

    val_dataset = HyperspectralDataset(
        val_df,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
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
    # Model
    # --------------------------------------------------------

    model = Strong3DCNN().to(device)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters: {parameters:,}"
    )

    # --------------------------------------------------------
    # Forward verification
    # --------------------------------------------------------

    images, labels = next(
        iter(train_loader)
    )

    with torch.no_grad():

        outputs = model(
            images.to(device)
        )

    print(
        f"Input: {tuple(images.shape)}"
    )

    print(
        f"Output: {tuple(outputs.shape)}"
    )

    print(
        "Forward pass: OK"
    )

    # --------------------------------------------------------
    # Loss / optimizer
    # --------------------------------------------------------

    criterion = nn.SmoothL1Loss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_mae = float("inf")

    best_metrics = None

    history = []

    best_path = (
        OUTPUT_DIR
        / "best_model.pt"
    )

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        start = time.perf_counter()

        train_loss, train_mae, train_rmse, train_r2 = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_mae, val_rmse, val_r2 = run_epoch(
            model,
            val_loader,
            criterion,
            None,
            device
        )

        scheduler.step(val_loss)

        elapsed = time.perf_counter() - start

        print(
            f"Epoch {epoch:02d}/{EPOCHS}: "
            f"train MAE={train_mae:.3f}; "
            f"val MAE={val_mae:.3f}; "
            f"val RMSE={val_rmse:.3f}; "
            f"val R²={val_r2:.4f}; "
            f"{elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_mae": train_mae,
            "train_rmse": train_rmse,
            "train_r2": train_r2,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "val_r2": val_r2
        })

        if val_mae < best_mae:

            best_mae = val_mae

            best_metrics = {
                "mae": val_mae,
                "rmse": val_rmse,
                "r2": val_r2
            }

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "spatial_size": SPATIAL_SIZE,
                    "normalization_divisor": NORMALIZATION_DIVISOR,
                    "seed": SEED,
                    "best_mae": val_mae,
                    "best_rmse": val_rmse,
                    "best_r2": val_r2
                },
                best_path
            )

            print(
                f"  New best: MAE={val_mae:.3f}"
            )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / "history.csv",
        index=False
    )

    print()
    print("================================")
    print("DAY 2 REGRESSION V2 COMPLETE")
    print("================================")

    print(
        f"Best MAE:  {best_metrics['mae']:.3f}"
    )

    print(
        f"Best RMSE: {best_metrics['rmse']:.3f}"
    )

    print(
        f"Best R²:   {best_metrics['r2']:.4f}"
    )

    print(
        f"Saved: {best_path}"
    )


if __name__ == "__main__":
    main()