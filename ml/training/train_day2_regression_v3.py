from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "train_final.csv"
CUBES_DIR = PROJECT_ROOT / "ot" / "ot"
OUTPUT_DIR = PROJECT_ROOT / "ml" / "experiments" / "day2_regression_v3"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
EPOCHS = 10
BATCH_SIZE = 16
SPATIAL_SIZE = 32
LEARNING_RATE = 0.0005


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class HyperspectralDataset(Dataset):

    def __init__(self, dataframe, normalization, augment=False):

        self.df = dataframe.reset_index(drop=True)
        self.normalization = normalization
        self.augment = augment
        self.factor = 128 // SPATIAL_SIZE

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        cube = np.load(
            CUBES_DIR / row["id"]
        ).astype(np.float32)

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

        if self.augment:

            if np.random.random() < 0.5:
                cube = cube[::-1, :, :]

            if np.random.random() < 0.5:
                cube = cube[:, ::-1, :]

        cube = np.ascontiguousarray(cube)

        # ------------------------------------------------
        # NORMALIZATION
        # ------------------------------------------------

        if self.normalization == "global":
            cube = cube / 28906.0

        elif self.normalization == "max":
            cube = cube / max(
                float(cube.max()),
                1.0
            )

        elif self.normalization == "percentile":
            scale = np.percentile(
                cube,
                99.5
            )

            cube = cube / max(
                float(scale),
                1.0
            )

        cube = np.clip(
            cube,
            0.0,
            1.5
        )

        cube = cube.transpose(
            2,
            0,
            1
        )

        cube = np.ascontiguousarray(cube)

        image = torch.from_numpy(
            cube
        ).unsqueeze(0)

        target = torch.tensor(
            float(row["label"]) / 100.0,
            dtype=torch.float32
        )

        return image, target


class Strong3DCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv3d(
                1, 16,
                kernel_size=(7, 3, 3),
                stride=(2, 1, 1),
                padding=(3, 1, 1),
                bias=False
            ),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(2),

            nn.Conv3d(
                16, 32,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(2),

            nn.Conv3d(
                32, 64,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(
                64, 64,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool3d(
                (1, 1, 1)
            )
        )

        self.regressor = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.30),

            nn.Linear(64, 32),

            nn.ReLU(inplace=True),

            nn.Dropout(0.20),

            nn.Linear(32, 1),

            nn.Sigmoid()
        )

    def forward(self, x):

        return self.regressor(
            self.features(x)
        ).squeeze(1)


def metrics(predictions, targets):

    predictions = np.asarray(
        predictions,
        dtype=np.float64
    ) * 100.0

    targets = np.asarray(
        targets,
        dtype=np.float64
    ) * 100.0

    errors = predictions - targets

    mae = np.mean(np.abs(errors))

    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    ss_res = np.sum(
        (targets - predictions) ** 2
    )

    ss_tot = np.sum(
        (targets - targets.mean()) ** 2
    )

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0
        else 0.0
    )

    return mae, rmse, r2


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    training = optimizer is not None

    model.train(training)

    predictions = []
    targets = []

    total_loss = 0.0

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
        )

        targets.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

    mae, rmse, r2 = metrics(
        predictions,
        targets
    )

    return (
        total_loss / len(loader.dataset),
        mae,
        rmse,
        r2
    )


def train_normalization(
    train_df,
    val_df,
    normalization,
    device
):

    print()
    print("=" * 45)
    print(
        f"NORMALIZATION: {normalization}"
    )
    print("=" * 45)

    train_dataset = HyperspectralDataset(
        train_df,
        normalization,
        augment=True
    )

    val_dataset = HyperspectralDataset(
        val_df,
        normalization,
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

    model = Strong3DCNN().to(device)

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

    best_mae = float("inf")
    best_result = None
    best_state = None

    for epoch in range(1, EPOCHS + 1):

        start = time.perf_counter()

        run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        _, mae, rmse, r2 = run_epoch(
            model,
            val_loader,
            criterion,
            None,
            device
        )

        scheduler.step(mae)

        elapsed = time.perf_counter() - start

        print(
            f"{normalization} "
            f"Epoch {epoch:02d}: "
            f"MAE={mae:.3f} "
            f"RMSE={rmse:.3f} "
            f"R2={r2:.4f} "
            f"({elapsed:.1f}s)"
        )

        if mae < best_mae:

            best_mae = mae

            best_result = (
                mae,
                rmse,
                r2
            )

            best_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }

    return best_result, best_state


def main():

    print("Loading dataset...")

    df = pd.read_csv(
        CSV_PATH
    )

    print(
        f"Samples: {len(df)}"
    )

    indices = np.arange(
        len(df)
    )

    rng = np.random.RandomState(
        SEED
    )

    rng.shuffle(indices)

    n_val = int(
        len(df) * 0.20
    )

    val_df = df.iloc[
        indices[:n_val]
    ].reset_index(drop=True)

    train_df = df.iloc[
        indices[n_val:]
    ].reset_index(drop=True)

    print(
        f"Train: {len(train_df)}"
    )

    print(
        f"Validation: {len(val_df)}"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    results = []

    overall_best = None

    for normalization in [
        "global",
        "max",
        "percentile"
    ]:

        result, state = train_normalization(
            train_df,
            val_df,
            normalization,
            device
        )

        mae, rmse, r2 = result

        results.append({
            "normalization": normalization,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        if (
            overall_best is None
            or mae < overall_best[1]
        ):

            overall_best = (
                normalization,
                mae,
                rmse,
                r2,
                state
            )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_DIR / "comparison.csv",
        index=False
    )

    normalization, mae, rmse, r2, state = (
        overall_best
    )

    torch.save(
        {
            "model_state_dict": state,
            "normalization": normalization,
            "spatial_size": SPATIAL_SIZE,
            "seed": SEED,
            "best_mae": mae,
            "best_rmse": rmse,
            "best_r2": r2
        },
        OUTPUT_DIR / "best_model.pt"
    )

    print()
    print("=" * 45)
    print("DAY 2 V3 COMPLETE")
    print("=" * 45)

    print(
        results_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"BEST NORMALIZATION: {normalization}"
    )

    print(
        f"Best MAE:  {mae:.3f}"
    )

    print(
        f"Best RMSE: {rmse:.3f}"
    )

    print(
        f"Best R²:   {r2:.4f}"
    )

    print(
        f"Saved: {OUTPUT_DIR / 'best_model.pt'}"
    )


if __name__ == "__main__":
    main()