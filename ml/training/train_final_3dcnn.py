from pathlib import Path
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "train_final.csv"
CUBES = ROOT / "ot" / "ot"
OUT = ROOT / "ml" / "experiments" / "final_3dcnn"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
EPOCHS = 20
BATCH = 8
SIZE = 64
LR = 0.0003
SCALE = 28906.0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class CubeDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.f = 128 // SIZE

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = np.load(CUBES / r["id"]).astype(np.float32)

        f = self.f
        x = x.reshape(SIZE, f, SIZE, f, 125).mean(
            axis=(1, 3), dtype=np.float32
        )

        if self.augment:
            if np.random.rand() < .5:
                x = x[::-1]
            if np.random.rand() < .5:
                x = x[:, ::-1]

        x = np.ascontiguousarray(x / SCALE)
        x = np.transpose(x, (2, 0, 1))
        x = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0)

        y = torch.tensor(
            float(r["label"]) / 100.0,
            dtype=torch.float32
        )

        return x, y


class Final3DCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 16, (7,3,3), stride=(2,1,1),
                      padding=(3,1,1), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 96, 3, padding=1, bias=False),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool3d(1)
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, 48),
            nn.ReLU(inplace=True),
            nn.Dropout(.25),
            nn.Linear(48, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


def metrics(p, y):
    p = np.asarray(p) * 100
    y = np.asarray(y) * 100

    mae = np.mean(np.abs(p-y))
    rmse = np.sqrt(np.mean((p-y)**2))

    ss_res = np.sum((y-p)**2)
    ss_tot = np.sum((y-y.mean())**2)

    r2 = 1 - ss_res / ss_tot

    return mae, rmse, r2


def epoch(model, loader, loss_fn, opt, device):
    train = opt is not None
    model.train(train)

    p, y = [], []

    for x, t in loader:
        x, t = x.to(device), t.to(device)

        if train:
            opt.zero_grad()

        with torch.set_grad_enabled(train):
            out = model(x)
            loss = loss_fn(out, t)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                opt.step()

        p.extend(out.detach().cpu().numpy())
        y.extend(t.detach().cpu().numpy())

    return metrics(p, y)


def main():
    df = pd.read_csv(CSV)

    idx = np.arange(len(df))
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)

    n = int(len(df) * .2)

    val = df.iloc[idx[:n]]
    train = df.iloc[idx[n:]]

    print(f"Train: {len(train)}")
    print(f"Validation: {len(val)}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Spatial size: {SIZE}")

    tr = DataLoader(
        CubeDataset(train, True),
        batch_size=BATCH,
        shuffle=True,
        num_workers=0
    )

    va = DataLoader(
        CubeDataset(val, False),
        batch_size=BATCH,
        shuffle=False,
        num_workers=0
    )

    model = Final3DCNN().to(device)

    print(
        "Parameters:",
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    loss_fn = nn.SmoothL1Loss()

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=.5, patience=2
    )

    best = float("inf")
    best_result = None

    for e in range(1, EPOCHS + 1):

        start = time.time()

        epoch(model, tr, loss_fn, opt, device)

        mae, rmse, r2 = epoch(
            model, va, loss_fn, None, device
        )

        scheduler.step(mae)

        print(
            f"Epoch {e:02d}: "
            f"MAE={mae:.3f} "
            f"RMSE={rmse:.3f} "
            f"R2={r2:.4f} "
            f"({time.time()-start:.1f}s)"
        )

        if mae < best:
            best = mae
            best_result = (mae, rmse, r2)

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "spatial_size": SIZE,
                    "normalization_divisor": SCALE,
                    "seed": SEED,
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2
                },
                OUT / "best_model.pt"
            )

    print()
    print("FINAL MODEL COMPLETE")
    print(f"Best MAE:  {best_result[0]:.3f}")
    print(f"Best RMSE: {best_result[1]:.3f}")
    print(f"Best R2:   {best_result[2]:.4f}")
    print(f"Saved: {OUT / 'best_model.pt'}")


if __name__ == "__main__":
    main()