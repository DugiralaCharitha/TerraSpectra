"""Generate Kaggle predictions from a trained hyperspectral 3D-CNN.

Example:
    py -3.12 ml/inference/generate_submission.py \
        --checkpoint ml/experiments/3dcnn/best_model.pt \
        --output ml/experiments/3dcnn/submission_week2.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.train_3dcnn import NORMALIZATION_DIVISOR, Spectral3DCNN


class TestCubeDataset(Dataset):
    """Lazily prepare test cubes, including the nine nonstandard-width cubes."""

    def __init__(self, test_csv: Path, cubes_dir: Path, spatial_size: int):
        with test_csv.open(newline="", encoding="utf-8") as handle:
            self.ids = [row["id"] for row in csv.DictReader(handle)]
        self.cubes_dir = cubes_dir
        self.spatial_size = spatial_size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        filename = self.ids[index]
        cube = np.load(self.cubes_dir / filename).astype(np.float32, copy=False)
        if cube.ndim != 3 or cube.shape[2] != 125:
            raise ValueError(f"Unexpected cube shape for {filename}: {cube.shape}")

        # Normal 128x128 cubes use exact average pooling. Nonstandard cubes are
        # resized only at inference so every Kaggle test ID receives a prediction.
        if cube.shape[:2] == (128, 128) and 128 % self.spatial_size == 0:
            factor = 128 // self.spatial_size
            cube = cube.reshape(self.spatial_size, factor, self.spatial_size, factor, 125)
            cube = cube.mean(axis=(1, 3), dtype=np.float32).transpose(2, 0, 1)
            image = torch.from_numpy(np.ascontiguousarray(cube)).unsqueeze(0)
        else:
            image = torch.from_numpy(np.ascontiguousarray(cube.transpose(2, 0, 1))).unsqueeze(0)
            image = functional.interpolate(image.unsqueeze(0), size=(125, self.spatial_size, self.spatial_size), mode="trilinear", align_corners=False).squeeze(0)
        return image / NORMALIZATION_DIVISOR, filename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "ml" / "experiments" / "3dcnn" / "best_model.pt")
    parser.add_argument("--test-csv", type=Path, default=PROJECT_ROOT / "test.csv")
    parser.add_argument("--cubes-dir", type=Path, default=PROJECT_ROOT / "ot" / "ot")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "ml" / "experiments" / "3dcnn" / "submission_week2.csv")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    width = checkpoint.get("width", 8)  # Week 2 checkpoint predates configurable width.
    model = Spectral3DCNN(checkpoint["num_classes"], width=width)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = TestCubeDataset(args.test_csv, args.cubes_dir, checkpoint["spatial_size"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    predictions = []
    with torch.no_grad():
        for images, filenames in loader:
            labels = model(images).argmax(dim=1).tolist()
            predictions.extend(zip(filenames, labels))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "TARGET"])
        writer.writerows(predictions)
    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
