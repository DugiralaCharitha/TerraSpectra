from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ml.training.train_3dcnn import read_samples, stratified_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SHAPE = (128, 128, 125)


class EvaluationDataset(Dataset):
    """Matches the original 3D-CNN validation preprocessing."""

    def __init__(self, samples, cubes_dir, spatial_size, normalization_divisor):
        self.samples = list(samples)
        self.cubes_dir = Path(cubes_dir)
        self.spatial_size = spatial_size
        self.normalization_divisor = normalization_divisor
        self.factor = 128 // spatial_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        cube = np.load(self.cubes_dir / sample.filename).astype(np.float32, copy=False)

        if cube.shape != EXPECTED_SHAPE:
            raise ValueError(f"Unexpected cube shape for {sample.filename}: {cube.shape}")

        cube = cube.reshape(
            self.spatial_size, self.factor,
            self.spatial_size, self.factor,
            125,
        ).mean(axis=(1, 3), dtype=np.float32)

        cube = cube / self.normalization_divisor
        cube = np.ascontiguousarray(cube.transpose(2, 0, 1))

        return (
            torch.from_numpy(cube).unsqueeze(0),
            torch.tensor(sample.label, dtype=torch.long),
            sample.filename,
        )


class Final3DCNN(nn.Module):
    """Exact architecture of ml/experiments/3dcnn/best_model.pt."""

    def __init__(self, num_classes, width):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, width, 3, stride=(2, 1, 1), padding=1),
            nn.BatchNorm3d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(width, width * 2, 3, stride=(2, 1, 1), padding=1),
            nn.BatchNorm3d(width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(width * 2, width * 4, 3, stride=(2, 1, 1), padding=1),
            nn.BatchNorm3d(width * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(width * 4, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "ml" / "experiments" / "3dcnn" / "best_model.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "ml" / "experiments" / "3dcnn" / "final_evaluation",
    )
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    num_classes = checkpoint["num_classes"]
    spatial_size = checkpoint["spatial_size"]
    normalization_divisor = checkpoint["normalization_divisor"]
    width = checkpoint["model_state_dict"]["features.0.weight"].shape[0]

    samples = read_samples(args.csv, args.cubes_dir)
    _, validation_samples = stratified_split(
        samples,
        validation_fraction=0.20,
        seed=args.seed,
    )

    dataset = EvaluationDataset(
        validation_samples,
        args.cubes_dir,
        spatial_size,
        normalization_divisor,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = Final3DCNN(num_classes, width).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    true_labels, predicted_labels, filenames = [], [], []

    with torch.no_grad():
        for images, labels, batch_filenames in loader:
            logits = model(images.to(device))
            predictions = logits.argmax(dim=1).cpu().numpy()

            true_labels.extend(labels.numpy().tolist())
            predicted_labels.extend(predictions.tolist())
            filenames.extend(batch_filenames)

    accuracy = accuracy_score(true_labels, predicted_labels)
    macro_f1 = f1_score(true_labels, predicted_labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(true_labels, predicted_labels, average="weighted", zero_division=0)
    matrix = confusion_matrix(true_labels, predicted_labels, labels=list(range(num_classes)))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = args.output_dir / "metrics.json"
    predictions_path = args.output_dir / "validation_predictions.csv"
    matrix_path = args.output_dir / "confusion_matrix.npy"
    figure_path = args.output_dir / "confusion_matrix.png"

    metrics = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "validation_samples": len(true_labels),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "spatial_size": spatial_size,
        "normalization_divisor": normalization_divisor,
        "model_width": width,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    with predictions_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "true_label", "predicted_label", "correct"])
        for filename, truth, prediction in zip(filenames, true_labels, predicted_labels):
            writer.writerow([filename, truth, prediction, truth == prediction])

    np.save(matrix_path, matrix)

    plt.figure(figsize=(20, 16))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.colorbar()
    plt.title("3D-CNN Validation Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=180)
    plt.close()

    print(f"Validation samples: {len(true_labels)}")
    print(f"Accuracy:           {accuracy:.4%}")
    print(f"Macro F1:           {macro_f1:.6f}")
    print(f"Weighted F1:        {weighted_f1:.6f}")
    print("\nSaved:")
    print(metrics_path)
    print(predictions_path)
    print(matrix_path)
    print(figure_path)


if __name__ == "__main__":
    main()