from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_SHAPE = (128, 128, 125)


def extract_mean_spectrum(cube: np.ndarray) -> np.ndarray:
    """Return one 125-band mean spectrum using foreground pixels only."""
    if cube.shape != EXPECTED_SHAPE:
        raise ValueError(f"Unexpected shape: {cube.shape}")

    foreground_mask = np.any(cube > 0, axis=2)

    if not foreground_mask.any():
        raise ValueError("Blank cube")

    return cube[foreground_mask].mean(axis=0).astype(np.float32)


def load_spectral_dataset(labels_path: Path, cube_directory: Path):
    """Load valid labelled cubes as features (X) and labels (y)."""
    labels = pd.read_csv(labels_path)

    features = []
    targets = []
    skipped = []

    for row in labels.itertuples(index=False):
        cube_path = cube_directory / row.id

        try:
            cube = np.load(cube_path)
            spectrum = extract_mean_spectrum(cube)

            features.append(spectrum)
            targets.append(row.label)

        except Exception as error:
            skipped.append(
                {
                    "id": row.id,
                    "label": row.label,
                    "reason": str(error),
                }
            )

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets)

    return X, y, pd.DataFrame(skipped)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    X, y, skipped = load_spectral_dataset(
        labels_path=project_root / "train_clean.csv",
        cube_directory=project_root / "ot" / "ot",
    )

    print("Feature matrix X shape:", X.shape)
    print("Label vector y shape:", y.shape)
    print("Number of classes:", len(np.unique(y)))
    print("Skipped samples:", len(skipped))
    print("\nSkipped-reason counts:")
    print(skipped["reason"].value_counts())