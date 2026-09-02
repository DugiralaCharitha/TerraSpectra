from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.feature_engineering.statistical_features import (
    extract_statistical_features,
)


RANDOM_SEED = 42
VALIDATION_SIZE = 0.20


def load_statistical_dataset(labels_path: Path, cube_directory: Path):
    """Load valid labelled cubes using statistical features."""

    labels = pd.read_csv(labels_path)

    features = []
    targets = []
    skipped = []

    for row in labels.itertuples(index=False):
        try:
            cube_path = cube_directory / row.id

            cube = np.load(cube_path)

            statistical_features = extract_statistical_features(cube)

            features.append(statistical_features)
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

    skipped_dataframe = pd.DataFrame(skipped)

    return X, y, skipped_dataframe


def main():
    project_root = Path(__file__).resolve().parents[2]

    labels_path = project_root / "train_final.csv"

    cube_directory = project_root / "ot" / "ot"

    print("Loading statistical features...")

    X, y, skipped = load_statistical_dataset(
        labels_path,
        cube_directory,
    )

    print(f"Feature matrix shape: {X.shape}")
    print(f"Valid samples: {len(X)}")
    print(f"Classes: {len(np.unique(y))}")
    print(f"Skipped samples: {len(skipped)}")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    print("Training scaled Logistic Regression...")

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    accuracy = accuracy_score(
        y_val,
        predictions,
    )

    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_val,
        predictions,
        average="weighted",
        zero_division=0,
    )

    output_directory = (
        project_root
        / "ml"
        / "experiments"
        / "statistical_logreg"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        output_directory / "model.joblib",
    )

    np.save(
        output_directory / "confusion_matrix.npy",
        confusion_matrix(
            y_val,
            predictions,
        ),
    )

    report = classification_report(
        y_val,
        predictions,
        zero_division=0,
    )

    metrics_text = (
        f"Accuracy: {accuracy:.4f}\n"
        f"Macro F1: {macro_f1:.4f}\n"
        f"Weighted F1: {weighted_f1:.4f}\n\n"
        f"Classification report:\n"
        f"{report}"
    )

    (
        output_directory / "metrics.txt"
    ).write_text(
        metrics_text,
        encoding="utf-8",
    )

    print("\nStatistical-baseline results")

    print(f"Accuracy: {accuracy:.4f}")

    print(f"Macro F1: {macro_f1:.4f}")

    print(f"Weighted F1: {weighted_f1:.4f}")

    print(f"Saved results to: {output_directory}")


if __name__ == "__main__":
    main()