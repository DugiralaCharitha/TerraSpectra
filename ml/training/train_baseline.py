from pathlib import Path

import joblib
import numpy as np
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

from ml.datasets.spectral_dataset import load_spectral_dataset


RANDOM_SEED = 42
VALIDATION_SIZE = 0.20


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    print("Loading valid spectral features...")
    X, y, skipped = load_spectral_dataset(
        labels_path=project_root / "train_clean.csv",
        cube_directory=project_root / "ot" / "ot",
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print("Training Logistic Regression baseline...")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
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

    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(y_val, predictions, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_val, predictions, average="weighted", zero_division=0)

    results_directory = project_root / "ml" / "experiments" / "baseline_logreg"
    results_directory.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, results_directory / "model.joblib")
    np.save(
        results_directory / "confusion_matrix.npy",
        confusion_matrix(y_val, predictions),
    )

    report = classification_report(y_val, predictions, zero_division=0)
    metrics_text = (
        f"Accuracy: {accuracy:.4f}\n"
        f"Macro F1: {macro_f1:.4f}\n"
        f"Weighted F1: {weighted_f1:.4f}\n\n"
        f"Classification report:\n{report}"
    )
    (results_directory / "metrics.txt").write_text(metrics_text, encoding="utf-8")

    print("\nBaseline results")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Saved results to: {results_directory}")