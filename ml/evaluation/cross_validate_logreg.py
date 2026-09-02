from pathlib import Path
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.datasets.spectral_dataset import load_spectral_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "train_final.csv"
CUBES_DIR = PROJECT_ROOT / "ot" / "ot"
OUTPUT_DIR = PROJECT_ROOT / "ml" / "experiments" / "statistical_logreg" / "five_fold_cv"


def main():
    X, y, skipped = load_spectral_dataset(CSV_PATH, CUBES_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_predictions = np.empty_like(y)
    fold_results = []

    for fold, (train_index, val_index) in enumerate(cv.split(X, y), start=1):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                solver="lbfgs",
               
            ),
        )

        model.fit(X[train_index], y[train_index])
        predictions = model.predict(X[val_index])
        all_predictions[val_index] = predictions

        result = {
            "fold": fold,
            "accuracy": accuracy_score(y[val_index], predictions),
            "macro_f1": f1_score(y[val_index], predictions, average="macro", zero_division=0),
            "weighted_f1": f1_score(y[val_index], predictions, average="weighted", zero_division=0),
        }
        fold_results.append(result)
        print(
            f"Fold {fold}: accuracy={result['accuracy']:.4%}, "
            f"macro F1={result['macro_f1']:.6f}, "
            f"weighted F1={result['weighted_f1']:.6f}"
        )

    summary = {
        "valid_samples": int(len(y)),
        "skipped_samples": int(len(skipped)),
        "folds": fold_results,
        "overall_accuracy": accuracy_score(y, all_predictions),
        "overall_macro_f1": f1_score(y, all_predictions, average="macro", zero_division=0),
        "overall_weighted_f1": f1_score(y, all_predictions, average="weighted", zero_division=0),
        "accuracy_mean": float(np.mean([x["accuracy"] for x in fold_results])),
        "accuracy_std": float(np.std([x["accuracy"] for x in fold_results])),
    }

    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    np.save(
        OUTPUT_DIR / "confusion_matrix.npy",
        confusion_matrix(y, all_predictions, labels=np.arange(101)),
    )

    print("\nFinal 5-fold result")
    print(f"Accuracy:    {summary['overall_accuracy']:.4%}")
    print(f"Macro F1:    {summary['overall_macro_f1']:.6f}")
    print(f"Weighted F1: {summary['overall_weighted_f1']:.6f}")
    print(f"Accuracy SD: {summary['accuracy_std']:.4%}")
    print(f"\nSaved: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()