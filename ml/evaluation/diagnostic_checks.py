"""
Hyperspectral Agriculture Project
Diagnostic Checks

Performs:
1. Label distribution analysis
2. Mean-spectrum nearest-neighbor label agreement
3. Regression mean baseline
4. Classification majority-class baseline

No model training is performed.
Existing experiment files are not modified.

Expected dataset:
    train_final.csv
    ot/ot/sampleXXXX.npy

Project root:
    C:/Users/abhi/Downloads/beyond-visible-spectrum-ai-for-agriculture-2025
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\abhi\Downloads\beyond-visible-spectrum-ai-for-agriculture-2025"
)

CSV_PATH = PROJECT_ROOT / "train_final.csv"
CUBES_DIR = PROJECT_ROOT / "ot" / "ot"

OUTPUT_DIR = PROJECT_ROOT / "ml" / "evaluation" / "diagnostic_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42
VALIDATION_FRACTION = 0.20

# Mean-spectrum NN test uses 125-dimensional vectors.
# We use Euclidean distance after per-sample normalization
# through StandardScaler fitted on the complete feature matrix.
#
# This is a diagnostic only, not a final ML evaluation.
USE_STANDARDIZATION = True


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def find_label_column(df):
    """
    Try to identify the label column.
    """

    candidates = [
        "label",
        "class",
        "target",
        "y",
        "Label",
        "Class",
        "Target",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    # Fall back to a column containing mostly numeric values
    numeric_candidates = []

    for col in df.columns:
        numeric = pd.to_numeric(df[col], errors="coerce")
        valid_fraction = numeric.notna().mean()

        if valid_fraction > 0.95:
            numeric_candidates.append(col)

    if len(numeric_candidates) == 1:
        return numeric_candidates[0]

    raise ValueError(
        "Could not automatically identify the label column. "
        f"Available columns: {list(df.columns)}"
    )


def find_id_column(df):
    """
    Try to identify the sample ID column.
    """

    candidates = [
        "id",
        "ID",
        "sample_id",
        "sample",
        "filename",
        "file",
        "image_id",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    # Look for a column containing sample-like strings
    for col in df.columns:
        values = df[col].astype(str)

        if values.str.contains("sample", case=False, regex=False).mean() > 0.5:
            return col

    # If no obvious ID column exists, use the first column
    return df.columns[0]


def resolve_cube_path(sample_id):
    """
    Convert an ID such as:
        sample2451
        sample2451.npy
        2451

    into:
        ot/ot/sample2451.npy
    """

    sample_id = str(sample_id).strip()

    if sample_id.endswith(".npy"):
        filename = sample_id
    elif sample_id.lower().startswith("sample"):
        filename = sample_id + ".npy"
    else:
        filename = f"sample{sample_id}.npy"

    return CUBES_DIR / filename


def load_mean_spectra(df, id_column):
    """
    Load every cube and calculate one mean value per spectral band.

    Cube expected shape:
        (128, 128, 125)

    Result:
        X shape = (samples, 125)
    """

    spectra = []
    valid_rows = []
    skipped = []

    total = len(df)

    print("\nLoading hyperspectral cubes...")
    print(f"Expected samples: {total}")
    print(f"Cubes directory: {CUBES_DIR}")

    start = time.time()

    for position, (_, row) in enumerate(df.iterrows(), start=1):

        sample_id = row[id_column]
        cube_path = resolve_cube_path(sample_id)

        if not cube_path.exists():
            skipped.append(
                {
                    "row": position - 1,
                    "sample_id": str(sample_id),
                    "reason": "cube_not_found",
                    "path": str(cube_path),
                }
            )
            continue

        try:
            cube = np.load(cube_path)

            if cube.ndim != 3:
                skipped.append(
                    {
                        "row": position - 1,
                        "sample_id": str(sample_id),
                        "reason": f"unexpected_ndim_{cube.ndim}",
                        "shape": str(cube.shape),
                    }
                )
                continue

            # Expected:
            # height x width x spectral_bands
            #
            # Mean over spatial dimensions.
            mean_spectrum = cube.astype(np.float32).mean(axis=(0, 1))

            if not np.all(np.isfinite(mean_spectrum)):
                skipped.append(
                    {
                        "row": position - 1,
                        "sample_id": str(sample_id),
                        "reason": "non_finite_spectrum",
                    }
                )
                continue

            spectra.append(mean_spectrum)
            valid_rows.append(position - 1)

        except Exception as exc:
            skipped.append(
                {
                    "row": position - 1,
                    "sample_id": str(sample_id),
                    "reason": "load_error",
                    "error": str(exc),
                }
            )

        if position % 100 == 0 or position == total:
            elapsed = time.time() - start
            print(
                f"Loaded {position}/{total} "
                f"({position / total * 100:.1f}%) "
                f"- {elapsed:.1f}s"
            )

    if not spectra:
        raise RuntimeError("No hyperspectral cubes could be loaded.")

    X = np.stack(spectra, axis=0)

    valid_df = df.iloc[valid_rows].reset_index(drop=True)

    return X, valid_df, skipped


# ============================================================
# 1. LABEL DISTRIBUTION
# ============================================================

def label_distribution(df, label_column):

    labels = pd.to_numeric(df[label_column], errors="coerce")

    if labels.isna().any():
        raise ValueError(
            f"Label column '{label_column}' contains non-numeric values."
        )

    labels = labels.astype(int)

    unique_labels = sorted(labels.unique())

    counts = labels.value_counts().sort_index()

    expected_labels = list(range(101))

    missing_labels = [
        label for label in expected_labels
        if label not in unique_labels
    ]

    extra_labels = [
        label for label in unique_labels
        if label not in expected_labels
    ]

    distribution = {
        str(int(label)): int(count)
        for label, count in counts.items()
    }

    result = {
        "label_column": label_column,
        "sample_count": int(len(labels)),
        "unique_label_count": int(len(unique_labels)),
        "minimum_label": int(labels.min()),
        "maximum_label": int(labels.max()),
        "unique_labels": [int(x) for x in unique_labels],
        "missing_labels_0_to_100": missing_labels,
        "unexpected_labels_outside_0_to_100": extra_labels,
        "min_class_count": int(counts.min()),
        "max_class_count": int(counts.max()),
        "mean_class_count": float(counts.mean()),
        "median_class_count": float(counts.median()),
        "class_counts": distribution,
    }

    counts_df = counts.rename("count").reset_index()
    counts_df.columns = ["label", "count"]

    counts_df.to_csv(
        OUTPUT_DIR / "label_distribution.csv",
        index=False,
    )

    return result


# ============================================================
# 2. NEAREST-NEIGHBOR SPECTRAL AGREEMENT
# ============================================================

def nearest_neighbor_test(X, labels):

    print("\nRunning nearest-neighbor spectral test...")

    start = time.time()

    X_work = X.copy()

    if USE_STANDARDIZATION:
        scaler = StandardScaler()
        X_work = scaler.fit_transform(X_work)

    # Normalize each sample to unit length.
    # This makes the comparison more about spectral shape
    # than absolute brightness.
    norms = np.linalg.norm(X_work, axis=1, keepdims=True)
    norms[norms == 0] = 1.0

    X_work = X_work / norms

    n = len(X_work)

    same_label = np.zeros(n, dtype=bool)
    nearest_indices = np.zeros(n, dtype=np.int64)
    distances = np.zeros(n, dtype=np.float32)

    # Process in chunks to avoid creating a massive
    # n x n distance matrix all at once.
    chunk_size = 100

    for start_idx in range(0, n, chunk_size):

        end_idx = min(start_idx + chunk_size, n)

        chunk = X_work[start_idx:end_idx]

        # Squared Euclidean distance.
        #
        # Since vectors are unit normalized:
        # ||a-b||² = 2 - 2*cosine_similarity
        similarities = chunk @ X_work.T

        # Exclude self-match.
        for local_idx in range(end_idx - start_idx):
            global_idx = start_idx + local_idx
            similarities[local_idx, global_idx] = -np.inf

        nn_idx = np.argmax(similarities, axis=1)

        nearest_indices[start_idx:end_idx] = nn_idx

        nn_similarity = similarities[
            np.arange(end_idx - start_idx),
            nn_idx,
        ]

        distances[start_idx:end_idx] = np.sqrt(
            np.maximum(0.0, 2.0 - 2.0 * nn_similarity)
        )

        same_label[start_idx:end_idx] = (
            labels[nn_idx] == labels[start_idx:end_idx]
        )

        processed = end_idx

        if processed % 500 == 0 or processed == n:
            print(
                f"Processed {processed}/{n} samples "
                f"({processed / n * 100:.1f}%)"
            )

    agreement = float(same_label.mean())

    # Random-label theoretical baseline:
    # for a balanced 101-class problem ≈ 1/101.
    # We also calculate the empirical probability from class frequencies.
    class_counts = pd.Series(labels).value_counts()
    probabilities = class_counts / len(labels)

    expected_random_agreement = float(
        np.sum(probabilities.values ** 2)
    )

    nn_table = pd.DataFrame(
        {
            "sample_index": np.arange(n),
            "label": labels,
            "nearest_neighbor_index": nearest_indices,
            "nearest_neighbor_label": labels[nearest_indices],
            "nearest_neighbor_distance": distances,
            "same_label": same_label,
        }
    )

    nn_table.to_csv(
        OUTPUT_DIR / "nearest_neighbor_results.csv",
        index=False,
    )

    result = {
        "samples": int(n),
        "nearest_neighbor_same_label_count": int(same_label.sum()),
        "nearest_neighbor_different_label_count": int((~same_label).sum()),
        "same_label_agreement": agreement,
        "same_label_agreement_percent": agreement * 100,
        "theoretical_101_class_chance_percent": 100 / 101,
        "empirical_random_label_agreement_percent":
            expected_random_agreement * 100,
        "mean_nearest_neighbor_distance": float(distances.mean()),
        "median_nearest_neighbor_distance": float(np.median(distances)),
        "interpretation": (
            "Potential spectral-label signal"
            if agreement > expected_random_agreement * 2
            else
            "Near-random spectral-label agreement"
        ),
    }

    print(
        f"\nNearest-neighbor same-label agreement: "
        f"{agreement * 100:.3f}%"
    )

    print(
        f"Empirical random-label expectation: "
        f"{expected_random_agreement * 100:.3f}%"
    )

    print(
        f"Elapsed: {time.time() - start:.1f}s"
    )

    return result


# ============================================================
# 3. REGRESSION MEAN BASELINE
# ============================================================

def regression_mean_baseline(df, label_column):

    labels = pd.to_numeric(
        df[label_column],
        errors="coerce",
    ).astype(float).to_numpy()

    train_idx, val_idx = train_test_split(
        np.arange(len(labels)),
        test_size=VALIDATION_FRACTION,
        random_state=SEED,
    )

    train_labels = labels[train_idx]
    val_labels = labels[val_idx]

    train_mean = float(train_labels.mean())

    predictions = np.full(
        len(val_labels),
        train_mean,
        dtype=np.float64,
    )

    mae = mean_absolute_error(
        val_labels,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            val_labels,
            predictions,
        )
    )

    r2 = r2_score(
        val_labels,
        predictions,
    )

    result = {
        "method": "constant_train_mean",
        "seed": SEED,
        "validation_fraction": VALIDATION_FRACTION,
        "train_samples": int(len(train_labels)),
        "validation_samples": int(len(val_labels)),
        "train_mean_prediction": train_mean,
        "validation_actual_mean": float(val_labels.mean()),
        "validation_actual_std": float(val_labels.std()),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }

    return result


# ============================================================
# 4. CLASSIFICATION MAJORITY BASELINE
# ============================================================

def classification_majority_baseline(df, label_column):

    labels = pd.to_numeric(
        df[label_column],
        errors="coerce",
    ).astype(int).to_numpy()

    train_idx, val_idx = train_test_split(
        np.arange(len(labels)),
        test_size=VALIDATION_FRACTION,
        random_state=SEED,
        stratify=labels,
    )

    train_labels = labels[train_idx]
    val_labels = labels[val_idx]

    values, counts = np.unique(
        train_labels,
        return_counts=True,
    )

    majority_label = int(
        values[np.argmax(counts)]
    )

    predictions = np.full(
        len(val_labels),
        majority_label,
        dtype=int,
    )

    accuracy = accuracy_score(
        val_labels,
        predictions,
    )

    result = {
        "method": "majority_class",
        "seed": SEED,
        "validation_fraction": VALIDATION_FRACTION,
        "train_samples": int(len(train_labels)),
        "validation_samples": int(len(val_labels)),
        "majority_class": majority_label,
        "majority_class_train_count": int(counts.max()),
        "validation_accuracy": float(accuracy),
        "validation_accuracy_percent": float(accuracy * 100),
        "101_class_uniform_chance_percent": 100 / 101,
    }

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = time.time()

    print("=" * 70)
    print("HYPERSPECTRAL AGRICULTURE — DIAGNOSTIC CHECKS")
    print("=" * 70)

    print(f"\nProject root:")
    print(PROJECT_ROOT)

    print(f"\nCSV:")
    print(CSV_PATH)

    print(f"\nCubes:")
    print(CUBES_DIR)

    print(f"\nOutput:")
    print(OUTPUT_DIR)

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"train_final.csv not found:\n{CSV_PATH}"
        )

    if not CUBES_DIR.exists():
        raise FileNotFoundError(
            f"Cube directory not found:\n{CUBES_DIR}"
        )

    # --------------------------------------------------------
    # Load CSV
    # --------------------------------------------------------

    print("\nLoading train_final.csv...")

    df = pd.read_csv(CSV_PATH)

    print(f"CSV rows: {len(df)}")
    print(f"CSV columns: {list(df.columns)}")

    label_column = find_label_column(df)
    id_column = find_id_column(df)

    print(f"\nDetected label column: {label_column}")
    print(f"Detected ID column: {id_column}")

    # --------------------------------------------------------
    # Label distribution
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("1. LABEL DISTRIBUTION")
    print("=" * 70)

    label_result = label_distribution(
        df,
        label_column,
    )

    print(
        f"Samples: {label_result['sample_count']}"
    )

    print(
        f"Unique labels: {label_result['unique_label_count']}"
    )

    print(
        f"Label range: "
        f"{label_result['minimum_label']} → "
        f"{label_result['maximum_label']}"
    )

    print(
        f"Class count range: "
        f"{label_result['min_class_count']} → "
        f"{label_result['max_class_count']}"
    )

    print(
        f"Missing labels from 0–100: "
        f"{label_result['missing_labels_0_to_100']}"
    )

    # --------------------------------------------------------
    # Load mean spectra
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("2. BUILD MEAN SPECTRAL FEATURES")
    print("=" * 70)

    X, valid_df, skipped = load_mean_spectra(
        df,
        id_column,
    )

    labels = pd.to_numeric(
        valid_df[label_column],
        errors="coerce",
    ).astype(int).to_numpy()

    print(
        f"\nMean-spectrum matrix shape: {X.shape}"
    )

    print(
        f"Successfully loaded: {len(valid_df)}"
    )

    print(
        f"Skipped: {len(skipped)}"
    )

    if skipped:
        with open(
            OUTPUT_DIR / "skipped_samples.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                skipped,
                f,
                indent=2,
            )

    # --------------------------------------------------------
    # Nearest neighbor
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("3. NEAREST-NEIGHBOR SPECTRAL AGREEMENT")
    print("=" * 70)

    nn_result = nearest_neighbor_test(
        X,
        labels,
    )

    # --------------------------------------------------------
    # Regression baseline
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("4. REGRESSION MEAN BASELINE")
    print("=" * 70)

    regression_result = regression_mean_baseline(
        valid_df,
        label_column,
    )

    print(
        f"Mean prediction: "
        f"{regression_result['train_mean_prediction']:.4f}"
    )

    print(
        f"MAE: "
        f"{regression_result['mae']:.4f}"
    )

    print(
        f"RMSE: "
        f"{regression_result['rmse']:.4f}"
    )

    print(
        f"R²: "
        f"{regression_result['r2']:.6f}"
    )

    # --------------------------------------------------------
    # Classification baseline
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("5. CLASSIFICATION MAJORITY BASELINE")
    print("=" * 70)

    classification_result = classification_majority_baseline(
        valid_df,
        label_column,
    )

    print(
        f"Majority class: "
        f"{classification_result['majority_class']}"
    )

    print(
        f"Accuracy: "
        f"{classification_result['validation_accuracy_percent']:.3f}%"
    )

    print(
        f"101-class chance: "
        f"{classification_result['101_class_uniform_chance_percent']:.3f}%"
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    summary = {
        "project": "Beyond Visible Spectrum AI for Agriculture 2025",
        "seed": SEED,
        "csv": str(CSV_PATH),
        "cubes_dir": str(CUBES_DIR),
        "output_dir": str(OUTPUT_DIR),
        "original_csv_samples": int(len(df)),
        "usable_cube_samples": int(len(valid_df)),
        "skipped_samples": int(len(skipped)),
        "label_distribution": label_result,
        "nearest_neighbor_test": nn_result,
        "regression_mean_baseline": regression_result,
        "classification_majority_baseline": classification_result,
        "conclusion": {
            "nn_signal_above_empirical_random": bool(
                nn_result["same_label_agreement"]
                > nn_result["empirical_random_label_agreement_percent"] / 100 * 2
            ),
            "regression_models_should_be_compared_against_mae":
                regression_result["mae"],
            "classification_models_should_be_compared_against_chance_percent":
                100 / 101,
        },
    }

    with open(
        OUTPUT_DIR / "diagnostic_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # Human-readable report
    report_path = OUTPUT_DIR / "diagnostic_report.txt"

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "HYPERSPECTRAL AGRICULTURE PROJECT\n"
            "DIAGNOSTIC REPORT\n"
            "=" * 70 + "\n\n"
        )

        f.write(
            "LABEL DISTRIBUTION\n"
            "-" * 70 + "\n"
        )

        f.write(
            f"Samples: {label_result['sample_count']}\n"
        )

        f.write(
            f"Unique labels: "
            f"{label_result['unique_label_count']}\n"
        )

        f.write(
            f"Label range: "
            f"{label_result['minimum_label']} - "
            f"{label_result['maximum_label']}\n"
        )

        f.write(
            f"Class count range: "
            f"{label_result['min_class_count']} - "
            f"{label_result['max_class_count']}\n"
        )

        f.write(
            f"Missing labels 0-100: "
            f"{label_result['missing_labels_0_to_100']}\n\n"
        )

        f.write(
            "NEAREST-NEIGHBOR TEST\n"
            "-" * 70 + "\n"
        )

        f.write(
            f"Same-label agreement: "
            f"{nn_result['same_label_agreement_percent']:.4f}%\n"
        )

        f.write(
            f"Empirical random agreement: "
            f"{nn_result['empirical_random_label_agreement_percent']:.4f}%\n"
        )

        f.write(
            f"101-class uniform chance: "
            f"{nn_result['theoretical_101_class_chance_percent']:.4f}%\n"
        )

        f.write(
            f"Interpretation: "
            f"{nn_result['interpretation']}\n\n"
        )

        f.write(
            "REGRESSION MEAN BASELINE\n"
            "-" * 70 + "\n"
        )

        f.write(
            f"Mean prediction: "
            f"{regression_result['train_mean_prediction']:.6f}\n"
        )

        f.write(
            f"MAE: "
            f"{regression_result['mae']:.6f}\n"
        )

        f.write(
            f"RMSE: "
            f"{regression_result['rmse']:.6f}\n"
        )

        f.write(
            f"R2: "
            f"{regression_result['r2']:.6f}\n\n"
        )

        f.write(
            "CLASSIFICATION MAJORITY BASELINE\n"
            "-" * 70 + "\n"
        )

        f.write(
            f"Majority class: "
            f"{classification_result['majority_class']}\n"
        )

        f.write(
            f"Accuracy: "
            f"{classification_result['validation_accuracy_percent']:.6f}%\n"
        )

        f.write(
            f"101-class chance: "
            f"{classification_result['101_class_uniform_chance_percent']:.6f}%\n\n"
        )

        f.write(
            "FILES CREATED\n"
            "-" * 70 + "\n"
            "diagnostic_summary.json\n"
            "diagnostic_report.txt\n"
            "label_distribution.csv\n"
            "nearest_neighbor_results.csv\n"
        )

        if skipped:
            f.write(
                "skipped_samples.json\n"
            )

    print("\n" + "=" * 70)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 70)

    print(
        f"\nResults saved to:\n{OUTPUT_DIR}"
    )

    print(
        "\nImportant files:"
    )

    print(
        "  diagnostic_summary.json"
    )

    print(
        "  diagnostic_report.txt"
    )

    print(
        "  label_distribution.csv"
    )

    print(
        "  nearest_neighbor_results.csv"
    )

    print(
        f"\nTotal runtime: "
        f"{time.time() - total_start:.1f} seconds"
    )


if __name__ == "__main__":
    main()