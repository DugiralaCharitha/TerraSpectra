from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"C:\Users\abhi\Downloads\beyond-visible-spectrum-ai-for-agriculture-2025")

CSV_PATH = ROOT / "train_final.csv"
CUBE_DIR = ROOT / "ot" / "ot"
OUTPUT_DIR = ROOT / "ml" / "evaluation" / "diagnostic_results" / "permutation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv(CSV_PATH)

print("=" * 70)
print("LABEL PERMUTATION TEST")
print("=" * 70)

print(f"CSV rows: {len(df)}")
print(f"Cubes directory: {CUBE_DIR}")

# ============================================================
# LOAD CUBES USING CSV IDs
# ============================================================

features = []
labels = []
ids = []

missing = []

for _, row in df.iterrows():

    sample_id = str(row["id"]).replace("sample", "").replace(".npy", "")
    label = int(row["label"])

    cube_path = CUBE_DIR / f"sample{sample_id}.npy"

    if not cube_path.exists():
        missing.append(sample_id)
        continue

    cube = np.load(cube_path)

    # Mean spectrum: average spatial dimensions
    spectrum = cube.mean(axis=(0, 1))

    features.append(spectrum)
    labels.append(label)
    ids.append(sample_id)

X = np.asarray(features, dtype=np.float32)
y = np.asarray(labels)

print()
print(f"Successfully loaded: {len(X)}")
print(f"Missing cubes: {len(missing)}")
print(f"Feature shape: {X.shape}")

if len(X) == 0:
    raise RuntimeError(
        "ZERO CUBES LOADED. Check CSV path, cube directory, "
        "and sample filename format."
    )

# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

train_idx, val_idx = train_test_split(
    np.arange(len(X)),
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train = X[train_idx]
X_val = X[val_idx]

y_train = y[train_idx]
y_val = y[val_idx]

# ============================================================
# NORMALIZATION
# ============================================================

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

std[std < 1e-8] = 1.0

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std

# ============================================================
# FUNCTION
# ============================================================

def run_test(train_labels, val_labels):

    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs"
    )

    model.fit(X_train, train_labels)

    predictions = model.predict(X_val)

    return accuracy_score(val_labels, predictions)

# ============================================================
# REAL LABEL TEST
# ============================================================

print()
print("=" * 70)
print("REAL LABEL TEST")
print("=" * 70)

real_accuracy = run_test(y_train, y_val)

print(f"Real-label accuracy: {real_accuracy * 100:.3f}%")

# ============================================================
# PERMUTATION TEST
# ============================================================

print()
print("=" * 70)
print("PERMUTATION TEST")
print("=" * 70)

rng = np.random.default_rng(42)

permutation_results = []

N_PERMUTATIONS = 10

for i in range(N_PERMUTATIONS):

    shuffled = rng.permutation(y_train)

    accuracy = run_test(shuffled, y_val)

    permutation_results.append(accuracy)

    print(
        f"Permutation {i + 1:02d}/{N_PERMUTATIONS}: "
        f"{accuracy * 100:.3f}%"
    )

mean_perm = np.mean(permutation_results)
std_perm = np.std(permutation_results)

# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "samples_loaded": int(len(X)),
    "missing_cubes": int(len(missing)),
    "real_accuracy": float(real_accuracy),
    "permutation_mean_accuracy": float(mean_perm),
    "permutation_std_accuracy": float(std_perm),
    "num_permutations": N_PERMUTATIONS,
    "chance_accuracy": float(1 / len(np.unique(y))),
    "accuracy_difference": float(real_accuracy - mean_perm)
}

pd.DataFrame([results]).to_csv(
    OUTPUT_DIR / "permutation_summary.csv",
    index=False
)

with open(OUTPUT_DIR / "permutation_report.txt", "w") as f:

    f.write("HYPERSPECTRAL AGRICULTURE — LABEL PERMUTATION TEST\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Samples loaded: {len(X)}\n")
    f.write(f"Missing cubes: {len(missing)}\n\n")

    f.write(f"Real-label accuracy: {real_accuracy * 100:.3f}%\n")
    f.write(
        f"Permutation mean accuracy: "
        f"{mean_perm * 100:.3f}%\n"
    )
    f.write(
        f"Permutation std: "
        f"{std_perm * 100:.3f}%\n"
    )
    f.write(
        f"Chance accuracy: "
        f"{(1 / len(np.unique(y))) * 100:.3f}%\n"
    )
    f.write(
        f"Real - permutation difference: "
        f"{(real_accuracy - mean_perm) * 100:.3f} percentage points\n"
    )

print()
print("=" * 70)
print("RESULT")
print("=" * 70)

print(f"Real-label accuracy:       {real_accuracy * 100:.3f}%")
print(f"Permutation mean:          {mean_perm * 100:.3f}%")
print(f"Permutation std:           {std_perm * 100:.3f}%")
print(f"Chance:                    {(1 / len(np.unique(y))) * 100:.3f}%")

print()
print("Saved:")
print(OUTPUT_DIR / "permutation_summary.csv")
print(OUTPUT_DIR / "permutation_report.txt")