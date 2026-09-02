import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# DUPLICATE / NEAR-DUPLICATE CHECK
# ============================================================

PROJECT_ROOT = r"C:\Users\abhi\Downloads\beyond-visible-spectrum-ai-for-agriculture-2025"
CSV_PATH = os.path.join(PROJECT_ROOT, "train_final.csv")
CUBES_DIR = os.path.join(PROJECT_ROOT, "ot", "ot")
OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "ml",
    "evaluation",
    "diagnostic_results",
    "duplicates"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("HYPERSPECTRAL AGRICULTURE — DUPLICATE CHECK")
print("=" * 70)

df = pd.read_csv(CSV_PATH)

features = []
labels = []
ids = []

print("\nLoading cubes...")

for i, row in df.iterrows():

    filename = str(row["id"])

    if not filename.endswith(".npy"):
        filename = filename + ".npy"

    path = os.path.join(CUBES_DIR, filename)

    if not os.path.exists(path):
        continue

    cube = np.load(path)

    # Mean spectrum: 125 values
    spectrum = cube.mean(axis=(0, 1))

    features.append(spectrum)
    labels.append(row["label"])
    ids.append(filename)

    if (i + 1) % 200 == 0:
        print(f"Loaded {i + 1}/{len(df)}")

X = np.asarray(features, dtype=np.float32)
y = np.asarray(labels)

print("\nFeature shape:", X.shape)

# Normalize each spectrum
norms = np.linalg.norm(X, axis=1, keepdims=True)
X_norm = X / np.maximum(norms, 1e-12)

print("\nCalculating nearest-neighbor similarities...")

similarities = X_norm @ X_norm.T

# Ignore self-similarity
np.fill_diagonal(similarities, -1)

nearest_idx = np.argmax(similarities, axis=1)
nearest_similarity = similarities[
    np.arange(len(X)),
    nearest_idx
]

nearest_same_label = (
    y == y[nearest_idx]
)

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(f"Samples checked: {len(X)}")
print(
    f"Nearest-neighbor similarity — mean: "
    f"{nearest_similarity.mean():.6f}"
)
print(
    f"Nearest-neighbor similarity — max: "
    f"{nearest_similarity.max():.6f}"
)
print(
    f"Nearest neighbors with similarity > 0.999: "
    f"{np.sum(nearest_similarity > 0.999)}"
)
print(
    f"Nearest neighbors with similarity > 0.9999: "
    f"{np.sum(nearest_similarity > 0.9999)}"
)

print(
    f"Nearest-neighbor same-label agreement: "
    f"{nearest_same_label.mean() * 100:.3f}%"
)

# Save detailed results
results = pd.DataFrame({
    "id": ids,
    "nearest_id": [ids[i] for i in nearest_idx],
    "similarity": nearest_similarity,
    "label": y,
    "nearest_label": y[nearest_idx],
    "same_label": nearest_same_label
})

results.to_csv(
    os.path.join(OUTPUT_DIR, "duplicate_results.csv"),
    index=False
)

print("\nSaved:")
print(os.path.join(OUTPUT_DIR, "duplicate_results.csv"))

print("\n" + "=" * 70)
print("CHECK COMPLETE")
print("=" * 70)