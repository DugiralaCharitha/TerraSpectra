import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib

ROOT = Path(".")
CSV = ROOT / "train_final.csv"
CUBES = ROOT / "ot" / "ot"
OUT = ROOT / "ml" / "experiments" / "extratrees_day1"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)

def features(cube):
    cube = cube.astype(np.float32)
    # Mean spectrum
    mean = cube.mean(axis=(0, 1))
    # Standard deviation spectrum
    std = cube.std(axis=(0, 1))
    # Percentiles capture spectral distribution
    p25 = np.percentile(cube, 25, axis=(0, 1))
    p75 = np.percentile(cube, 75, axis=(0, 1))
    return np.concatenate([mean, std, p25, p75])

X, y = [], []

for i, row in enumerate(df.itertuples(index=False), 1):
    cube = np.load(CUBES / row.id)
    X.append(features(cube))
    y.append(row.label)
    if i % 250 == 0:
        print(f"Processed {i}/{len(df)}")

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y)

print("Feature matrix:", X.shape)
print("Classes:", len(np.unique(y)))

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

model = ExtraTreesClassifier(
    n_estimators=500,
    max_features="sqrt",
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("Training ExtraTrees...")
model.fit(X_train, y_train)

pred = model.predict(X_val)

accuracy = accuracy_score(y_val, pred)
macro_f1 = f1_score(y_val, pred, average="macro", zero_division=0)
weighted_f1 = f1_score(y_val, pred, average="weighted", zero_division=0)

print()
print("========== DAY 1 RESULT ==========")
print(f"Accuracy:   {accuracy:.4%}")
print(f"Macro F1:   {macro_f1:.6f}")
print(f"Weighted F1:{weighted_f1:.6f}")
print("==================================")

joblib.dump(model, OUT / "model.joblib")
np.save(OUT / "X_features.npy", X)
np.save(OUT / "y_labels.npy", y)

pd.DataFrame({
    "metric": ["accuracy", "macro_f1", "weighted_f1"],
    "value": [accuracy, macro_f1, weighted_f1]
}).to_csv(OUT / "metrics.csv", index=False)

print()
print("Saved to:", OUT)
