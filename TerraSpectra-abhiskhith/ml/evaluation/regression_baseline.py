import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CSV = os.path.join(ROOT, "train_final.csv")
CUBES = os.path.join(ROOT, "ot", "ot")

df = pd.read_csv(CSV)

def get_cube(x):
    x = str(x)
    if x.endswith(".npy"):
        return x
    return f"sample{int(x)}.npy"

X = []
y = []
missing = 0

for _, row in df.iterrows():
    path = os.path.join(CUBES, get_cube(row["id"]))

    if not os.path.exists(path):
        missing += 1
        continue

    cube = np.load(path).astype(np.float32)

    # Mean spectrum: 125 spectral features
    feature = cube.mean(axis=(0, 1))

    X.append(feature)
    y.append(float(row["label"]))

X = np.asarray(X)
y = np.asarray(y)

print("=" * 70)
print("REGRESSION BASELINE")
print("=" * 70)
print(f"Samples: {len(y)}")
print(f"Features: {X.shape}")
print(f"Missing cubes: {missing}")
print(f"Target min: {y.min():.2f}")
print(f"Target max: {y.max():.2f}")
print(f"Target mean: {y.mean():.2f}")
print()

train_idx, val_idx = train_test_split(
    np.arange(len(y)),
    test_size=0.20,
    random_state=42
)

y_train = y[train_idx]
y_val = y[val_idx]

# Mean baseline
prediction = np.full_like(y_val, y_train.mean(), dtype=float)

mae = mean_absolute_error(y_val, prediction)
rmse = np.sqrt(mean_squared_error(y_val, prediction))
r2 = r2_score(y_val, prediction)

print("MEAN BASELINE")
print("-" * 70)
print(f"Prediction mean: {y_train.mean():.4f}")
print(f"MAE:             {mae:.4f}")
print(f"RMSE:            {rmse:.4f}")
print(f"R2:              {r2:.6f}")

out = os.path.join(ROOT, "ml", "evaluation", "diagnostic_results")
os.makedirs(out, exist_ok=True)

with open(os.path.join(out, "regression_baseline.txt"), "w") as f:
    f.write("REGRESSION BASELINE\n")
    f.write(f"Samples: {len(y)}\n")
    f.write(f"Features: {X.shape}\n")
    f.write(f"Missing cubes: {missing}\n")
    f.write(f"Target min: {y.min():.4f}\n")
    f.write(f"Target max: {y.max():.4f}\n")
    f.write(f"Target mean: {y.mean():.4f}\n")
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"R2: {r2:.6f}\n")

print()
print("Saved regression_baseline.txt")
print("=" * 70)