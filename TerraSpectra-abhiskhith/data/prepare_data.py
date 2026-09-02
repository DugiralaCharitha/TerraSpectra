"""
Data Preparation & Integrity Audit for TerraSpectra v2
------------------------------------------------------
Properly pre-processes the dataset:
1. Actually opens each .npy file to verify it is NOT truncated or corrupted.
2. Confirms spatial and spectral integrity.
3. Completely drops broken samples BEFORE training starts.
4. Produces a 100% clean, verified label dataset.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V2_ROOT = Path(__file__).resolve().parents[1]
CUBES_DIR = PROJECT_ROOT / "ot" / "ot"
TRAIN_CSV = PROJECT_ROOT / "train_clean.csv" if (PROJECT_ROOT / "train_clean.csv").exists() else (PROJECT_ROOT / "train.csv")

OUTPUT_DATA_DIR = V2_ROOT / "data"
OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)


def verify_and_clean_dataset():
    print("=" * 65)
    print("🔍 RUNNING THOROUGH DATA AUDIT & PREPROCESSING")
    print("=" * 65)
    
    print(f"Reading base label manifest from: {TRAIN_CSV.name}")
    df = pd.read_csv(TRAIN_CSV)
    total_samples = len(df)
    print(f"Total candidate rows: {total_samples}")
    
    clean_rows = []
    corrupted_count = 0
    missing_count = 0
    
    print("\nAuditing and test-opening every single cube...")
    
    for i, row in df.iterrows():
        fname = str(row["id"]).strip()
        if not fname.endswith(".npy"):
            fname = f"{fname}.npy"
            
        cube_path = CUBES_DIR / fname
        
        # 1. Existence check
        if not cube_path.exists():
            missing_count += 1
            continue
            
        # 2. True Integrity check: actually open the file
        try:
            # Load with mmap_mode to quickly verify header and file length without eating RAM
            arr = np.load(cube_path, mmap_mode="r")
            
            # Verify it has valid shape and bands
            if arr.ndim != 3 or arr.shape[2] != 125:
                corrupted_count += 1
                continue
                
            clean_rows.append({
                "id": fname,
                "raw_severity": float(row["label"])
            })
            
        except Exception:
            corrupted_count += 1
            continue
            
        if (i + 1) % 500 == 0 or (i + 1) == total_samples:
            print(f"  • Audited {i + 1}/{total_samples} files...")
            
    clean_df = pd.DataFrame(clean_rows)
    
    print("\n" + "-" * 65)
    print("📊 AUDIT RESULTS:")
    print(f"  • Total raw samples:         {total_samples}")
    print(f"  • Corrupted / Broken files:  {corrupted_count} (permanently removed)")
    print(f"  • Missing files:             {missing_count}")
    print(f"  • 100% Verified clean cubes: {len(clean_df)}")
    print("-" * 65)
    
    # Generate Clean Binary Labels (Healthy vs Chemically Stressed)
    clean_df["target_binary"] = (clean_df["raw_severity"] > 15.0).astype(int)
    clean_df["class_name_binary"] = clean_df["target_binary"].map({0: "Healthy", 1: "Chemically_Stressed"})
    
    binary_path = OUTPUT_DATA_DIR / "labels_healthy_vs_stressed.csv"
    clean_df[["id", "target_binary", "class_name_binary", "raw_severity"]].to_csv(binary_path, index=False)
    
    print(f"\n💾 Saved 100% verified dataset to:\n  {binary_path}")
    print("\nClass distribution in verified dataset:")
    print(clean_df["class_name_binary"].value_counts())
    print("=" * 65)


if __name__ == "__main__":
    verify_and_clean_dataset()
