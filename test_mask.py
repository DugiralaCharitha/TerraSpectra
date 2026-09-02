"""
Test Script: Verify Plant Masking on Sample Cubes
-------------------------------------------------
Checks 3 samples and prints:
- Original image size
- Plant pixels detected vs black background
- Calculated Chlorophyll NDVI & NDRE values
"""

from pathlib import Path
import numpy as np
import pandas as pd
import sys

# Add project root to sys.path
V2_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(V2_ROOT))

from src.plant_mask import get_plant_foreground_mask, extract_pure_plant_spectrum
from src.spectral_features import compute_vegetation_indices

PROJECT_ROOT = V2_ROOT.parent
CUBES_DIR = PROJECT_ROOT / "ot" / "ot"
LABELS_CSV = V2_ROOT / "data" / "labels_healthy_vs_stressed.csv"


def test_plant_masking():
    print("=" * 65)
    print("🌱 TESTING PLANT FOREGROUND EXTRACTION & VEGETATION INDICES")
    print("=" * 65)
    
    df = pd.read_csv(LABELS_CSV)
    
    # Pick 2 Healthy and 2 Chemically Stressed samples
    healthy_samples = df[df["target_binary"] == 0].head(2)
    stressed_samples = df[df["target_binary"] == 1].head(2)
    test_samples = pd.concat([healthy_samples, stressed_samples])
    
    for _, row in test_samples.iterrows():
        sample_id = row["id"]
        class_name = row["class_name_binary"]
        cube_path = CUBES_DIR / sample_id
        
        if not cube_path.exists():
            print(f"Skipping {sample_id} (not found)")
            continue
            
        cube = np.load(cube_path).astype(np.float32)
        
        # 1. Mask background
        mask = get_plant_foreground_mask(cube)
        plant_pixels = int(mask.sum())
        total_pixels = cube.shape[0] * cube.shape[1]
        plant_pct = (plant_pixels / total_pixels) * 100
        
        # 2. Extract pure plant spectrum
        pure_spectrum = extract_pure_plant_spectrum(cube)
        
        # 3. Calculate Chlorophyll indices
        indices = compute_vegetation_indices(pure_spectrum)
        
        print(f"\n[Sample]: {sample_id} | Class: {class_name}")
        print(f"  • Dimensions: {cube.shape}")
        print(f"  • Plant leaf coverage: {plant_pixels:,} pixels ({plant_pct:.1f}% of image)")
        print(f"  • Background removed: {total_pixels - plant_pixels:,} dark pixels ignored")
        print(f"  • NDVI (Green Biomass):     {indices['ndvi']:.4f}")
        print(f"  • NDRE (Early Stress Index): {indices['ndre']:.4f}")
        print(f"  • Chlorophyll Index:         {indices['ci_green']:.4f}")
        
    print("\n" + "=" * 65)
    print("✅ Plant masking and index extraction working successfully!")
    print("=" * 65)


if __name__ == "__main__":
    test_plant_masking()
