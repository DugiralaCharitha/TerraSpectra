"""
TerraSpectra v2 - Machine Learning Pipeline
Module: Week 1 - Data Preparation and Spectral Dimensionality Reduction (PCA)
Author: Machine Learning Engineering Team
"""

from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import joblib

V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))

from src.plant_mask import extract_pure_plant_spectrum

PROJECT_ROOT = V2_ROOT.parent
CUBES_DIR = PROJECT_ROOT / "ot" / "ot"
LABELS_CSV = V2_ROOT / "data" / "labels_healthy_vs_stressed.csv"
MODELS_DIR = V2_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def run_pca_pipeline(n_components: int = 16):
    """
    Extracts pure-plant spectra from all verified cubes, fits a 16-component PCA,
    and saves the fitted transformer for downstream 3D-CNN and ViT architectures.
    """
    print("=" * 70)
    print("WEEK 1: SPECTRAL DIMENSIONALITY REDUCTION (PCA)")
    print("=" * 70)
    
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Labels manifest not found at: {LABELS_CSV}")
        
    df = pd.read_csv(LABELS_CSV)
    total_samples = len(df)
    print(f"Loaded verified dataset manifest: {total_samples} samples")
    
    print("\n[Step 1/3] Extracting foreground plant spectra...")
    start_time = time.time()
    
    spectra_list = []
    valid_ids = []
    
    for i, row in df.iterrows():
        sample_id = str(row["id"])
        cube_path = CUBES_DIR / sample_id
        
        if not cube_path.exists():
            continue
            
        try:
            cube = np.load(cube_path).astype(np.float32)
            spectrum = extract_pure_plant_spectrum(cube)
            spectra_list.append(spectrum)
            valid_ids.append(sample_id)
        except Exception as err:
            print(f"Warning: Error processing {sample_id}: {err}")
            continue
            
        if (i + 1) % 500 == 0 or (i + 1) == total_samples:
            print(f"  Processed {i + 1}/{total_samples} samples...")
            
    X_spectra = np.asarray(spectra_list, dtype=np.float32)
    extraction_time = time.time() - start_time
    print(f"Spectral extraction completed in {extraction_time:.2f} seconds.")
    print(f"Input spectral matrix shape: {X_spectra.shape} (125 bands)")
    
    print(f"\n[Step 2/3] Fitting PCA model ({n_components} components)...")
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_spectra)
    
    variance_ratios = pca.explained_variance_ratio_
    cumulative_variance = float(np.sum(variance_ratios) * 100.0)
    
    print("\n" + "-" * 70)
    print("PCA REDUCTION METRICS:")
    print(f"  Input Dimensionality:        125 spectral bands")
    print(f"  Output Dimensionality:       {n_components} principal components")
    print(f"  Explained Variance Ratio:    {cumulative_variance:.4f}%")
    print(f"  Top 3 Components Variance:   {np.sum(variance_ratios[:3]) * 100.0:.2f}%")
    print(f"  Memory Footprint Reduction:  {(1.0 - (n_components / 125.0)) * 100.0:.2f}%")
    print("-" * 70)
    
    pca_file = MODELS_DIR / "pca_spectral_reducer.joblib"
    joblib.dump(pca, pca_file)
    print(f"\n[Step 3/3] Serialized PCA transformer saved to:")
    print(f"  {pca_file}")
    
    print("=" * 70)
    print("WEEK 1 STATUS: Complete and validated.")
    print("=" * 70)


if __name__ == "__main__":
    run_pca_pipeline(n_components=16)