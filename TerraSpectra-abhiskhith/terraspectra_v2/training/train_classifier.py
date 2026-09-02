"""
TerraSpectra v2: High-Accuracy Crop Stress Classifier
------------------------------------------------------
Trains a balanced classifier on pure-plant spectral features & NDVI/NDRE indices.
Directly fulfills the Week 2 Infotact Solutions deliverable:
'Classify healthy vs. chemically stressed plants'
"""

from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
import joblib

# Paths setup
V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))

from src.plant_mask import extract_pure_plant_spectrum
from src.spectral_features import extract_spectral_feature_vector

PROJECT_ROOT = V2_ROOT.parent
CUBES_DIR = PROJECT_ROOT / "ot" / "ot"
LABELS_CSV = V2_ROOT / "data" / "labels_healthy_vs_stressed.csv"
MODELS_DIR = V2_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_model():
    print("=" * 65)
    print("🚀 TRAINING HIGH-ACCURACY HEALTHY VS. STRESSED CLASSIFIER")
    print("=" * 65)
    
    df = pd.read_csv(LABELS_CSV)
    print(f"Total labeled samples available: {len(df)}")
    
    # Extract features for all samples
    print("\nExtracting pure-leaf spectral signatures & vegetation indices...")
    start_time = time.time()
    
    features_list = []
    labels_list = []
    valid_ids = []
    
    for i, row in df.iterrows():
        sample_id = row["id"]
        cube_path = CUBES_DIR / sample_id
        
        if not cube_path.exists():
            continue
            
        try:
            cube = np.load(cube_path).astype(np.float32)
            pure_spectrum = extract_pure_plant_spectrum(cube)
            feature_vector = extract_spectral_feature_vector(pure_spectrum)
            
            features_list.append(feature_vector)
            labels_list.append(int(row["target_binary"]))
            valid_ids.append(sample_id)
        except Exception:
            continue
            
        if (i + 1) % 400 == 0 or (i + 1) == len(df):
            print(f"  • Processed {i + 1}/{len(df)} samples...")
            
    X = np.asarray(features_list, dtype=np.float32)
    y = np.asarray(labels_list, dtype=np.int64)
    
    elapsed = time.time() - start_time
    print(f"✅ Features extracted successfully in {elapsed:.1f}s!")
    print(f"Feature matrix shape: {X.shape} (378 spectral & index features per sample)")
    
    # Stratified Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Testing/Validation set: {len(X_test)} samples")
    
    # Train Balanced ExtraTrees Classifier
    print("\nTraining the model...")
    model = ExtraTreesClassifier(
        n_estimators=150,
        max_depth=15,
        class_weight="balanced",  # Ensures Healthy plants are given equal importance
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, predictions) * 100.0
    bal_acc = balanced_accuracy_score(y_test, predictions) * 100.0
    roc_auc = roc_auc_score(y_test, probabilities) * 100.0
    cm = confusion_matrix(y_test, predictions)
    
    print("\n" + "=" * 65)
    print("🎯 FINAL MODEL PERFORMANCE")
    print("=" * 65)
    print(f"  ⭐ Accuracy:          {acc:.2f}%")
    print(f"  ⭐ Balanced Accuracy: {bal_acc:.2f}%")
    print(f"  ⭐ ROC-AUC Score:     {roc_auc:.2f}%")
    print("\nDetailed Breakdown:")
    print(classification_report(y_test, predictions, target_names=["Healthy", "Chemically Stressed"]))
    print("Confusion Matrix:")
    print(f"  [[Healthy correct: {cm[0][0]},  Healthy mistaken: {cm[0][1]}],")
    print(f"   [Stressed mistaken: {cm[1][0]}, Stressed correct: {cm[1][1]}]]")
    
    # Save the trained model
    save_path = MODELS_DIR / "crop_stress_classifier.joblib"
    joblib.dump(model, save_path)
    print(f"\n💾 Model successfully saved to: {save_path}")
    print("=" * 65)


if __name__ == "__main__":
    train_model()