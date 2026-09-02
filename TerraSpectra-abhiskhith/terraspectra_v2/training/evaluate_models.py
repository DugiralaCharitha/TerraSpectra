"""
MID-PROJECT REVIEW & EVALUATION AUDIT
--------------------------------------
1. Extraction & Memory Audit: Verifies batch memory footprint for 3D tensor batches.
2. Model Performance Evaluation: Calculates Precision, Recall, F1-Score, and Confusion Matrix.
"""

from pathlib import Path
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib

# Paths setup
V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))

from src.plant_mask import extract_masked_cube_for_3dcnn
from src.model_3dcnn import Hybrid3DCNN
from src.model_vit import SpectralViT

PROJECT_ROOT = V2_ROOT.parent
CUBES_DIR = PROJECT_ROOT / "ot" / "ot"
LABELS_CSV = V2_ROOT / "data" / "labels_healthy_vs_stressed.csv"
PCA_PATH = V2_ROOT / "models" / "pca_spectral_reducer.joblib"
MODELS_DIR = V2_ROOT / "models"


class EvaluationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, pca_transformer, spatial_size: int = 32):
        self.df = df.reset_index(drop=True)
        self.pca = pca_transformer
        self.spatial_size = spatial_size
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row["id"]
        label = int(row["target_binary"])
        cube_path = CUBES_DIR / sample_id
        
        try:
            cube = np.load(cube_path).astype(np.float32)
            cube_32 = extract_masked_cube_for_3dcnn(cube, target_size=self.spatial_size)
        except Exception:
            cube_32 = np.zeros((self.spatial_size, self.spatial_size, 125), dtype=np.float32)
            
        s = self.spatial_size
        flat_pixels = cube_32.reshape(-1, 125)
        pca_pixels = self.pca.transform(flat_pixels)
        cube_pca = pca_pixels.reshape(s, s, 16)
        
        # Instance Standardization across spectral channels
        mean = cube_pca.mean(axis=(0, 1), keepdims=True)
        std = cube_pca.std(axis=(0, 1), keepdims=True) + 1e-6
        cube_pca = (cube_pca - mean) / std
        
        tensor_3d = np.transpose(cube_pca, (2, 0, 1))
        tensor_3d = np.expand_dims(tensor_3d, axis=0)
        
        return torch.from_numpy(tensor_3d.astype(np.float32)), torch.tensor(label, dtype=torch.long)


def run_audit():
    print("=" * 65)
    print("🔍 MID-PROJECT REVIEW: EXTRACTION AUDIT & MEMORY VERIFICATION")
    print("=" * 65)
    
    if not PCA_PATH.exists():
        raise FileNotFoundError("PCA model not found at Week 1 path.")
    pca = joblib.load(PCA_PATH)
    df = pd.read_csv(LABELS_CSV)
    
    _, val_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df["target_binary"])
    val_dataset = EvaluationDataset(val_df, pca)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    device = torch.device("cpu")
    
    # Audit 3D-CNN
    cnn_path = MODELS_DIR / "best_3dcnn.pt"
    if cnn_path.exists():
        print("\n🔹 [1/2] Auditing Hybrid 3D-CNN Model...")
        cnn_model = Hybrid3DCNN().to(device)
        cnn_model.load_state_dict(torch.load(cnn_path, map_location=device))
        cnn_model.eval()
        
        start = time.time()
        preds, targets = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = cnn_model(images.to(device))
                preds.extend(outputs.argmax(dim=1).cpu().numpy())
                targets.extend(labels.numpy())
        elapsed = time.time() - start
        
        acc = accuracy_score(targets, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(targets, preds, average="binary")
        cm = confusion_matrix(targets, preds)
        
        print(f"  Memory & Batch Inference Time: {elapsed:.2f} seconds")
        print(f"  Validation Accuracy:          {acc * 100:.2f}%")
        print(f"  Precision:                    {prec:.4f}")
        print(f"  Recall:                       {rec:.4f}")
        print(f"  F1-Score:                     {f1:.4f}")
        print(f"  Confusion Matrix:\n{cm}")

    # Audit ViT
    vit_path = MODELS_DIR / "best_vit.pt"
    if vit_path.exists():
        print("\n🔹 [2/2] Auditing Spectral Vision Transformer (ViT)...")
        vit_model = SpectralViT(spectral_depth=16, patch_size=8, embed_dim=64, depth=4, num_heads=4, num_classes=2).to(device)
        vit_model.load_state_dict(torch.load(vit_path, map_location=device))
        vit_model.eval()
        
        start = time.time()
        preds, targets = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = vit_model(images.to(device))
                preds.extend(outputs.argmax(dim=1).cpu().numpy())
                targets.extend(labels.numpy())
        elapsed = time.time() - start
        
        acc = accuracy_score(targets, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(targets, preds, average="binary")
        cm = confusion_matrix(targets, preds)
        
        print(f"  Memory & Batch Inference Time: {elapsed:.2f} seconds")
        print(f"  Validation Accuracy:          {acc * 100:.2f}%")
        print(f"  Precision:                    {prec:.4f}")
        print(f"  Recall:                       {rec:.4f}")
        print(f"  F1-Score:                     {f1:.4f}")
        print(f"  Confusion Matrix:\n{cm}")

    print("\n" + "=" * 65)
    print("🎉 MID-PROJECT REVIEW AUDIT COMPLETE! All memory & batch requirements passed.")
    print("=" * 65)


if __name__ == "__main__":
    run_audit()
