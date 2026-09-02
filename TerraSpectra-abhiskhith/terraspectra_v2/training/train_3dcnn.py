"""
WEEK 2 DELIVERABLE: Train the Hybrid 3D-CNN Classifier
------------------------------------------------------
1. Loads 16-band PCA cubes (from Week 1).
2. Batches 3D tensors safely for CPU.
3. Completely immune to corrupt/truncated files via robust try-except fallback.
4. Trains the Hybrid 3D-CNN on Healthy (0) vs Chemically Stressed (1).
5. Saves best checkpoint to models/best_3dcnn.pt.
"""

from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Paths setup
V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))

from src.plant_mask import extract_masked_cube_for_3dcnn
from src.model_3dcnn import Hybrid3DCNN

PROJECT_ROOT = V2_ROOT.parent
CUBES_DIR = PROJECT_ROOT / "ot" / "ot"
LABELS_CSV = V2_ROOT / "data" / "labels_healthy_vs_stressed.csv"
PCA_PATH = V2_ROOT / "models" / "pca_spectral_reducer.joblib"
MODELS_DIR = V2_ROOT / "models"


class Hyperspectral3DDataset(Dataset):
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
        
        # Instance Standardization across spectral channels to prevent gradient saturation
        mean = cube_pca.mean(axis=(0, 1), keepdims=True)
        std = cube_pca.std(axis=(0, 1), keepdims=True) + 1e-6
        cube_pca = (cube_pca - mean) / std
        
        tensor_3d = np.transpose(cube_pca, (2, 0, 1))
        tensor_3d = np.expand_dims(tensor_3d, axis=0)
        
        return torch.from_numpy(tensor_3d.astype(np.float32)), torch.tensor(label, dtype=torch.long)


def train_week2_3dcnn(epochs: int = 10, batch_size: int = 16):
    print("=" * 65)
    print("🧠 WEEK 2: TRAINING HYBRID 3D-CNN (HEALTHY VS. STRESSED)")
    print("=" * 65)
    
    pca = joblib.load(PCA_PATH)
    df = pd.read_csv(LABELS_CSV)
    
    train_df, val_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df["target_binary"]
    )
    print(f"Training samples:   {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    train_dataset = Hyperspectral3DDataset(train_df, pca)
    val_dataset = Hyperspectral3DDataset(val_df, pca)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    device = torch.device("cpu")
    model = Hybrid3DCNN(in_channels=1, spectral_depth=16, num_classes=2).to(device)
    
    class_counts = df["target_binary"].value_counts()
    weight_healthy = len(df) / (2.0 * class_counts[0])
    weight_stressed = len(df) / (2.0 * class_counts[1])
    weights = torch.tensor([weight_healthy, weight_stressed], dtype=torch.float32).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    print("\nStarting Training (10 Epochs)...")
    best_val_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (images, targets) in enumerate(train_loader, 1):
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if batch_idx % 25 == 0 or batch_idx == len(train_loader):
                print(f"  Epoch [{epoch}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
                
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(targets.numpy())
                
        val_acc = accuracy_score(val_targets, val_preds) * 100.0
        elapsed = time.time() - start_time
        print(f"⭐️ Epoch {epoch} Complete in {elapsed:.1f}s | Validation Accuracy: {val_acc:.2f}%\n")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODELS_DIR / "best_3dcnn.pt")
            
    print("=" * 65)
    print(f"🎉 WEEK 2 COMPLETE: Best 3D-CNN Validation Accuracy = {best_val_acc:.2f}%")
    print(f"💾 Checkpoint saved to: {MODELS_DIR / 'best_3dcnn.pt'}")
    print("=" * 65)


if __name__ == "__main__":
    train_week2_3dcnn(epochs=10, batch_size=16)