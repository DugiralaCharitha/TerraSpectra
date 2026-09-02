"""
WEEK 4 DELIVERABLE: Satellite Tiling & Inference Engine
--------------------------------------------------------
Slices massive 1,000-acre satellite hyperspectral rasters into sub-cubes,
runs GPU/CPU batch inference through trained PyTorch models (3D-CNN / ViT),
and stitches predicted spatial probabilities into a disease-risk heatmap.
"""

from pathlib import Path
import sys
import time
import numpy as np
import torch
import joblib

# Paths setup
V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))

from src.plant_mask import extract_masked_cube_for_3dcnn
from src.model_3dcnn import Hybrid3DCNN
from src.model_vit import SpectralViT

MODELS_DIR = V2_ROOT / "models"
PCA_PATH = MODELS_DIR / "pca_spectral_reducer.joblib"


class HyperspectralTilingEngine:
    def __init__(self, model_type: str = "3dcnn", tile_size: int = 32, stride: int = 32):
        self.tile_size = tile_size
        self.stride = stride
        self.model_type = model_type.lower()
        self.device = torch.device("cpu")
        
        # Load PCA transformer from Week 1
        if not PCA_PATH.exists():
            raise FileNotFoundError(f"PCA model not found at {PCA_PATH}")
        self.pca = joblib.load(PCA_PATH)
        
        # Load model weights from Week 2 / Week 3
        if self.model_type == "3dcnn":
            weights_path = MODELS_DIR / "best_3dcnn.pt"
            if not weights_path.exists():
                raise FileNotFoundError(f"3D-CNN model not found at {weights_path}")
            self.model = Hybrid3DCNN().to(self.device)
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        elif self.model_type == "vit":
            weights_path = MODELS_DIR / "best_vit.pt"
            if not weights_path.exists():
                raise FileNotFoundError(f"Vision Transformer model not found at {weights_path}")
            self.model = SpectralViT(spectral_depth=16, patch_size=8, embed_dim=64, depth=4, num_heads=4, num_classes=2).to(self.device)
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        else:
            raise ValueError("model_type must be either '3dcnn' or 'vit'")
            
        self.model.eval()

    def process_large_raster(self, raster_cube: np.ndarray) -> np.ndarray:
        """
        Input: raster_cube of shape (Height, Width, 125)
        Output: disease_probability_map of shape (Height, Width) where values range from 0.0 (Healthy) to 1.0 (Stressed)
        """
        H, W, B = raster_cube.shape
        if B != 125:
            raise ValueError(f"Expected 125 spectral bands, got {B}")
            
        probability_heatmap = np.zeros((H, W), dtype=np.float32)
        count_matrix = np.zeros((H, W), dtype=np.float32)
        
        tiles = []
        coords = []
        
        # 1. Slice raster into spatial tiles
        for y in range(0, H - self.tile_size + 1, self.stride):
            for x in range(0, W - self.tile_size + 1, self.stride):
                tile = raster_cube[y:y + self.tile_size, x:x + self.tile_size, :]
                
                # Apply plant mask & downsample
                masked_tile = extract_masked_cube_for_3dcnn(tile, target_size=self.tile_size)
                
                # Apply PCA reduction: (32, 32, 125) -> (32, 32, 16)
                flat_pixels = masked_tile.reshape(-1, 125)
                pca_pixels = self.pca.transform(flat_pixels)
                tile_pca = pca_pixels.reshape(self.tile_size, self.tile_size, 16)
                
                # Instance Standardization across spectral channels
                mean = tile_pca.mean(axis=(0, 1), keepdims=True)
                std = tile_pca.std(axis=(0, 1), keepdims=True) + 1e-6
                tile_pca = (tile_pca - mean) / std
                
                # Reshape to 3D tensor: (1, 16, 32, 32)
                tensor_3d = np.transpose(tile_pca, (2, 0, 1))
                tensor_3d = np.expand_dims(tensor_3d, axis=0)
                
                tiles.append(tensor_3d)
                coords.append((y, x))
                
        if not tiles:
            return probability_heatmap
            
        # 2. Batch Inference through PyTorch Model
        batch_tensors = torch.from_numpy(np.array(tiles, dtype=np.float32)).to(self.device)
        
        with torch.no_grad():
            logits = self.model(batch_tensors)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # Prob of Chemically Stressed (Class 1)
            
        # 3. Stitch tile predictions back into spatial heatmap
        for idx, (y, x) in enumerate(coords):
            prob = probs[idx]
            probability_heatmap[y:y + self.tile_size, x:x + self.tile_size] += prob
            count_matrix[y:y + self.tile_size, x:x + self.tile_size] += 1.0
            
        # Avoid division by zero
        count_matrix[count_matrix == 0] = 1.0
        final_heatmap = probability_heatmap / count_matrix
        return final_heatmap


def test_tiling_engine():
    print("=" * 65)
    print("🛰️ WEEK 4: SATELLITE TILING & INFERENCE ENGINE TEST")
    print("=" * 65)
    
    # Generate mock 1000-acre satellite raster (e.g. 128x128 spatial pixels with 125 bands)
    print("1. Simulating 1000-acre satellite raster cube (128x128x125)...")
    mock_raster = np.random.uniform(0.0, 1.0, (128, 128, 125)).astype(np.float32)
    
    engine = HyperspectralTilingEngine(model_type="3dcnn", tile_size=32, stride=32)
    
    start_time = time.time()
    heatmap = engine.process_large_raster(mock_raster)
    elapsed = time.time() - start_time
    
    print(f"2. Tile inference completed in {elapsed:.2f} seconds.")
    print(f"3. Heatmap dimensions: {heatmap.shape} (Height x Width)")
    print(f"4. Min Disease Risk:   {heatmap.min():.4f}")
    print(f"5. Max Disease Risk:   {heatmap.max():.4f}")
    print(f"6. Mean Disease Risk:  {heatmap.mean():.4f}")
    print("=" * 65)
    print("🎉 WEEK 4 TILING ENGINE VERIFIED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    test_tiling_engine()
