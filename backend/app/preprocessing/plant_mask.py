"""
Plant Masking Engine (Background Suppressor)
--------------------------------------------
Solves the background dilution problem:
In hyperspectral scans, plants only cover a fraction of the image.
Averaging over black background pixels dilutes the chlorophyll signal by up to 99%.

This module:
1. Automatically identifies real plant leaf pixels.
2. Eliminates background/shadow pixels.
3. Computes pure-leaf spectral statistics without background noise.
"""

from pathlib import Path
import numpy as np


def get_plant_foreground_mask(cube: np.ndarray) -> np.ndarray:
    """
    Returns a 2D boolean mask (H, W) where True = Green Plant, False = Black Background.
    Works robustly for both raw integer cubes and float-normalized cubes of any spatial dimensions.
    """
    mean_brightness = cube.mean(axis=-1)
    max_val = float(mean_brightness.max())
    if max_val <= 0:
        return np.ones(cube.shape[:2], dtype=bool)
        
    threshold = max_val * 0.02
    mask = mean_brightness > threshold
    
    if not mask.any():
        p80 = np.percentile(mean_brightness, 80)
        mask = mean_brightness >= p80
        
    return mask


def extract_pure_plant_spectrum(cube: np.ndarray) -> np.ndarray:
    """
    Returns a 1D vector (125 bands) of the average reflection
    taken ONLY from the plant leaves (100% background-free).
    """
    mask = get_plant_foreground_mask(cube)
    plant_pixels = cube[mask]
    if len(plant_pixels) == 0:
        plant_pixels = cube.reshape(-1, cube.shape[-1])
        
    pure_spectrum = plant_pixels.mean(axis=0).astype(np.float32)
    return pure_spectrum


def extract_masked_cube_for_3dcnn(cube: np.ndarray, target_size: int = 32) -> np.ndarray:
    """
    Prepares a clean spatial-spectral cube for the 3D-CNN:
    1. Pads or crops spatial dimensions to exactly (128, 128).
    2. Masks out background.
    3. Spatially pools to target_size (e.g. 32x32).
    4. Normalizes to a stable [0, 1] range.
    """
    h, w, c = cube.shape
    
    # Handle irregular shapes (e.g. 128x57) by padding to 128x128
    if h != 128 or w != 128:
        standard_cube = np.zeros((128, 128, c), dtype=np.float32)
        h_crop = min(h, 128)
        w_crop = min(w, 128)
        standard_cube[:h_crop, :w_crop, :] = cube[:h_crop, :w_crop, :]
        cube = standard_cube

    mask = get_plant_foreground_mask(cube)
    
    masked_cube = cube.copy().astype(np.float32)
    masked_cube[~mask] = 0.0
    
    f = 128 // target_size
    downsampled = masked_cube.reshape(target_size, f, target_size, f, c).mean(axis=(1, 3))
    downsampled_mask = mask.reshape(target_size, f, target_size, f).mean(axis=(1, 3))
    
    valid = downsampled_mask > 0.05
    downsampled[valid] = downsampled[valid] / downsampled_mask[valid, None]
    downsampled[~valid] = 0.0
    
    plant_vals = downsampled[downsampled > 0]
    scale = np.percentile(plant_vals, 99) if len(plant_vals) > 0 else 1.0
    downsampled = np.clip(downsampled / max(float(scale), 1.0), 0.0, 1.0)
    
    return downsampled.astype(np.float32)
