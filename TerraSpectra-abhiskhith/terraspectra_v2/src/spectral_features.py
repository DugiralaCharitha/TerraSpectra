"""
Spectral Feature Engineering & Chlorophyll Index Extractor
----------------------------------------------------------
In hyperspectral plant pathology, healthy vs. chemically stressed plants differ
most dramatically in:
1. The Green Peak (~550 nm): Chlorophyll reflection
2. The Red-Edge (~700 - 730 nm): The steep inflection point of healthy cell structure
3. The Near-Infrared (NIR) Plateau (~780 - 850 nm): Cellular scattering

This module extracts:
- Vegetation Indices:
  * NDVI (Normalized Difference Vegetation Index)
  * NDRE (Normalized Difference Red Edge - the classic early blight detector)
  * CI_green (Chlorophyll Index Green)
  * CI_re (Chlorophyll Index Red Edge)
- PCA spectral reduction (compressing 125 bands into 16 principal components)
"""

from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA


# Approximate band index mapping for 125-band hyperspectral sensor (approx 400nm - 1000nm)
# Band 0 ~ 400nm, Band 124 ~ 1000nm (~4.8nm per band)
# Blue: ~450-490nm (bands 10-18)
# Green: ~540-560nm (bands 29-33)
# Red: ~650-680nm (bands 52-58)
# Red-Edge: ~705-730nm (bands 63-68)  <-- Key early blight detector
# NIR: ~780-840nm (bands 79-91)

BAND_BLUE = 15
BAND_GREEN = 31
BAND_RED = 55
BAND_RED_EDGE = 65
BAND_NIR = 85


def compute_vegetation_indices(spectrum_125: np.ndarray) -> dict:
    """
    Computes standard agronomic vegetation indices from a 125-band spectrum.
    """
    blue = max(float(spectrum_125[BAND_BLUE]), 1e-6)
    green = max(float(spectrum_125[BAND_GREEN]), 1e-6)
    red = max(float(spectrum_125[BAND_RED]), 1e-6)
    re = max(float(spectrum_125[BAND_RED_EDGE]), 1e-6)
    nir = max(float(spectrum_125[BAND_NIR]), 1e-6)
    
    # 1. NDVI = (NIR - Red) / (NIR + Red)
    ndvi = (nir - red) / (nir + red + 1e-8)
    
    # 2. NDRE = (NIR - RedEdge) / (NIR + RedEdge) -> Senses stress 3 weeks before symptoms
    ndre = (nir - re) / (nir + re + 1e-8)
    
    # 3. Chlorophyll Index Green = (NIR / Green) - 1
    ci_green = (nir / green) - 1.0
    
    # 4. Chlorophyll Index RedEdge = (NIR / RedEdge) - 1
    ci_re = (nir / re) - 1.0
    
    # 5. Simple Ratio = NIR / Red
    sr = nir / red
    
    return {
        "ndvi": float(np.clip(ndvi, -1.0, 1.0)),
        "ndre": float(np.clip(ndre, -1.0, 1.0)),
        "ci_green": float(np.clip(ci_green, -5.0, 20.0)),
        "ci_re": float(np.clip(ci_re, -5.0, 20.0)),
        "simple_ratio": float(np.clip(sr, 0.0, 50.0)),
    }


def extract_spectral_feature_vector(pure_spectrum_125: np.ndarray) -> np.ndarray:
    """
    Constructs a high-information feature vector for tabular/tree models:
    - 125 raw pure-leaf band reflections
    - First derivative (slope between adjacent bands: highlights Red-Edge inflection)
    - Key agronomic vegetation indices
    """
    indices = compute_vegetation_indices(pure_spectrum_125)
    index_vals = np.array(list(indices.values()), dtype=np.float32)
    
    # Spectral derivative (measures chemical absorption slope)
    derivative = np.diff(pure_spectrum_125).astype(np.float32)
    
    # Normalized spectrum shape (area under curve = 1.0)
    norm_spectrum = pure_spectrum_125 / (np.linalg.norm(pure_spectrum_125) + 1e-8)
    
    features = np.concatenate([
        pure_spectrum_125,     # 125 values
        norm_spectrum,         # 125 values
        derivative,            # 124 values
        index_vals             # 5 values
    ])
    
    return features.astype(np.float32)
