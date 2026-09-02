import numpy as np

EXPECTED_SHAPE = (128, 128, 125)


def extract_statistical_features(cube: np.ndarray) -> np.ndarray:
    """Create spectral and spatial summary features from one valid cube."""
    if cube.shape != EXPECTED_SHAPE:
        raise ValueError(f"Unexpected shape: {cube.shape}")

    foreground_mask = np.any(cube > 0, axis=2)

    if not foreground_mask.any():
        raise ValueError("Blank cube")

    pixels = cube[foreground_mask].astype(np.float32)

    mean_spectrum = pixels.mean(axis=0)
    std_spectrum = pixels.std(axis=0)
    foreground_fraction = np.array(
        [foreground_mask.mean()],
        dtype=np.float32,
    )

    return np.concatenate(
        [mean_spectrum, std_spectrum, foreground_fraction]
    )