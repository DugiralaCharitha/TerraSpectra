"""Reusable CPU predictor for the saved hyperspectral 3D-CNN checkpoint."""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.train_3dcnn import NORMALIZATION_DIVISOR, Spectral3DCNN


@dataclass(frozen=True)
class Prediction:
    """Model result suitable for serialising in an API response."""

    predicted_label: int
    confidence: float

    def to_dict(self):
        return asdict(self)


class HyperspectralPredictor:
    """Loads one checkpoint once and predicts individual NPY hyperspectral cubes."""

    def __init__(self, checkpoint_path: Path | str = PROJECT_ROOT / "ml" / "experiments" / "3dcnn" / "best_model.pt"):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.spatial_size = checkpoint["spatial_size"]
        self.model = Spectral3DCNN(checkpoint["num_classes"], width=checkpoint.get("width", 8))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def _prepare_cube(self, cube: np.ndarray) -> torch.Tensor:
        if cube.ndim != 3 or cube.shape[2] != 125:
            raise ValueError(f"Expected (height, width, 125) cube, got {cube.shape}")
        cube = cube.astype(np.float32, copy=False)
        if cube.shape[:2] == (128, 128) and 128 % self.spatial_size == 0:
            factor = 128 // self.spatial_size
            cube = cube.reshape(self.spatial_size, factor, self.spatial_size, factor, 125)
            cube = cube.mean(axis=(1, 3), dtype=np.float32).transpose(2, 0, 1)
            image = torch.from_numpy(np.ascontiguousarray(cube)).unsqueeze(0)
        else:
            image = torch.from_numpy(np.ascontiguousarray(cube.transpose(2, 0, 1))).unsqueeze(0)
            image = functional.interpolate(image.unsqueeze(0), size=(125, self.spatial_size, self.spatial_size), mode="trilinear", align_corners=False).squeeze(0)
        return image.unsqueeze(0) / NORMALIZATION_DIVISOR

    def predict_cube(self, cube: np.ndarray) -> Prediction:
        with torch.no_grad():
            probabilities = self.model(self._prepare_cube(cube)).softmax(dim=1)[0]
        label = int(probabilities.argmax().item())
        return Prediction(predicted_label=label, confidence=float(probabilities[label].item()))

    def predict_file(self, cube_path: Path | str) -> Prediction:
        return self.predict_cube(np.load(cube_path))
