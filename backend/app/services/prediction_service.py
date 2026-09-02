from pathlib import Path

import joblib
import numpy as np
import torch

from app.models.model_3dcnn import Hybrid3DCNN
from app.preprocessing.plant_mask import extract_masked_cube_for_3dcnn


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

CHECKPOINT_DIR = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "checkpoints"
)

MODEL_PATH = CHECKPOINT_DIR / "best_3dcnn.pt"
PCA_PATH = CHECKPOINT_DIR / "pca_spectral_reducer.joblib"


# ---------------------------------------------------------
# MODEL SETTINGS
# ---------------------------------------------------------

DEVICE = torch.device("cpu")

EXPECTED_BANDS = 125
PCA_COMPONENTS = 16
TILE_SIZE = 32
STRIDE = 32

_model = None
_pca = None


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

def _load_model():
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {MODEL_PATH}"
            )

        _model = Hybrid3DCNN(
            in_channels=1,
            spectral_depth=16,
            num_classes=2,
        ).to(DEVICE)

        state_dict = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True,
        )

        _model.load_state_dict(state_dict)
        _model.eval()

    return _model


# ---------------------------------------------------------
# LOAD PCA
# ---------------------------------------------------------

def _load_pca():
    global _pca

    if _pca is None:
        if not PCA_PATH.exists():
            raise FileNotFoundError(
                f"PCA model not found: {PCA_PATH}"
            )

        _pca = joblib.load(PCA_PATH)

        if not hasattr(_pca, "n_components_"):
            raise ValueError(
                "Invalid PCA model: n_components_ is missing."
            )

        if _pca.n_components_ != PCA_COMPONENTS:
            raise ValueError(
                f"Expected PCA with {PCA_COMPONENTS} components, "
                f"got {_pca.n_components_}."
            )

    return _pca


# ---------------------------------------------------------
# PREDICT ONE TILE
# ---------------------------------------------------------

def _predict_tile(
    tile: np.ndarray,
    model,
    pca,
) -> float:

    masked_cube = extract_masked_cube_for_3dcnn(
        tile,
        target_size=TILE_SIZE,
    )

    pixels = masked_cube.reshape(
        -1,
        EXPECTED_BANDS,
    )

    pixels_pca = pca.transform(pixels)

    tile_pca = pixels_pca.reshape(
        TILE_SIZE,
        TILE_SIZE,
        PCA_COMPONENTS,
    )

    # Instance standardization
    mean = tile_pca.mean(
        axis=(0, 1),
        keepdims=True,
    )

    std = tile_pca.std(
        axis=(0, 1),
        keepdims=True,
    ) + 1e-6

    tile_pca = (
        tile_pca - mean
    ) / std

    # (32, 32, 16) -> (16, 32, 32)
    tile_pca = np.transpose(
        tile_pca,
        (2, 0, 1),
    )

    # Add batch dimension
    tensor = torch.from_numpy(
        tile_pca.astype(np.float32)
    ).unsqueeze(0)

    # Add Conv3D channel dimension
    # (1, 16, 32, 32)
    # ->
    # (1, 1, 16, 32, 32)
    tensor = tensor.unsqueeze(1)

    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

    # Class 1 = Chemically Stressed
    return float(
        probabilities[0, 1].item()
    )


# ---------------------------------------------------------
# MAIN PREDICTION FUNCTION
# ---------------------------------------------------------

def predict_image(image_path: str):

    """
    Run Hybrid 3D-CNN prediction on a
    hyperspectral .npy cube.

    Expected shape:

        (Height, Width, 125)
    """

    if not image_path:
        raise ValueError(
            "Image path is required."
        )

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if path.suffix.lower() != ".npy":
        raise ValueError(
            "Hyperspectral prediction requires a .npy file."
        )

    cube = np.load(path)

    if cube.ndim != 3:
        raise ValueError(
            f"Expected a 3D hyperspectral cube, "
            f"got shape {cube.shape}."
        )

    if cube.shape[2] != EXPECTED_BANDS:
        raise ValueError(
            f"Expected {EXPECTED_BANDS} spectral bands, "
            f"got {cube.shape[2]}."
        )

    model = _load_model()
    pca = _load_pca()

    tile_probabilities = []

    height, width, _ = cube.shape

    # Process 32x32 tiles
    for y in range(
        0,
        height - TILE_SIZE + 1,
        STRIDE,
    ):
        for x in range(
            0,
            width - TILE_SIZE + 1,
            STRIDE,
        ):

            tile = cube[
                y:y + TILE_SIZE,
                x:x + TILE_SIZE,
                :,
            ]

            probability = _predict_tile(
                tile,
                model,
                pca,
            )

            tile_probabilities.append(
                probability
            )

    if not tile_probabilities:
        raise ValueError(
            "Input image must be at least 32x32 pixels."
        )

    # Average tile predictions
    stressed_probability = float(
        np.mean(tile_probabilities)
    )

    # -----------------------------------------------------
    # CLASSIFICATION
    # -----------------------------------------------------

    if stressed_probability >= 0.5:

        prediction = "Chemically Stressed"
        class_name = "Chemically Stressed Crop"
        confidence = stressed_probability

        message = (
            "The hyperspectral analysis indicates "
            "chemical stress in the crop."
        )

    else:

        prediction = "Healthy"
        class_name = "Healthy Crop"
        confidence = 1.0 - stressed_probability

        message = (
            "The hyperspectral analysis indicates "
            "a healthy crop."
        )

    # -----------------------------------------------------
    # RESPONSE
    # -----------------------------------------------------

    return {
        "status": "success",
        "prediction": prediction,
        "class_name": class_name,
        "confidence": round(
            confidence,
            4,
        ),
        "stress_probability": round(
            stressed_probability,
            4,
        ),
        "tiles_processed": len(
            tile_probabilities
        ),
        "message": message,
    }