from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------
# Paths
# --------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

CUBES = ROOT / "ot" / "ot"

CHECKPOINT = (
    ROOT
    / "ml"
    / "experiments"
    / "day2_regression_v3"
    / "best_model.pt"
)


# --------------------------------------------------------
# Model
# --------------------------------------------------------

class Strong3DCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv3d(
                1,
                16,
                (7, 3, 3),
                stride=(2, 1, 1),
                padding=(3, 1, 1),
                bias=False
            ),

            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(
                16,
                32,
                3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(
                32,
                64,
                3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(
                64,
                64,
                3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool3d(1)
        )

        self.regressor = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.30),

            nn.Linear(64, 32),

            nn.ReLU(inplace=True),

            nn.Dropout(0.20),

            nn.Linear(32, 1),

            nn.Sigmoid()
        )

    def forward(self, x):

        return self.regressor(
            self.features(x)
        ).squeeze(1)


# --------------------------------------------------------
# Load model
# --------------------------------------------------------

def load_model():

    model = Strong3DCNN()

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


# --------------------------------------------------------
# Prediction
# --------------------------------------------------------

def predict(filename):

    path = CUBES / filename

    if not path.exists():

        raise FileNotFoundError(
            f"Cube not found: {path}"
        )

    cube = np.load(
        path
    ).astype(np.float32)

    print(
        f"Original cube shape: {cube.shape}"
    )

    # ----------------------------------------------------
    # Validate hyperspectral bands
    # ----------------------------------------------------

    if cube.ndim != 3:

        raise ValueError(
            f"Expected 3D cube, got {cube.shape}"
        )

    if cube.shape[2] != 125:

        raise ValueError(
            f"Expected 125 spectral bands, got {cube.shape[2]}"
        )

    # ----------------------------------------------------
    # Convert any spatial size to 128 x 128
    # ----------------------------------------------------

    fixed = np.zeros(
        (128, 128, 125),
        dtype=np.float32
    )

    h = min(
        cube.shape[0],
        128
    )

    w = min(
        cube.shape[1],
        128
    )

    fixed[
        :h,
        :w,
        :
    ] = cube[
        :h,
        :w,
        :
    ]

    cube = fixed

    # ----------------------------------------------------
    # 128 x 128 -> 32 x 32
    # ----------------------------------------------------

    cube = cube.reshape(
        32,
        4,
        32,
        4,
        125
    )

    cube = cube.mean(
        axis=(1, 3),
        dtype=np.float32
    )

    # ----------------------------------------------------
    # Normalization used by V3
    # ----------------------------------------------------

    cube = cube / 28906.0

    # ----------------------------------------------------
    # Convert:
    #
    # (32, 32, 125)
    #       ↓
    # (125, 32, 32)
    #       ↓
    # (1, 1, 125, 32, 32)
    # ----------------------------------------------------

    cube = np.transpose(
        cube,
        (2, 0, 1)
    )

    cube = np.ascontiguousarray(
        cube
    )

    x = torch.from_numpy(
        cube
    ).unsqueeze(0).unsqueeze(0)

    # ----------------------------------------------------
    # Model
    # ----------------------------------------------------

    model = load_model()

    # ----------------------------------------------------
    # Prediction
    # ----------------------------------------------------

    with torch.no_grad():

        prediction = model(
            x
        ).item()

    prediction = prediction * 100.0

    return prediction


# --------------------------------------------------------
# Main
# --------------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "py -3.12 ml\\inference\\predict_regression.py sampleXXXX.npy"
        )

        sys.exit(1)

    filename = sys.argv[1]

    prediction = predict(
        filename
    )

    print()
    print(
        "REGRESSION PREDICTION"
    )
    print(
        "====================="
    )
    print(
        f"Sample: {filename}"
    )
    print(
        f"Predicted value: {prediction:.2f}"
    )
    print(
        "Done."
    )