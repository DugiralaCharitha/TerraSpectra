from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ml.training.train_3dcnn import (
    HyperspectralDataset,
    read_samples,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEED = 42
NUM_SAMPLES = 20
EPOCHS = 100
BATCH_SIZE = 4
SPATIAL_SIZE = 32
LEARNING_RATE = 0.001


class Strong3DCNN(nn.Module):
    """
    Stronger 3D-CNN for hyperspectral data.

    Input:
        (batch, 1, 125, 32, 32)
    """

    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv3d(
                1,
                16,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.Conv3d(
                16,
                16,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(
                kernel_size=(2, 2, 2)
            ),

            # Block 2
            nn.Conv3d(
                16,
                32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(
                32,
                32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(
                kernel_size=(2, 2, 2)
            ),

            # Block 3
            nn.Conv3d(
                32,
                64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(
                64,
                64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            # Preserve remaining information
            nn.AdaptiveAvgPool3d(
                (4, 4, 4)
            ),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                64 * 4 * 4 * 4,
                256,
            ),

            nn.ReLU(inplace=True),

            # No dropout during overfit test
            nn.Linear(
                256,
                num_classes,
            ),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def main():

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Loading dataset...")

    samples = read_samples(
        PROJECT_ROOT / "train_final.csv",
        PROJECT_ROOT / "ot" / "ot",
    )

    num_classes = (
        max(sample.label for sample in samples)
        + 1
    )

    print(f"Total valid samples: {len(samples)}")
    print(f"Number of classes: {num_classes}")

    # Fixed tiny subset
    tiny_samples = samples[:NUM_SAMPLES]

    dataset = HyperspectralDataset(
        tiny_samples,
        PROJECT_ROOT / "ot" / "ot",
        spatial_size=SPATIAL_SIZE,
        augment=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    model = Strong3DCNN(
        num_classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()
    print(f"Device: {device}")
    print(
        f"Testing stronger 3D-CNN "
        f"on only {NUM_SAMPLES} samples"
    )
    print(
        f"Model parameters: "
        f"{parameter_count:,}"
    )
    print()

    for epoch in range(1, EPOCHS + 1):

        model.train()

        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(images)

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            predictions = logits.argmax(
                dim=1
            )

            total_correct += (
                predictions == labels
            ).sum().item()

            total_examples += labels.size(0)

        average_loss = (
            total_loss
            / total_examples
        )

        accuracy = (
            total_correct
            / total_examples
        )

        print(
            f"Epoch {epoch:03d}/{EPOCHS}: "
            f"loss={average_loss:.4f}; "
            f"accuracy={accuracy:.2%}"
        )

        if accuracy >= 0.95:

            print()
            print(
                "SUCCESS: Stronger 3D-CNN "
                "can memorize the tiny dataset."
            )
            print(
                "The data pipeline is working. "
                "The original model was likely too weak."
            )

            break


if __name__ == "__main__":
    main()