"""
WEEK 2 DELIVERABLE: Hybrid 3D-CNN Architecture in PyTorch
---------------------------------------------------------
Captures spatial farm patterns (Height x Width) and 
spectral chemical signatures (Depth) simultaneously.
Outputs 2 classes: 0 (Healthy) vs 1 (Chemically Stressed).
"""

import torch
import torch.nn as nn


class Hybrid3DCNN(nn.Module):
    def __init__(self, in_channels: int = 1, spectral_depth: int = 16, num_classes: int = 2):
        super().__init__()
        
        # 3D Convolutional Feature Extractor
        # Input tensor shape: (Batch, 1, Depth=16, Height=32, Width=32)
        self.features = nn.Sequential(
            # Block 1: Extracts local spectral-spatial patterns
            nn.Conv3d(in_channels, 16, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),  # Downsamples depth: 16->8, spatial: 32->16
            
            # Block 2: Captures broader chemical variations across plant leaves
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),  # Downsamples depth: 8->4, spatial: 16->8
            
            # Block 3: High-level disease feature representations
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            # Global Average Pooling (keeps memory footprint tiny & CPU fast)
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        # Classification Head (Healthy vs. Chemically Stressed)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)  # 2 logits
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        logits = self.classifier(features)
        return logits


if __name__ == "__main__":
    # Sanity check forward pass
    model = Hybrid3DCNN()
    mock_input = torch.randn(2, 1, 16, 32, 32)
    output = model(mock_input)
    print("✅ Hybrid 3D-CNN forward pass verified!")
    print(f"Input shape:  {mock_input.shape}")
    print(f"Output shape: {output.shape} (2 logits for binary classification)")