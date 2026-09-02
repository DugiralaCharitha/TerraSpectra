"""
WEEK 3 DELIVERABLE: Spectral Vision Transformer (ViT) in PyTorch
-----------------------------------------------------------------
Applies Multi-Head Self-Attention across hyperspectral spatial-spectral tokens
to capture long-range chemical correlations and subtle chlorophyll shifts.
Outputs 2 classes: 0 (Healthy) vs 1 (Chemically Stressed).
"""

import math
import torch
import torch.nn as nn


class PatchEmbed3D(nn.Module):
    """
    Splits the 3D cube (1, Depth=16, Height=32, Width=32) into non-overlapping patches
    and projects them into an embedding dimension D.
    """
    def __init__(self, patch_size: int = 8, in_channels: int = 1, embed_dim: int = 64, spectral_depth: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (32 // patch_size) * (32 // patch_size)  # (4 x 4) = 16 spatial patches
        
        # 3D convolution to extract patch embeddings across the full spectral depth
        self.proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=(spectral_depth, patch_size, patch_size),
            stride=(spectral_depth, patch_size, patch_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 1, 16, 32, 32)
        x = self.proj(x)  # -> (B, embed_dim, 1, H_p, W_p)
        x = x.squeeze(2)  # -> (B, embed_dim, H_p, W_p)
        x = x.flatten(2).transpose(1, 2)  # -> (B, num_patches, embed_dim)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Standard ViT Transformer Encoder block with Multi-Head Self-Attention + MLP.
    """
    def __init__(self, embed_dim: int = 64, num_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-Attention with residual connection
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class SpectralViT(nn.Module):
    """
    Complete Spectral Vision Transformer Architecture:
    1. Patch Embedding across 16 PCA spectral bands.
    2. Learnable [CLS] classification token + 1D Positional Embeddings.
    3. Multi-layer Transformer Encoder.
    4. Classification MLP Head.
    """
    def __init__(
        self,
        spectral_depth: int = 16,
        patch_size: int = 8,
        embed_dim: int = 64,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size,
            in_channels=1,
            embed_dim=embed_dim,
            spectral_depth=spectral_depth
        )
        
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Stack of Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification Head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # 1. Patchify & Embed: (B, 16, embed_dim)
        x = self.patch_embed(x)
        
        # 2. Prepend CLS token: (B, 17, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 3. Add positional encoding
        x = self.pos_drop(x + self.pos_embed)
        
        # 4. Pass through Transformer Encoder layers
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        # 5. Extract CLS token output for classification
        cls_out = x[:, 0]
        logits = self.head(cls_out)
        return logits


if __name__ == "__main__":
    # Sanity check forward pass
    model = SpectralViT()
    mock_input = torch.randn(2, 1, 16, 32, 32)
    output = model(mock_input)
    print("✅ Spectral Vision Transformer (ViT) forward pass verified!")
    print(f"Input shape:  {mock_input.shape}")
    print(f"Output shape: {output.shape} (2 logits for Healthy vs. Stressed)")
