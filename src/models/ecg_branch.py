"""
1D-ResNet for extracting morphological features from 250 Hz ECG waveforms.

This module implements Branch A of the multi-modal late fusion network:
  - ResidualBlock1D: basic building block with skip connections.
  - ECGFeatureExtractor: stacked residual encoder -> 128-dim embedding
    representing a proxy for Cardiac Output.
  - ECGBaselineModel: ECG-only baseline that regresses Systolic/Diastolic
    directly from the embedding (Milestone 2).
"""

import torch
import torch.nn as nn


# Building block

class ResidualBlock1D(nn.Module):
    """1-D residual block with two Conv1d layers and an optional projection
    shortcut for dimension / stride mismatches.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride for the first convolution (controls downsampling).
        dropout: Dropout probability applied after the first activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.main_path = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels,
                kernel_size=5, stride=stride, padding=2, bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv1d(
                out_channels, out_channels,
                kernel_size=5, stride=1, padding=2, bias=False,
            ),
            nn.BatchNorm1d(out_channels),
        )

        # Skip (shortcut) connection: project when dimensions change
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape ``(B, C_in, L)``.

        Returns:
            Tensor of shape ``(B, C_out, L')``.
        """
        return self.relu(self.main_path(x) + self.skip(x))


# Feature extractor (Branch A)

class ECGFeatureExtractor(nn.Module):
    """Stacked 1D-ResNet that maps a raw ECG waveform to a 128-dim embedding.

    Input shape:  ``(B, 1, seq_len)``, single-channel waveform.
    Output shape: ``(B, 128)``, Cardiac-Output proxy embedding.
    """

    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(
                1, 16,
                kernel_size=15, stride=2, padding=7, bias=False,
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = ResidualBlock1D(16, 32, stride=2, dropout=dropout)
        self.layer2 = ResidualBlock1D(32, 64, stride=2, dropout=dropout)
        self.layer3 = ResidualBlock1D(64, 128, stride=2, dropout=dropout)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape ``(B, 1, seq_len)``.

        Returns:
            Tensor of shape ``(B, 128)``.
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)       # (B, 128, 1)
        x = self.flatten(x)           # (B, 128)
        return x


# ECG-only baseline (Milestone 2)

class ECGBaselineModel(nn.Module):
    """ECG-only regression model that predicts [Systolic, Diastolic] BP.

    Wraps :class:`ECGFeatureExtractor` with a lightweight MLP head.

    Input shape:  ``(B, 1, seq_len)``
    Output shape: ``(B, 2)``
    """

    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()

        self.extractor = ECGFeatureExtractor(dropout=dropout)

        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape ``(B, 1, seq_len)``.

        Returns:
            Tensor of shape ``(B, 2)``: predicted [Systolic, Diastolic].
        """
        embedding: torch.Tensor = self.extractor(x)
        return self.head(embedding)


# Quick sanity check
if __name__ == "__main__":
    dummy_x: torch.Tensor = torch.randn(16, 1, 2500)  # Batch=16, 10s @ 250Hz

    model = ECGBaselineModel()
    out: torch.Tensor = model(dummy_x)

    print(f"Input  shape : {dummy_x.shape}")
    print(f"Output shape : {out.shape}")
    assert out.shape == torch.Size([16, 2]), (
        f"Expected (16, 2), got {out.shape}"
    )

    total_params: int = sum(p.numel() for p in model.parameters())
    trainable: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters   : {total_params:,} total, {trainable:,} trainable")
    print("Shape verification passed.")