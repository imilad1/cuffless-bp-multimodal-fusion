"""
Multi-modal late fusion network for cuffless blood pressure estimation.

Combines the 128-dim ECG embedding (Cardiac Output proxy) with the 64-dim
CGM embedding (Vascular Resistance proxy) and regresses Systolic/Diastolic BP.
"""

import torch
import torch.nn as nn

try:
    from models.ecg_branch import ECGFeatureExtractor
    from models.cgm_branch import CGMFeatureExtractor
except ModuleNotFoundError:
    from ecg_branch import ECGFeatureExtractor
    from cgm_branch import CGMFeatureExtractor


class MultiModalBPRegressor(nn.Module):
    """Late-fusion network that predicts BP from ECG + CGM inputs.

    Architecture:
      - Branch A: :class:`ECGFeatureExtractor` -> 128-dim
      - Branch B: :class:`CGMFeatureExtractor` -> 64-dim
      - Fusion:   Concatenate (192-dim) -> MLP -> [Systolic, Diastolic]

    Args:
        dropout: Dropout probability used in the fusion head.
    """

    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()

        self.ecg_extractor = ECGFeatureExtractor(dropout=dropout)
        self.cgm_extractor = CGMFeatureExtractor(dropout=dropout)

        self.fusion_head = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(
        self,
        x_ecg: torch.Tensor,
        x_cgm: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x_ecg: Tensor of shape ``(B, 1, seq_len)``, raw ECG waveform.
            x_cgm: Tensor of shape ``(B, 7, 2)``, CGM glucose + derivative.

        Returns:
            Tensor of shape ``(B, 2)``: predicted [Systolic, Diastolic].
        """
        ecg_emb: torch.Tensor = self.ecg_extractor(x_ecg)   # (B, 128)
        cgm_emb: torch.Tensor = self.cgm_extractor(x_cgm)   # (B, 64)
        fused: torch.Tensor = torch.cat([ecg_emb, cgm_emb], dim=1)  # (B, 192)
        return self.fusion_head(fused)


# Quick sanity check
if __name__ == "__main__":
    dummy_ecg: torch.Tensor = torch.randn(16, 1, 2500)   # Batch=16, 10s @ 250Hz
    dummy_cgm: torch.Tensor = torch.randn(16, 7, 2)      # Batch=16, 7 timesteps

    model = MultiModalBPRegressor()
    out: torch.Tensor = model(dummy_ecg, dummy_cgm)

    print(f"ECG input shape : {dummy_ecg.shape}")
    print(f"CGM input shape : {dummy_cgm.shape}")
    print(f"Output shape    : {out.shape}")
    assert out.shape == torch.Size([16, 2]), (
        f"Expected (16, 2), got {out.shape}"
    )

    total_params: int = sum(p.numel() for p in model.parameters())
    trainable: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters      : {total_params:,} total, {trainable:,} trainable")
    print("Fusion model shape verification passed.")