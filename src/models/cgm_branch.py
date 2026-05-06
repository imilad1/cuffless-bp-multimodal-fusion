"""
LSTM-based feature extractor for CGM (Continuous Glucose Monitoring) data.

Processes a 30-minute glucose history window (glucose values + rate-of-change)
and outputs a 64-dimensional embedding representing the metabolic/vascular
resistance state, Branch B of the multi-modal late fusion network.
"""

import torch
import torch.nn as nn


class CGMFeatureExtractor(nn.Module):
    """2-layer LSTM that maps a CGM history window to a 64-dim embedding.

    Input shape:  ``(B, seq_len, 2)``, glucose value + derivative per timestep.
    Output shape: ``(B, 64)``, metabolic state embedding.

    Args:
        input_size: Number of features per timestep (default 2: glucose + derivative).
        hidden_size: LSTM hidden dimension.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout between LSTM layers.
    """

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape ``(B, seq_len, 2)``.

        Returns:
            Tensor of shape ``(B, 64)``: last hidden state of the sequence.
        """
        # output: (B, seq_len, hidden_size), (h_n, c_n)
        _, (h_n, _) = self.lstm(x)
        # h_n shape: (num_layers, B, hidden_size); take last layer
        return h_n[-1]


# Quick sanity check
if __name__ == "__main__":
    dummy_x: torch.Tensor = torch.randn(16, 7, 2)  # Batch=16, 7 timesteps, 2 features

    model = CGMFeatureExtractor()
    out: torch.Tensor = model(dummy_x)

    print(f"Input  shape : {dummy_x.shape}")
    print(f"Output shape : {out.shape}")
    assert out.shape == torch.Size([16, 64]), (
        f"Expected (16, 64), got {out.shape}"
    )

    total_params: int = sum(p.numel() for p in model.parameters())
    print(f"Parameters   : {total_params:,}")
    print("CGM branch shape verification passed.")
