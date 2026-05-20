"""
Sensor Encoder
Takes a multivariate time-series window (e.g., temperature, pressure, vibration)
and produces a fixed-size embedding using a Transformer encoder with positional encodings.
"""

import torch
import torch.nn as nn


class SensorEncoder(nn.Module):
    def __init__(
        self,
        n_sensors: int = 8,
        seq_len: int = 128,
        embed_dim: int = 256,
        nhead: int = 4,
        n_layers: int = 3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_sensors, embed_dim)
        self.pos_enc = nn.Parameter(torch.zeros(1, seq_len, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=512,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, n_sensors]  — multivariate sensor time-series window
        Returns:
            [B, seq_len, embed_dim]  — dense unpooled sensor embedding sequence
        """
        x = self.input_proj(x) + self.pos_enc   # [B, seq_len, embed_dim]
        return self.transformer(x)               # [B, seq_len, embed_dim]
