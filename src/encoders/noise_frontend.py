import torch
import torch.nn as nn


class LearnableNoiseFrontend(nn.Module):
    """
    A lightweight 1D convolutional frontend that learns to suppress
    industrial noise before the WavJEPA encoder sees the waveform.

    Key design decisions:
    - Residual connection: starts as identity, learns small corrections
    - Small kernel sizes: captures local noise patterns efficiently
    - No aggressive compression: output is same length as input
    - Dropout: prevents overfitting on small synthetic datasets
    """

    def __init__(self, channels: int = 16, dropout: float = 0.1):
        super().__init__()

        self.conv_block = nn.Sequential(
            # Layer 1: detect noise patterns
            nn.Conv1d(
                in_channels=1,
                out_channels=channels,
                kernel_size=64,
                stride=1,
                padding=32,    # keeps same length
            ),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 2: suppress what layer 1 detected
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=32,
                stride=1,
                padding=16,
            ),
            nn.BatchNorm1d(channels),
            nn.ReLU(),

            # Layer 3: project back to single channel (waveform shape)
            nn.Conv1d(
                in_channels=channels,
                out_channels=1,
                kernel_size=16,
                stride=1,
                padding=8,
            ),
        )

        # Learnable gate: how much of the frontend output to mix in
        # Starts at 0 → pure residual (safe initialisation)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: raw waveform [B, T]
        Returns:
            cleaned waveform [B, T] — same shape
        """
        # Add channel dim for Conv1d: [B, T] → [B, 1, T]
        x_in = x.unsqueeze(1)

        # Frontend correction
        correction = self.conv_block(x_in)  # [B, 1, T+small_offset]

        # Trim to exactly match input length (padding can add 1-2 samples)
        correction = correction[:, :, :x_in.size(2)]

        # Gated residual: output = input + gate * correction
        # gate starts at 0 so training begins from identity
        out = x_in + torch.sigmoid(self.gate) * correction

        # Remove channel dim: [B, 1, T] → [B, T]
        return out.squeeze(1)