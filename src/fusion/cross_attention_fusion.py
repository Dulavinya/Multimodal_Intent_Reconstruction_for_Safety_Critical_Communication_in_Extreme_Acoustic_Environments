"""
Cross-Attention Fusion Module
Speech embedding acts as query; sensor embedding provides keys and values.
Allows the model to dynamically weight sensor context relative to what was heard.
"""

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        speech_dim: int = 768,
        sensor_dim: int = 256,
        fused_dim: int = 512,
        nhead: int = 8,
    ):
        super().__init__()
        self.q_proj = nn.Linear(speech_dim, fused_dim)
        self.k_proj = nn.Linear(sensor_dim, fused_dim)
        self.v_proj = nn.Linear(sensor_dim, fused_dim)
        self.cross_attn = nn.MultiheadAttention(fused_dim, nhead, batch_first=True)
        self.norm = nn.LayerNorm(fused_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, speech_emb: torch.Tensor, sensor_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            speech_emb: [B, N_patches, speech_dim]
            sensor_emb: [B, sensor_dim]
        Returns:
            out: [B, fused_dim]
        """
        Q = self.q_proj(speech_emb)                    # [B, N, fused_dim]
        K = self.k_proj(sensor_emb.unsqueeze(1))       # [B, 1, fused_dim]
        V = self.v_proj(sensor_emb.unsqueeze(1))       # [B, 1, fused_dim]

        fused, _ = self.cross_attn(Q, K, V)            # [B, N, fused_dim]
        fused = self.norm(fused + Q)                   # residual + layer norm
        out = self.pool(fused.transpose(1, 2)).squeeze(-1)  # [B, fused_dim]
        return out
