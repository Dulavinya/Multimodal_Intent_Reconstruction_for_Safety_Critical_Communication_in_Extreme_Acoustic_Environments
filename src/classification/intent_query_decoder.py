"""
Intent Query Decoder (IQD) — DETR-style classification head for AthenAI.

Replaces mean-pooling + FC classifier with a set of 20 learnable class query
vectors that cross-attend over the full WavJEPA temporal sequence [B, N, 768].

Each query learns to selectively hunt for time-patches most evidential for its
specific safety command, providing noise-selective attention that mean-pooling
fundamentally cannot perform.

Architecture:
    class_queries [1, C, D]  →  Q
    wavjepa_seq   [B, N, D]  →  K, V
    cross_attn    [B, C, D]
    LayerNorm + Dropout
    per_class_head [B, C, 1] → squeeze → [B, C] logits
    temperature scaling → softmax → pred
"""

import torch
import torch.nn as nn


class IntentQueryDecoder(nn.Module):
    """
    DETR-style intent decoder using learned class queries.

    Args:
        n_commands: number of safety command classes (default 20)
        query_dim:  embedding dimension, must match WavJEPA output dim (768)
        n_heads:    number of attention heads (query_dim must be divisible)
        dropout:    dropout rate applied to the per-class scoring head
    """

    def __init__(
        self,
        n_commands: int = 20,
        query_dim: int = 768,
        n_heads: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()

        # 20 learnable query vectors — one per safety command
        self.class_queries = nn.Embedding(n_commands, query_dim)

        # Cross-attention: queries attend over the WavJEPA sequence
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=n_heads,
            dropout=0.1,           # internal attn dropout for regularisation
            batch_first=True,
        )

        self.norm = nn.LayerNorm(query_dim)

        # Per-class scoring: squeeze each attended query to a scalar logit
        self.scoring_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(query_dim, 1),
        )

        # Learnable temperature for calibration (matches CommandClassifier API)
        self.temperature = nn.Parameter(torch.ones(1))

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise class queries with small normal noise for symmetry breaking."""
        nn.init.normal_(self.class_queries.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.scoring_head[1].weight)
        nn.init.zeros_(self.scoring_head[1].bias)

    def forward(self, audio_seq: torch.Tensor):
        """
        Args:
            audio_seq: [B, N, query_dim]  — full WavJEPA temporal sequence

        Returns:
            pred_cmd:   [B]               — predicted command index
            confidence: [B]               — max calibrated probability
            probs:      [B, n_commands]
        """
        B = audio_seq.size(0)

        # Expand class queries to batch: [1, C, D] → [B, C, D]
        query_ids = torch.arange(
            self.class_queries.num_embeddings, device=audio_seq.device
        )
        Q = self.class_queries(query_ids).unsqueeze(0).expand(B, -1, -1)  # [B, C, D]

        # Cross-attention: Q=class queries, K=V=WavJEPA patches
        attended, _ = self.cross_attn(Q, audio_seq, audio_seq)  # [B, C, D]
        attended = self.norm(attended + Q)                        # residual + LN

        # Score each query independently: [B, C, D] → [B, C, 1] → [B, C]
        logits = self.scoring_head(attended).squeeze(-1)          # [B, C]

        calibrated = logits / self.temperature
        probs = torch.softmax(calibrated, dim=-1)
        confidence = probs.max(dim=-1).values                     # [B]
        pred_cmd = probs.argmax(dim=-1)                           # [B]

        return pred_cmd, confidence, probs


def mc_dropout_inference_iqd(
    model: IntentQueryDecoder,
    audio_seq: torch.Tensor,
    n_passes: int = 20,
):
    """
    Monte Carlo Dropout inference for the Intent Query Decoder.

    Runs n_passes forward passes with dropout active (model.train()),
    then aggregates mean probabilities and computes predictive entropy.

    Args:
        model:     IntentQueryDecoder instance
        audio_seq: [B, N, query_dim] — full WavJEPA sequence (NOT mean-pooled)
        n_passes:  number of stochastic forward passes

    Returns:
        pred_cmd:   [B]
        confidence: [B]
        uncertainty:[B]  — predictive entropy (higher = more uncertain)
    """
    model.train()  # keep dropout active
    preds = torch.stack(
        [model(audio_seq)[2] for _ in range(n_passes)]
    )                                                   # [n_passes, B, C]
    mean_probs = preds.mean(0)                          # [B, C]
    eps = 1e-9
    uncertainty = -(mean_probs * (mean_probs + eps).log()).sum(-1)  # [B]
    return mean_probs.argmax(-1), mean_probs.max(-1).values, uncertainty
