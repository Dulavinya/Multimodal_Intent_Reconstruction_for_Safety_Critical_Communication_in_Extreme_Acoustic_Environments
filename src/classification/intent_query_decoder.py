"""
Intent Query Decoder (IQD) — Multimodal Dual-Stream version for AthenAI.

Replaces the previous concatenation-aware decoder with a parallel architecture:
1. Intent Queries independently attend to Audio and Sensors in parallel streams.
2. A Multi-Modal Transformer block fuses the resulting hypotheses.
3. Iterative refinement allows the queries to "reason" across modalities over multiple layers.

This prevents high-quality audio features from "drowning out" the secondary sensor signals,
allowing the model to explicitly weight physical context against acoustic evidence.
"""

import torch
import torch.nn as nn


class IntentQueryDecoder(nn.Module):
    """
    Multimodal Dual-Stream Intent Decoder using learned class queries and Transformer fusion.

    Architecture (per layer):
        Q_a = MultiheadAttention(Q, audio, audio)
        Q_s = MultiheadAttention(Q, sensor, sensor)  (if sensor provided)
        H   = TransformerEncoderLayer(Concat(Q_a, Q_s))
        Q   = Norm(Refine(H) + Q)
    """

    def __init__(
        self,
        n_commands: int = 20,
        query_dim: int = 768,
        n_heads: int = 8,
        dropout: float = 0.3,
        n_layers: int = 2,
    ):
        super().__init__()
        self.n_commands = n_commands
        self.query_dim = query_dim

        # 20 learnable query vectors — one per safety command
        self.class_queries = nn.Embedding(n_commands, query_dim)

        # ── Dual-Stream Attention Blocks ─────────────────────────────────────
        self.audio_attns = nn.ModuleList([
            nn.MultiheadAttention(query_dim, n_heads, dropout=0.1, batch_first=True)
            for _ in range(n_layers)
        ])
        
        # Sensor attention (only used if sensor_seq provided in forward)
        self.sensor_attns = nn.ModuleList([
            nn.MultiheadAttention(query_dim, n_heads, dropout=0.1, batch_first=True)
            for _ in range(n_layers)
        ])

        # ── Transformer Fusion Blocks ─────────────────────────────────────────
        # This block lets the queries communicate and weigh the two modalities
        self.fusion_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=query_dim,
                nhead=n_heads,
                dim_feedforward=query_dim * 2,
                dropout=0.1,
                batch_first=True,
                activation="gelu"
            )
            for _ in range(n_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(query_dim)
            for _ in range(n_layers)
        ])

        # Final scoring head
        self.scoring_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(query_dim, 1),
        )

        self.temperature = nn.Parameter(torch.ones(1))
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.class_queries.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.scoring_head[1].weight)
        nn.init.zeros_(self.scoring_head[1].bias)

    def _run_layers(self, audio_seq: torch.Tensor, sensor_seq: torch.Tensor | None = None) -> torch.Tensor:
        B = audio_seq.size(0)
        query_ids = torch.arange(self.n_commands, device=audio_seq.device)
        Q = self.class_queries(query_ids).unsqueeze(0).expand(B, -1, -1)  # [B, C, D]

        for i in range(len(self.audio_attns)):
            # 1. Parallel Attention
            Q_a, _ = self.audio_attns[i](Q, audio_seq, audio_seq)  # [B, C, D]
            
            if sensor_seq is not None:
                Q_s, _ = self.sensor_attns[i](Q, sensor_seq, sensor_seq) # [B, C, D]
                
                # 2. Transformer Fusion
                # Concatenate along sequence dim -> [B, 2*C, D]
                # Then let the Transformer layer perform cross-modal reasoning
                combined = torch.cat([Q_a, Q_s], dim=1) 
                fused = self.fusion_layers[i](combined) # [B, 2*C, D]
                
                # Pool back to C queries (e.g., average the audio and sensor refined queries)
                Q_new = (fused[:, :self.n_commands, :] + fused[:, self.n_commands:, :]) / 2
            else:
                Q_new = Q_a

            # 3. Residual & Norm
            Q = self.norms[i](Q_new + Q)

        return Q

    def logits(self, audio_seq: torch.Tensor, sensor_seq: torch.Tensor | None = None) -> torch.Tensor:
        Q = self._run_layers(audio_seq, sensor_seq)
        raw = self.scoring_head(Q).squeeze(-1)
        # Numerical stability: clamp temperature to prevent division by near-zero
        temp = torch.clamp(self.temperature, min=0.01)
        return raw / temp

    def forward(self, audio_seq: torch.Tensor, sensor_seq: torch.Tensor | None = None):
        calibrated = self.logits(audio_seq, sensor_seq)
        probs = torch.softmax(calibrated, dim=-1)
        confidence, pred_cmd = probs.max(dim=-1)
        return pred_cmd, confidence, probs


def mc_dropout_inference_iqd(
    model: IntentQueryDecoder,
    audio_seq: torch.Tensor,
    sensor_seq: torch.Tensor | None = None,
    n_passes: int = 20,
):
    """Monte Carlo Dropout inference supporting multimodal dual-stream."""
    model.train()
    preds = torch.stack(
        [model(audio_seq, sensor_seq)[2] for _ in range(n_passes)]
    )
    mean_probs = preds.mean(0)
    eps = 1e-9
    uncertainty = -(mean_probs * (mean_probs + eps).log()).sum(-1)
    return mean_probs.argmax(-1), mean_probs.max(-1).values, uncertainty
