"""
Command Classification Head with Confidence Estimation
Predicts the safety command label from (fused or speech-only) embedding.
- Temperature scaling for confidence calibration
- Monte Carlo Dropout for inference-time uncertainty estimation
"""

import torch
import torch.nn as nn


class CommandClassifier(nn.Module):
    def __init__(self, input_dim: int = 512, n_commands: int = 20, dropout: float = 0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_commands),
        )
        self.temperature = nn.Parameter(torch.ones(1))  # learnable temperature

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, input_dim]
        Returns:
            pred_cmd:   [B]        — predicted command index
            confidence: [B]        — max calibrated probability
            probs:      [B, n_commands]
        """
        logits = self.fc(x)                        # [B, n_commands]
        calibrated = logits / self.temperature
        probs = torch.softmax(calibrated, dim=-1)
        confidence = probs.max(dim=-1).values      # [B]
        pred_cmd = probs.argmax(dim=-1)            # [B]
        return pred_cmd, confidence, probs


def mc_dropout_inference(
    model: CommandClassifier,
    x: torch.Tensor,
    n_passes: int = 20,
):
    """
    Monte Carlo Dropout inference: runs n_passes forward passes with dropout active,
    then aggregates to produce mean probabilities and predictive entropy.

    Args:
        model:    CommandClassifier instance
        x:        [B, input_dim]
        n_passes: number of stochastic forward passes
    Returns:
        pred_cmd:   [B]
        confidence: [B]
        uncertainty:[B]  — predictive entropy (higher = more uncertain)
    """
    model.train()  # keep dropout active during inference
    preds = torch.stack([model(x)[2] for _ in range(n_passes)])  # [n_passes, B, C]
    mean_probs = preds.mean(0)                                     # [B, C]
    uncertainty = -(mean_probs * mean_probs.log()).sum(-1)         # [B] — entropy
    return mean_probs.argmax(-1), mean_probs.max(-1).values, uncertainty
