"""
Audio-JEPA Encoder
Operates on mel-spectrogram patches using a ViT backbone.
Uses a momentum-updated target encoder (EMA) and a context encoder trained via gradient descent.
"""

import copy
import torch
import torch.nn as nn
from transformers import ViTModel


class AudioJEPAEncoder(nn.Module):
    def __init__(self, embed_dim: int = 768, patch_size: int = 16, n_mels: int = 128):
        super().__init__()
        self.patch_embed = nn.Conv2d(1, embed_dim, patch_size, patch_size)
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')

        # EMA (momentum) target encoder — frozen, updated manually
        self.target_encoder = copy.deepcopy(self.vit)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_spec: [B, 1, n_mels, T]  — single-channel log-mel spectrogram
        Returns:
            context_emb: [B, N_patches, embed_dim]
        """
        patches = self.patch_embed(mel_spec)           # [B, D, H', W']
        patches = patches.flatten(2).transpose(1, 2)   # [B, N, D]
        context_emb = self.vit(inputs_embeds=patches).last_hidden_state
        return context_emb                             # [B, N, embed_dim]

    @torch.no_grad()
    def update_target(self, momentum: float = 0.996):
        """EMA update: target ← momentum * target + (1 - momentum) * context"""
        for p_ctx, p_tgt in zip(self.vit.parameters(), self.target_encoder.parameters()):
            p_tgt.data = momentum * p_tgt.data + (1 - momentum) * p_ctx.data
