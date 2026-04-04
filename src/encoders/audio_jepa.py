"""
WavJEPA audio encoder.
Encodes raw 16 kHz waveforms with the WavJEPA-NAT backbone.
"""

import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, AutoModel


class WavJEPAEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained('labhamlet/wavjepa-nat-base')
        self.feature_extractor = AutoFeatureExtractor.from_pretrained('labhamlet/wavjepa-nat-base')

        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.eval()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [B, T] raw 16 kHz audio waveforms
        Returns:
            last_hidden_state: [B, N, 768]
        """
        if waveform.dim() != 2:
            raise ValueError(f"waveform must have shape [B, T], got {tuple(waveform.shape)}")

        device = waveform.device
        inputs = self.feature_extractor(
            [sample.detach().cpu().numpy() for sample in waveform],
            sampling_rate=16000,
            return_tensors='pt',
            padding=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.last_hidden_state
