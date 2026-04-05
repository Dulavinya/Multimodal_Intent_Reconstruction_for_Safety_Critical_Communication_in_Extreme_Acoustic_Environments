"""
WavJEPA audio encoder.
Encodes raw 16 kHz waveforms with the WavJEPA-NAT backbone.
"""

import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, AutoModel


class WavJEPAEncoder(nn.Module):
    def __init__(self, freeze_encoder: bool = True):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            'labhamlet/wavjepa-nat-base',
            trust_remote_code=True,
        )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            'labhamlet/wavjepa-nat-base',
            trust_remote_code=True,
        )

        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
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
        input_values = inputs['input_values'].to(device)

        # Use grad context based on model training mode and freeze setting
        grad_enabled = torch.is_grad_enabled() and not self.freeze_encoder
        with torch.set_grad_enabled(grad_enabled):
            outputs = self.model(input_values)

        if hasattr(outputs, 'last_hidden_state'):
            hidden = outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 0 and torch.is_tensor(outputs[0]):
            hidden = outputs[0]
        elif torch.is_tensor(outputs):
            hidden = outputs
        else:
            raise TypeError(f"Unsupported WavJEPA output type: {type(outputs)}")

        # WavJEPANat may return [B, C, N, D]; collapse channel views to [B, N, D].
        if hidden.dim() == 4:
            hidden = hidden.mean(dim=1)

        if hidden.dim() != 3:
            raise ValueError(f"Expected hidden state shape [B, N, D], got {tuple(hidden.shape)}")

        return hidden
