"""
Clean Audio Synthesizer (AudioGen)
AudioGen is NOT used to decode noisy embeddings directly.
It receives the predicted command string as a text prompt and synthesizes
a clean, canonical audio waveform — the verified output for human operators.
"""

import torch
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write


class CleanAudioSynthesizer:
    def __init__(self, model_name: str = 'facebook/audiogen-medium', duration: float = 3.0):
        self.model = AudioGen.get_pretrained(model_name)
        self.model.set_generation_params(duration=duration)

    def synthesize(self, predicted_command: str, confidence: float) -> torch.Tensor:
        """
        Synthesize a clean audio waveform conditioned on the predicted command string.

        Args:
            predicted_command: e.g. 'Stop the machine immediately'
            confidence:        0.0–1.0 scalar from classifier
        Returns:
            waveform: [1, samples] at 16 kHz
        """
        if confidence < 0.4:
            prompt = f'unclear safety alert: {predicted_command}'
        else:
            prompt = f'clear industrial safety command: {predicted_command}'

        wav = self.model.generate([prompt])   # [1, 1, samples]
        return wav.squeeze(0)                  # [1, samples]

    def save(self, waveform: torch.Tensor, path: str, sample_rate: int = 16000):
        """Save synthesized waveform to disk."""
        audio_write(path, waveform.cpu(), sample_rate, strategy='loudness')
