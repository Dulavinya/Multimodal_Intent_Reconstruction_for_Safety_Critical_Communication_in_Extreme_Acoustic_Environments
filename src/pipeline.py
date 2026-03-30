"""
AthenAI — Full System Integration
Complete inference pipeline: noisy audio → command classification → clean audio + safety alert.

Two modes:
  'base' — audio-only (no sensor fusion)
  'full' — multimodal (audio + sensor time-series via cross-attention fusion)
"""

import torch
from typing import Optional, Dict, Any

from .encoders import AudioJEPAEncoder, SensorEncoder
from .fusion import CrossAttentionFusion
from .classification import CommandClassifier, mc_dropout_inference
from .synthesis import CleanAudioSynthesizer
from .alert import generate_alert
from .utils import waveform_to_mel, COMMAND_VOCAB


class AthenAISystem:
    def __init__(self, mode: str = 'full'):
        """
        Args:
            mode: 'base' (audio-only) or 'full' (multimodal with sensor fusion)
        """
        assert mode in ('base', 'full'), "mode must be 'base' or 'full'"
        self.mode = mode

        self.audio_encoder = AudioJEPAEncoder(embed_dim=768)
        self.classifier = CommandClassifier(
            input_dim=768 if mode == 'base' else 512
        )
        self.synthesizer = CleanAudioSynthesizer()

        if mode == 'full':
            self.sensor_encoder = SensorEncoder(n_sensors=8)
            self.fusion = CrossAttentionFusion()

    def infer(
        self,
        noisy_waveform: torch.Tensor,
        sensor_window: Optional[torch.Tensor] = None,
        snr_db: float = 0.0,
        sensor_state: str = "nominal",
        llm_client=None,
        n_mc: int = 20,
    ) -> Dict[str, Any]:
        """
        Run full inference pipeline.

        Args:
            noisy_waveform: [B, T]  — raw noisy audio at 16 kHz
            sensor_window:  [B, seq_len, n_sensors]  — required when mode='full'
            snr_db:         estimated signal-to-noise ratio in dB
            sensor_state:   human-readable sensor summary for the LLM alert
            llm_client:     LLM client with .complete(prompt, max_tokens) interface
            n_mc:           number of Monte Carlo Dropout forward passes
        Returns:
            dict with keys: command, confidence, uncertainty, clean_audio, alert
        """
        # ── Preprocessing ────────────────────────────────────────────────────
        mel = waveform_to_mel(noisy_waveform)          # [B, 1, n_mels, T_frames]

        # ── Phase 0: Encoding ─────────────────────────────────────────────────
        speech_emb = self.audio_encoder(mel)           # [B, N_patches, 768]

        if self.mode == 'full' and sensor_window is not None:
            sensor_emb = self.sensor_encoder(sensor_window)        # [B, 256]
            inp = self.fusion(speech_emb, sensor_emb)              # [B, 512]
        else:
            inp = speech_emb.mean(1)                               # [B, 768]

        # ── Phase 1: Command Classification ──────────────────────────────────
        cmd_idx, confidence, uncertainty = mc_dropout_inference(
            self.classifier, inp, n_passes=n_mc
        )
        command_str = COMMAND_VOCAB[cmd_idx.item()]

        # ── Phase 2: Clean Audio Generation ──────────────────────────────────
        clean_wav = self.synthesizer.synthesize(command_str, confidence.item())

        # ── Phase 3: LLM Safety Alert ─────────────────────────────────────────
        alert = None
        if llm_client is not None:
            alert = generate_alert(
                command=command_str,
                confidence=confidence.item(),
                snr_db=snr_db,
                sensor_state=sensor_state,
                llm_client=llm_client,
            )

        return {
            'command':     command_str,
            'confidence':  confidence.item(),
            'uncertainty': uncertainty.item(),
            'clean_audio': clean_wav,
            'alert':       alert,
        }
