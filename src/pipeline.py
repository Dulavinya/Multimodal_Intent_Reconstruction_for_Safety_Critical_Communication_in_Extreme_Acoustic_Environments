"""
AthenAI — Full System Integration
Complete inference pipeline: noisy audio → command classification → clean audio + safety alert.

Two modes — both use the Intent Query Decoder (IQD):
  'base' — audio-only
             WavJEPA[B, N, 768] → IQD → command
  'full' — multimodal (audio + sensor time-series)
             WavJEPA[B, N, 768] + SensorSeq[B, T, 768] → cat[B, N+T, 768] → IQD → command

The dense sensor embedding sequence is projected to the audio dimension and
concatenated. IQD class queries then attend over the joint dense sequence,
gaining access to both acoustic and physical-sensor evidence simultaneously.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from .encoders import WavJEPAEncoder, SensorEncoder
from .classification import IntentQueryDecoder, mc_dropout_inference_iqd
from .synthesis import CleanAudioSynthesizer
from .alert import generate_alert
from .utils import COMMAND_VOCAB

AUDIO_DIM  = 768
SENSOR_DIM = 256
N_SENSORS  = 8


class AthenAISystem:
    def __init__(self, mode: str = 'full', n_layers: int = 2):
        """
        Args:
            mode:     'base' (audio-only) or 'full' (multimodal with sensor)
            n_layers: number of stacked IQD cross-attention layers
        """
        assert mode in ('base', 'full'), "mode must be 'base' or 'full'"
        self.mode = mode

        self.audio_encoder = WavJEPAEncoder()

        if mode == 'full':
            self.sensor_encoder = SensorEncoder(n_sensors=N_SENSORS)
            self.sensor_proj    = nn.Linear(SENSOR_DIM, AUDIO_DIM)

        self.decoder = IntentQueryDecoder(
            n_commands=len(COMMAND_VOCAB),
            query_dim=AUDIO_DIM,
            n_heads=8,
            dropout=0.3,
            n_layers=n_layers,
        )
        self.synthesizer = CleanAudioSynthesizer()

    def _build_sequences(
        self,
        noisy_waveform: torch.Tensor,
        sensor_window: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Encode inputs and build the individual modal sequences for IQD.

        Returns:
            speech_seq: [B, N, 768]
            sensor_seq: [B, T, 768] (or None)
        """
        speech_seq = self.audio_encoder(noisy_waveform)           # [B, N, 768]
        sensor_seq = None

        if self.mode == 'full' and sensor_window is not None:
            # Dense sequential embeddings
            s_emb = self.sensor_encoder(sensor_window)             # [B, T, 256]
            sensor_seq = self.sensor_proj(s_emb)                   # [B, T, 768]

        return speech_seq, sensor_seq

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
            noisy_waveform: [B, T]               — raw noisy audio at 16 kHz
            sensor_window:  [B, seq_len, n_sensors] — required when mode='full'
            snr_db:         estimated signal-to-noise ratio in dB
            sensor_state:   human-readable sensor summary for the LLM alert
            llm_client:     LLM client with .complete(prompt, max_tokens) interface
            n_mc:           number of Monte Carlo Dropout forward passes

        Returns:
            dict with keys: command, confidence, uncertainty, clean_audio, alert
        """
        # ── Phase 0: Build multimodal sequences ──────────────────────────────
        speech_seq, sensor_seq = self._build_sequences(noisy_waveform, sensor_window)

        # ── Phase 1: Command Classification via IQD ───────────────────────────
        cmd_idx, confidence, uncertainty = mc_dropout_inference_iqd(
            self.decoder, speech_seq, sensor_seq, n_passes=n_mc
        )
        command_str = COMMAND_VOCAB[cmd_idx.item()]

        # ── Phase 2: Clean Audio Generation ───────────────────────────────────
        clean_wav = self.synthesizer.synthesize(command_str, confidence.item())

        # ── Phase 3: LLM Safety Alert ──────────────────────────────────────────
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
