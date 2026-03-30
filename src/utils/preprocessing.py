"""
Audio Preprocessing Utilities
Converts raw waveforms to log-mel spectrograms suitable for the Audio-JEPA encoder.
"""

import torch
import torchaudio
import torchaudio.transforms as T


def waveform_to_mel(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 256,
    f_min: float = 0.0,
    f_max: float = 8000.0,
) -> torch.Tensor:
    """
    Convert a raw audio waveform to a log-mel spectrogram.

    Args:
        waveform:    [B, T] or [T]  — mono waveform at `sample_rate`
        sample_rate: sampling rate of the input waveform
        n_mels:      number of mel filter banks
        n_fft:       FFT window size
        hop_length:  hop length between frames
        f_min:       minimum frequency for mel filterbank
        f_max:       maximum frequency for mel filterbank
    Returns:
        log_mel: [B, 1, n_mels, T_frames]
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # [1, T]

    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
    )
    mel = mel_transform(waveform)           # [B, n_mels, T_frames]
    log_mel = torch.log1p(mel)              # log compression
    return log_mel.unsqueeze(1)             # [B, 1, n_mels, T_frames]
