"""
Quick inference demo for AthenAI.
Usage:
    python run_inference.py --audio path/to/noisy.wav [--sensors path/to/sensors.npy] [--mode full]
"""

import argparse
import torch
import torchaudio
import numpy as np

from src import AthenAISystem


def main():
    parser = argparse.ArgumentParser(description="AthenAI Inference Demo")
    parser.add_argument('--audio', required=True, help='Path to noisy audio .wav file')
    parser.add_argument('--sensors', default=None, help='Path to .npy sensor window [seq_len, n_sensors]')
    parser.add_argument('--mode', default='full', choices=['base', 'full'])
    parser.add_argument('--snr_db', type=float, default=0.0, help='Estimated SNR in dB')
    parser.add_argument('--n_mc', type=int, default=20, help='MC Dropout passes')
    args = parser.parse_args()

    # Load audio
    waveform, sr = torchaudio.load(args.audio)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)  # to mono
    waveform = waveform  # [1, T]

    # Load sensor window (optional)
    sensor_window = None
    if args.sensors and args.mode == 'full':
        sensor_np = np.load(args.sensors)          # [seq_len, n_sensors]
        sensor_window = torch.tensor(sensor_np, dtype=torch.float32).unsqueeze(0)  # [1, seq_len, n_sensors]

    system = AthenAISystem(mode=args.mode)

    result = system.infer(
        noisy_waveform=waveform,
        sensor_window=sensor_window,
        snr_db=args.snr_db,
        n_mc=args.n_mc,
    )

    print(f"\n--- AthenAI Inference Result ---")
    print(f"  Command    : {result['command']}")
    print(f"  Confidence : {result['confidence']:.4f}")
    print(f"  Uncertainty: {result['uncertainty']:.4f}")
    if result['alert']:
        print(f"\n--- Safety Alert ---\n{result['alert']}")


if __name__ == '__main__':
    main()
