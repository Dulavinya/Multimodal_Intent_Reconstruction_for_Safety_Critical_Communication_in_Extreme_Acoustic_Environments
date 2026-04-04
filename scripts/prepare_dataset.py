"""
Data Preparation Script for AthenAI Project
============================================

Generates mixed speech+noise audio samples with synchronized sensor data.
- Resamples speech and noise to 16kHz mono
- Mixes at multiple SNR levels: [20, 10, 0, -5, -10, -20] dB
- Extracts random 128-timestep windows from SKAB sensor CSVs
- Maps speech commands to AthenAI vocabulary
- Splits into train/val/test stratified by command label
- Generates metadata.csv with file locations and split information
"""

import os
import sys
import argparse
import json
import uuid
from pathlib import Path
from collections import defaultdict
import random

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

# Load vocab directly to avoid pipeline imports
COMMAND_VOCAB = [
    "stop the machine",
    "emergency shutdown",
    "evacuate immediately",
    "reduce speed",
    "increase pressure",
    "decrease pressure",
    "open valve",
    "close valve",
    "activate safety lock",
    "release safety lock",
    "call supervisor",
    "check sensor",
    "restart system",
    "halt conveyor",
    "start conveyor",
    "fire alarm",
    "chemical leak alert",
    "electrical hazard",
    "all clear",
    "unknown command",
]

COMMAND_TO_IDX = {cmd: idx for idx, cmd in enumerate(COMMAND_VOCAB)}


# ============================================================================
# Vocabulary Mapping: Speech Commands → AthenAI Safety Commands
# ============================================================================

SPEECH_TO_SAFETY_MAPPING = {
    # Numeric commands
    "zero": "all clear",
    "one": "activate safety lock",
    "two": "increase pressure",
    "three": "reduce speed",
    "four": "check sensor",
    "five": "open valve",
    "six": "close valve",
    "seven": "start conveyor",
    "eight": "halt conveyor",
    "nine": "restart system",
    
    # Action commands
    "stop": "stop the machine",
    "go": "start conveyor",
    "forward": "increase pressure",
    "backward": "reduce speed",
    "up": "increase pressure",
    "down": "decrease pressure",
    "left": "open valve",
    "right": "close valve",
    
    # Status commands
    "yes": "all clear",
    "no": "emergency shutdown",
    "on": "activate safety lock",
    "off": "release safety lock",
    
    # Request commands
    "follow": "call supervisor",
    "learn": "check sensor",
    
    # Person/object commands
    "bed": "restart system",
    "bird": "chemical leak alert",
    "cat": "electrical hazard",
    "dog": "electrical hazard",
    "happy": "all clear",
    "house": "all clear",
    "marvin": "call supervisor",
    "sheila": "call supervisor",
    "tree": "evacuate immediately",
    "wow": "emergency shutdown",
    
    # Sound/visual
    "visual": "all clear",
    "_background_noise_": "unknown command",
}


class DataPreparator:
    """Prepares mixed audio + sensor data for AthenAI training."""
    
    def __init__(self, project_root, n_samples=1000, seed=42):
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data"
        self.speech_dir = self.data_dir / "speech_commands"
        self.noise_dir = self.data_dir / "noise"
        self.sensors_dir = self.data_dir / "sensors"
        self.mixed_dir = self.data_dir / "mixed"
        
        self.n_samples = n_samples
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # SNR levels in dB
        self.snr_levels = [20, 10, 0, -5, -10, -20]
        
        # Audio parameters
        self.target_sr = 16000
        self.resampler = None
        self.current_sr = None
        
        # Statistics
        self.stats = {
            "total_samples": 0,
            "per_label": defaultdict(int),
            "per_snr": defaultdict(int),
            "per_split": defaultdict(int),
        }
        
        self.metadata = []
        
    def validate_directories(self):
        """Validate that required directories exist."""
        if not self.speech_dir.exists():
            raise FileNotFoundError(f"Speech commands dir not found: {self.speech_dir}")
        if not self.noise_dir.exists():
            raise FileNotFoundError(f"Noise dir not found: {self.noise_dir}")
        if not self.sensors_dir.exists():
            raise FileNotFoundError(f"Sensors dir not found: {self.sensors_dir}")
        
        self.mixed_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ All required directories found")
        
    def load_speech_files(self):
        """Load all speech command files organized by label."""
        speech_files = defaultdict(list)
        
        for label_dir in self.speech_dir.iterdir():
            if label_dir.is_dir():
                wav_files = list(label_dir.glob("*.wav"))
                speech_files[label_dir.name] = wav_files
        
        print(f"✓ Found {len(speech_files)} speech command labels:")
        for label, files in sorted(speech_files.items()):
            print(f"    {label}: {len(files)} files")
        
        return speech_files
    
    def load_noise_files(self):
        """Load all noise files."""
        noise_files = list(self.noise_dir.glob("*.wav"))
        print(f"✓ Found {len(noise_files)} noise files")
        return noise_files
    
    def load_sensor_files(self):
        """Load all SKAB sensor CSV files."""
        sensor_files = list(self.sensors_dir.glob("*.csv"))
        print(f"✓ Found {len(sensor_files)} sensor CSV files")
        return sensor_files
    
    def load_and_resample_audio(self, audio_path):
        """Load audio and resample to 16kHz mono."""
        waveform, sr = torchaudio.load(str(audio_path))
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.target_sr:
            resampler = T.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
        
        return waveform.squeeze(0).numpy(), self.target_sr
    
    def mix_speech_noise(self, speech, noise, snr_db):
        """Mix speech and noise at specified SNR level.
        
        SNR formula: noise_scaled = noise * (rms(speech) / (rms(noise) * 10^(snr/20)))
        """
        # Ensure same length by repeating noise if needed
        if len(noise) < len(speech):
            n_repeats = (len(speech) // len(noise)) + 1
            noise = np.tile(noise, n_repeats)[:len(speech)]
        elif len(noise) > len(speech):
            noise = noise[:len(speech)]
        
        # Calculate RMS values
        rms_speech = np.sqrt(np.mean(speech ** 2))
        rms_noise = np.sqrt(np.mean(noise ** 2))
        
        # Avoid division by zero
        if rms_noise < 1e-10:
            return speech
        
        # Scale noise according to SNR formula
        snr_linear = 10 ** (snr_db / 20)
        noise_scaled = noise * (rms_speech / (rms_noise * snr_linear))
        
        # Mix
        mixed = speech + noise_scaled
        
        # Soft clipping if needed
        max_val = np.max(np.abs(mixed))
        if max_val > 1.0:
            mixed = mixed / max_val * 0.95
        
        return mixed
    
    def load_sensor_window(self, csv_path, window_size=128, n_features=8):
        """Load random contiguous window from SKAB sensor CSV.
        
        - Read CSV file (semicolon-separated)
        - Extract numeric columns (up to n_features)
        - Normalize each column to [0, 1]
        - Return random contiguous window
        """
        df = pd.read_csv(csv_path, sep=";")
        
        # Get numeric columns (exclude timestamp/index columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = numeric_cols[:n_features]  # Use up to 8 features
        
        if len(numeric_cols) < n_features:
            # Pad with zeros if fewer than 8 features
            n_pad = n_features - len(numeric_cols)
            for i in range(n_pad):
                numeric_cols.append(None)
        
        # Extract and normalize data
        data = []
        for col in numeric_cols:
            if col is not None and col in df.columns:
                col_data = df[col].values.astype(float)
                # Normalize to [0, 1]
                col_min = np.min(col_data)
                col_max = np.max(col_data)
                if col_max > col_min:
                    col_data = (col_data - col_min) / (col_max - col_min)
                else:
                    col_data = np.zeros_like(col_data)
                data.append(col_data)
            else:
                # Pad with zeros if column missing
                data.append(np.zeros(len(df)))
        
        # Stack into [n_timesteps, n_features]
        data = np.stack(data, axis=1)  # [n_timesteps, n_features]
        
        if len(data) < window_size:
            # Pad if CSV is shorter than window_size
            pad_size = window_size - len(data)
            data = np.vstack([data, np.zeros((pad_size, n_features))])
            return data[:window_size]
        
        # Random contiguous window
        start_idx = np.random.randint(0, len(data) - window_size + 1)
        window = data[start_idx:start_idx + window_size]
        
        return window
    
    def get_command_idx(self, speech_label):
        """Map speech command label to AthenAI vocabulary index."""
        # Get safety command from mapping
        safety_command = SPEECH_TO_SAFETY_MAPPING.get(speech_label, "unknown command")
        
        # Get index from vocab
        command_idx = COMMAND_TO_IDX.get(safety_command, COMMAND_TO_IDX["unknown command"])
        
        return safety_command, command_idx
    
    def generate_samples(self):
        """Generate mixed audio + sensor sample pairs."""
        print("\n" + "="*70)
        print("LOADING DATASETS")
        print("="*70)
        
        speech_files = self.load_speech_files()
        noise_files = self.load_noise_files()
        sensor_files = self.load_sensor_files()
        
        if not noise_files:
            raise ValueError("No noise files found")
        if not sensor_files:
            raise ValueError("No sensor files found")
        
        # Flatten speech files for sampling
        speech_list = []
        for label, files in speech_files.items():
            speech_list.extend([(label, f) for f in files])
        
        if not speech_list:
            raise ValueError("No speech files found")
        
        print(f"\n✓ Loaded datasets:")
        print(f"    Speech samples: {len(speech_list)}")
        print(f"    Noise files: {len(noise_files)}")
        print(f"    Sensor files: {len(sensor_files)}")
        
        print("\n" + "="*70)
        print(f"GENERATING {self.n_samples} MIXED SAMPLES")
        print("="*70)
        
        # Generate samples
        samples_per_snr = self.n_samples // len(self.snr_levels)
        
        sample_id = 0
        for snr_idx, snr_db in enumerate(self.snr_levels):
            n_this_snr = samples_per_snr
            if snr_idx == len(self.snr_levels) - 1:
                # Ensure we reach exactly n_samples in the last SNR level
                n_this_snr = self.n_samples - sample_id
            
            print(f"\nSNR {snr_db:+3d} dB: generating {n_this_snr} samples...")
            
            for _ in tqdm(range(n_this_snr), desc=f"SNR {snr_db:+3d}"):
                # Sample speech, noise, sensor
                speech_label, speech_path = random.choice(speech_list)
                noise_path = random.choice(noise_files)
                sensor_path = random.choice(sensor_files)
                
                # Load and prepare audio
                try:
                    speech, _ = self.load_and_resample_audio(speech_path)
                    noise, _ = self.load_and_resample_audio(noise_path)
                except Exception as e:
                    print(f"Error loading audio: {e}")
                    continue
                
                # Mix at current SNR
                mixed = self.mix_speech_noise(speech, noise, snr_db)
                
                # Load sensor window
                try:
                    sensor_window = self.load_sensor_window(sensor_path)
                except Exception as e:
                    print(f"Error loading sensor: {e}")
                    continue
                
                # Get command mapping
                safety_command, command_idx = self.get_command_idx(speech_label)
                
                # Generate unique ID
                sample_id_str = str(uuid.uuid4())[:8]
                audio_filename = f"{sample_id_str}_audio.wav"
                sensor_filename = f"{sample_id_str}_sensor.npy"
                
                # Save audio
                audio_path = self.mixed_dir / audio_filename
                torchaudio.save(
                    str(audio_path),
                    torch.from_numpy(mixed).unsqueeze(0),
                    self.target_sr
                )
                
                # Save sensor
                sensor_path_npy = self.mixed_dir / sensor_filename
                np.save(str(sensor_path_npy), sensor_window)
                
                # Record metadata
                self.metadata.append({
                    "audio_file": audio_filename,
                    "sensor_file": sensor_filename,
                    "command_label": safety_command,
                    "command_idx": command_idx,
                    "snr_db": snr_db,
                    "split": None,  # Will be assigned later
                })
                
                # Update statistics
                self.stats["total_samples"] += 1
                self.stats["per_label"][safety_command] += 1
                self.stats["per_snr"][snr_db] += 1
                
                sample_id += 1
        
        return self.metadata
    
    def split_data(self, train_ratio=0.8, val_ratio=0.1):
        """Split data into train/val/test stratified by command label."""
        print("\n" + "="*70)
        print("STRATIFYING TRAIN/VAL/TEST SPLIT")
        print("="*70)
        
        # Group by label
        label_groups = defaultdict(list)
        for idx, sample in enumerate(self.metadata):
            label = sample["command_label"]
            label_groups[label].append(idx)
        
        # Assign splits stratified by label
        for label, indices in label_groups.items():
            random.shuffle(indices)
            
            n_train = max(1, int(len(indices) * train_ratio))
            n_val = max(1, int(len(indices) * val_ratio))
            
            for i, idx in enumerate(indices):
                if i < n_train:
                    split = "train"
                elif i < n_train + n_val:
                    split = "val"
                else:
                    split = "test"
                
                self.metadata[idx]["split"] = split
                self.stats["per_split"][split] += 1
        
        print("✓ Split assignment complete:")
        for split in ["train", "val", "test"]:
            count = self.stats["per_split"][split]
            pct = 100.0 * count / self.stats["total_samples"]
            print(f"    {split:5s}: {count:4d} samples ({pct:5.1f}%)")
    
    def save_metadata(self):
        """Save metadata CSV file."""
        metadata_df = pd.DataFrame(self.metadata)
        metadata_path = self.mixed_dir / "metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)
        print(f"\n✓ Saved metadata to {metadata_path}")
        return metadata_df
    
    def print_report(self):
        """Print generation report."""
        print("\n" + "="*70)
        print("GENERATION REPORT")
        print("="*70)
        
        print(f"\nTOTAL SAMPLES: {self.stats['total_samples']}")
        
        print(f"\nPER-LABEL DISTRIBUTION:")
        for label in sorted(self.stats["per_label"].keys()):
            count = self.stats["per_label"][label]
            pct = 100.0 * count / self.stats["total_samples"]
            print(f"    {label:30s}: {count:4d} ({pct:5.1f}%)")
        
        print(f"\nPER-SNR DISTRIBUTION:")
        for snr in sorted(self.stats["per_snr"].keys()):
            count = self.stats["per_snr"][snr]
            pct = 100.0 * count / self.stats["total_samples"]
            print(f"    SNR {snr:+3d} dB: {count:4d} ({pct:5.1f}%)")
        
        print(f"\nTRAIN/VAL/TEST SPLIT:")
        for split in ["train", "val", "test"]:
            count = self.stats["per_split"][split]
            pct = 100.0 * count / self.stats["total_samples"]
            print(f"    {split:5s}: {count:4d} ({pct:5.1f}%)")
        
        print("\n" + "="*70)
    
    def run(self):
        """Execute full data preparation pipeline."""
        self.validate_directories()
        self.generate_samples()
        self.split_data()
        self.save_metadata()
        self.print_report()
        
        return self.stats


def main():
    parser = argparse.ArgumentParser(
        description="Prepare mixed audio + sensor data for AthenAI training"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1000,
        help="Total number of mixed samples to generate (default: 1000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--project_root",
        type=str,
        default="/home/nithira/Multimodal_Intent_Reconstruction_for_Safety_Critical_Communication_in_Extreme_Acoustic_Environments",
        help="Project root directory"
    )
    
    args = parser.parse_args()
    
    preparator = DataPreparator(
        project_root=args.project_root,
        n_samples=args.n_samples,
        seed=args.seed
    )
    
    stats = preparator.run()
    
    print("\n✓ Data preparation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
