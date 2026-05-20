"""
Data Preparation Script for AthenAI Project
============================================

State-of-the-Art overhauling:
- Target-driven class balancing (equal samples per safety command)
- Multiprocessing (scales easily to 50k+ samples)
- Continuous SNR sampling U(-25, 15) dB
- Speech Augmentation (Pitch/Speed shifts, reverberation)
- Sensor Augmentation (Gaussian jitter, Random dynamic scaling)
"""

import os
import sys
import argparse
import uuid
import multiprocessing
from pathlib import Path
from collections import defaultdict
import random
import traceback
import math

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

COMMAND_VOCAB = [
    "stop the machine", "emergency shutdown", "evacuate immediately", "reduce speed",
    "increase pressure", "decrease pressure", "open valve", "close valve",
    "activate safety lock", "release safety lock", "call supervisor", "check sensor",
    "restart system", "halt conveyor", "start conveyor", "fire alarm",
    "chemical leak alert", "electrical hazard", "all clear", "unknown command"
]
COMMAND_TO_IDX = {cmd: idx for idx, cmd in enumerate(COMMAND_VOCAB)}

SPEECH_TO_SAFETY_MAPPING = {
    "zero": "all clear", "one": "activate safety lock", "two": "increase pressure", 
    "three": "reduce speed", "four": "check sensor", "five": "open valve", 
    "six": "close valve", "seven": "start conveyor", "eight": "halt conveyor", 
    "nine": "restart system", "stop": "stop the machine", "go": "start conveyor", 
    "forward": "increase pressure", "backward": "reduce speed", "up": "increase pressure", 
    "down": "decrease pressure", "left": "open valve", "right": "close valve", 
    "yes": "all clear", "no": "emergency shutdown", "on": "activate safety lock", 
    "off": "release safety lock", "follow": "call supervisor", "learn": "check sensor", 
    "bed": "restart system", "bird": "chemical leak alert", "cat": "electrical hazard", 
    "dog": "electrical hazard", "happy": "all clear", "house": "fire alarm", 
    "marvin": "call supervisor", "sheila": "call supervisor", "tree": "evacuate immediately", 
    "wow": "emergency shutdown", "visual": "all clear", "_background_noise_": "unknown command",
}

# --- Audio Augmentation Helpers ---
_RESAMPLERS = {}
def get_resampler(orig_freq, new_freq):
    key = (orig_freq, new_freq)
    if key not in _RESAMPLERS:
        _RESAMPLERS[key] = T.Resample(orig_freq, new_freq)
    return _RESAMPLERS[key]

def apply_reverb(waveform, sample_rate):
    """Simple simulated reverberation via delay lines."""
    if random.random() > 0.5:
        # Add a single decay reflection
        delay_ms = random.uniform(20, 100)
        decay = random.uniform(0.1, 0.4)
        delay_samples = int(sample_rate * (delay_ms / 1000.0))
        
        padded = torch.nn.functional.pad(waveform, (0, delay_samples))
        delayed = torch.nn.functional.pad(waveform, (delay_samples, 0)) * decay
        waveform = padded + delayed
        
        # Max length back to original padded size approximately
    return waveform

def apply_speed_pitch_perturbation(waveform, orig_freq=16000):
    """Change pitch/speed by resampling and pretending it remained at orig_freq."""
    if random.random() > 0.7:
        # Resample factor between 0.85 (faster/higher) and 1.15 (slower/lower)
        factor = random.uniform(0.85, 1.15)
        new_freq = int(orig_freq * factor)
        resampler = get_resampler(orig_freq, new_freq)
        waveform = resampler(waveform)
    return waveform

def augment_sensor(window):
    """Apply slight jitter and scaling to sensor data."""
    if random.random() > 0.3:
        # Random scaling [0.9, 1.1]
        scale = np.random.uniform(0.9, 1.1, size=window.shape[1])
        window = window * scale
    if random.random() > 0.3:
        # Gaussian jitter
        jitter = np.random.normal(0, 0.05, window.shape)
        window = np.clip(window + jitter, 0, 1)
    return window

def mix_speech_noise(speech, noise, snr_db):
    """Mix speech and noise at a specific SNR."""
    if noise.shape[-1] < speech.shape[-1]:
        repeats = math.ceil(speech.shape[-1] / noise.shape[-1])
        noise = noise.repeat(1, repeats)
        
    # Random crop noise instead of start
    max_start = max(0, noise.shape[-1] - speech.shape[-1])
    start = random.randint(0, max_start)
    noise = noise[:, start:start+speech.shape[-1]]
    
    rms_speech = torch.sqrt(torch.mean(speech ** 2))
    rms_noise = torch.sqrt(torch.mean(noise ** 2))
    
    if rms_noise < 1e-10:
        return speech
        
    snr_linear = 10 ** (snr_db / 20)
    noise_scaled = noise * (rms_speech / (rms_noise * snr_linear))
    mixed = speech + noise_scaled
    
    max_val = torch.max(torch.abs(mixed))
    if max_val > 1.0:
        mixed = mixed / max_val * 0.95
    return mixed

# --- Worker Logic ---
def create_sample_worker(task):
    """Worker function to generate a single sample."""
    try:
        (task_id, target_command, target_idx, speech_path, noise_path, 
         sensor_path, mixed_dir, target_sr, is_unknown) = task
        
        # Load and augment speech
        speech, sr = torchaudio.load(str(speech_path))
        if speech.shape[0] > 1:
            speech = speech.mean(dim=0, keepdim=True)
            
        # Optional: Randomly crop background noise files if they are long
        if is_unknown and speech.shape[-1] > target_sr * 2:
            max_start = speech.shape[-1] - target_sr * 2
            start = random.randint(0, max_start)
            speech = speech[:, start:start + target_sr * 2]
            
        if sr != target_sr:
            speech = get_resampler(sr, target_sr)(speech)
            
        speech = apply_speed_pitch_perturbation(speech, target_sr)
        speech = apply_reverb(speech, target_sr)
        
        # Load noise
        noise, n_sr = torchaudio.load(str(noise_path))
        if noise.shape[0] > 1:
            noise = noise.mean(dim=0, keepdim=True)
        if n_sr != target_sr:
            noise = get_resampler(n_sr, target_sr)(noise)
            
        # Mix
        snr_db = random.uniform(-25.0, 15.0)
        mixed = mix_speech_noise(speech, noise, snr_db)
        
        # Load and augment sensor
        df = pd.read_csv(sensor_path, sep=";")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:8]
        data = []
        for col in numeric_cols:
            col_data = df[col].values.astype(float)
            col_min, col_max = np.min(col_data), np.max(col_data)
            if col_max > col_min:
                col_data = (col_data - col_min) / (col_max - col_min)
            else:
                col_data = np.zeros_like(col_data)
            data.append(col_data)
            
        # Pad features if < 8
        while len(data) < 8:
            data.append(np.zeros(len(df)))
            
        data = np.stack(data, axis=1)
        window_size = 128
        if len(data) < window_size:
            pad_size = window_size - len(data)
            data = np.vstack([data, np.zeros((pad_size, 8))])
            window = data[:window_size]
        else:
            start_idx = random.randint(0, len(data) - window_size)
            window = data[start_idx:start_idx + window_size]
            
        window = augment_sensor(window)
        
        # Save
        uuid_str = str(uuid.uuid4())[:8]
        audio_filename = f"{task_id}_{uuid_str}_audio.wav"
        sensor_filename = f"{task_id}_{uuid_str}_sensor.npy"
        
        audio_path = os.path.join(mixed_dir, audio_filename)
        sensor_out_path = os.path.join(mixed_dir, sensor_filename)
        
        torchaudio.save(audio_path, mixed, target_sr)
        np.save(sensor_out_path, window)
        
        return {
            "audio_file": audio_filename,
            "sensor_file": sensor_filename,
            "command_label": target_command,
            "command_idx": target_idx,
            "snr_db": snr_db,
            "error": None
        }
    except Exception as e:
        return {
            "error": str(e)
        }


# --- Main Manager ---
class DataPreparatorFast:
    def __init__(self, project_root, n_samples=50000, seed=42):
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data"
        self.speech_dir = self.data_dir / "speech_commands"
        self.noise_dir = self.data_dir / "noise"
        self.sensors_dir = self.data_dir / "sensors"
        self.mixed_dir = self.data_dir / "mixed"
        
        self.n_samples = n_samples
        self.seed = seed
        self.target_sr = 16000
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    def run(self):
        print("Clearing old mixed dataset dir...")
        self.mixed_dir.mkdir(parents=True, exist_ok=True)
        # Clear existing
        for f in self.mixed_dir.glob("*"):
            if f.is_file():
                f.unlink()
                
        # 1. Map available speech files to Safety Intents
        intent_to_speech_files = defaultdict(list)
        for label_dir in self.speech_dir.iterdir():
            if not label_dir.is_dir(): continue
            sp_label = label_dir.name
            safe_intent = SPEECH_TO_SAFETY_MAPPING.get(sp_label, None)
            if safe_intent is not None:
                wavs = list(label_dir.glob("*.wav"))
                intent_to_speech_files[safe_intent].extend(wavs)
                
        # 2. Gather noises and sensors
        noise_files = list(self.noise_dir.glob("*.wav"))
        sensor_files = list(self.sensors_dir.glob("*.csv"))
        
        n_classes = len(COMMAND_VOCAB)
        samples_per_class = max(1, self.n_samples // n_classes)
        
        print(f"Generating ~{samples_per_class} samples per class to total {self.n_samples}")
        
        tasks = []
        task_id = 0
        for intent in COMMAND_VOCAB:
            idx = COMMAND_TO_IDX[intent]
            available_speech = intent_to_speech_files[intent]
            if not available_speech:
                print(f"WARNING: No speech files mapped for '{intent}'! Proceeding to next class.")
                continue
                
            for _ in range(samples_per_class):
                s_path = random.choice(available_speech)
                n_path = random.choice(noise_files)
                sens_path = random.choice(sensor_files)
                # target is unknown if the word is background noise
                is_unk = intent == "unknown command"
                
                tasks.append((task_id, intent, idx, s_path, n_path, sens_path, str(self.mixed_dir), self.target_sr, is_unk))
                task_id += 1
                
        # Random shuffle tasks so progress bar is generic
        random.shuffle(tasks)
        
        print(f"Dispatching {len(tasks)} tasks across {multiprocessing.cpu_count()} workers...")
        metadata = []
        errors = 0
        
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            for result in tqdm(pool.imap_unordered(create_sample_worker, tasks), total=len(tasks)):
                if result["error"] is not None:
                    errors += 1
                else:
                    metadata.append(result)
                    
        print(f"Completed with {errors} errors.")
        
        # Train/Val/Test Split (80/10/10) stratified
        df = pd.DataFrame(metadata)
        df["split"] = ""
        
        for intent in df["command_label"].unique():
            idx = df[df["command_label"] == intent].index
            idx = np.random.permutation(idx)
            n = len(idx)
            n_train = int(n * 0.8)
            n_val = int(n * 0.1)
            
            df.loc[idx[:n_train], "split"] = "train"
            df.loc[idx[n_train:n_train+n_val], "split"] = "val"
            df.loc[idx[n_train+n_val:], "split"] = "test"
            
        csv_path = self.mixed_dir / "metadata.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved metadata to {csv_path}")
        print("\nData distribution:")
        print(df.groupby(["command_label", "split"]).size().unstack(fill_value=0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=50000)
    parser.add_argument("--project_root", type=str, default="/home/nithira/Multimodal_Intent_Reconstruction_for_Safety_Critical_Communication_in_Extreme_Acoustic_Environments")
    args = parser.parse_args()
    
    prep = DataPreparatorFast(project_root=args.project_root, n_samples=args.n_samples)
    prep.run()
