# AthenAI Data Preparation Script

## Overview

`scripts/prepare_dataset.py` generates mixed speech+noise audio samples with synchronized sensor windows for training the AthenAI multimodal intent recognition system.

## Features

- **Speech Processing**: Loads and resamples speech commands to 16 kHz mono
- **Noise Augmentation**: Mixes clean speech with background noise at multiple SNR levels
  - SNR levels: [20, 10, 0, -5, -10, -20] dB
  - Uses standard audio SNR formula for realistic mixing
- **Sensor Integration**: Extracts 128-timestep windows from SKAB sensor data
  - Up to 8 features per timestep
  - Independent column normalization to [0, 1] range
- **Command Mapping**: Maps speech command labels to AthenAI safety vocabulary
  - 20 command classes defined in `src/utils/vocab.py`
  - Unknown commands mapped to "unknown command" (index 19)
- **Data Splitting**: Stratified train/val/test split (80/10/10 by default)
  - Maintains label distribution across splits
  - Reproducible with seed parameter
- **Metadata Tracking**: Generates `metadata.csv` with complete sample provenance

## Usage

### Basic Usage (1000 samples, default seed)

```bash
cd /home/nithira/Multimodal_Intent_Reconstruction_for_Safety_Critical_Communication_in_Extreme_Acoustic_Environments
.venv/bin/python scripts/prepare_dataset.py
```

### Custom Number of Samples

```bash
.venv/bin/python scripts/prepare_dataset.py --n_samples 5000
```

### Custom Random Seed (for reproducibility)

```bash
.venv/bin/python scripts/prepare_dataset.py --n_samples 1000 --seed 123
```

### Full Example with All Options

```bash
.venv/bin/python scripts/prepare_dataset.py \
    --n_samples 2000 \
    --seed 42 \
    --project_root /home/nithira/Multimodal_Intent_Reconstruction_for_Safety_Critical_Communication_in_Extreme_Acoustic_Environments
```

## Input Data Structure

The script expects the following directory structure:

```
data/
├── speech_commands/     # 36 folders with labeled speech audio
│   ├── yes/            # ~4000 .wav files
│   ├── no/             # ~3900 .wav files
│   ├── stop/           # ~3800 .wav files
│   ├── up/
│   ├── down/
│   └── ... (30 more command labels)
├── noise/              # 128 .wav files (~1-10 minutes each)
│   ├── AirConditioner_1.wav
│   ├── AirConditioner_2.wav
│   └── ...
└── sensors/            # 17 CSV files (SKAB anomaly detection dataset)
    ├── 0.csv
    ├── 1.csv
    └── ... (up to 15 + anomaly-free.csv)
```

## Output Data Structure

Generated files are saved to `data/mixed/`:

```
data/mixed/
├── metadata.csv                 # Indexed metadata for all samples
├── <uuid>_audio.wav            # Mixed speech+noise at 16 kHz mono
├── <uuid>_sensor.npy           # Sensor window [128, 8] normalized to [0,1]
├── <uuid2>_audio.wav
├── <uuid2>_sensor.npy
└── ... (2*n_samples files total)
```

### metadata.csv Columns

| Column | Type | Description |
|--------|------|-------------|
| audio_file | string | Path to mixed audio WAV file |
| sensor_file | string | Path to sensor data NPY file |
| command_label | string | AthenAI safety command label |
| command_idx | int | Command vocabulary index (0-19) |
| snr_db | float | SNR level of this sample in dB |
| split | string | dataset split: "train", "val", or "test" |

## Command Vocabulary Mapping

Speech commands from Google Speech Commands v0.02 are mapped to AthenAI safety vocabulary:

```
Speech Commands              →  AthenAI Safety Commands
─────────────────────────────────────────────────────
yes, happy, wow, house       →  all clear
no, tree                     →  emergency shutdown
stop                         →  stop the machine
go, seven                    →  start conveyor
up, forward                  →  increase pressure
down, backward               →  reduce speed
left, five                   →  open valve
right, six                   →  close valve
on                           →  activate safety lock
off                          →  release safety lock
follow, marvin, sheila       →  call supervisor
four, learn                  →  check sensor
one, bed                     →  restart system
eight                        →  halt conveyor
bird, cat, dog               →  electrical hazard
three, nine                  →  (mapped based on speech patterns)
(unknown or unlisted)        →  unknown command (index 19)
```

## Audio Mixing Formula

SNR (Signal-to-Noise Ratio) mixing uses the standard formula:

```
noise_scaled = noise * (rms(speech) / (rms(noise) * 10^(snr_db/20)))
mixed = speech + noise_scaled
```

Where:
- `rms(x)` = root mean square of signal x
- `snr_db` = desired SNR in decibels
- Soft clipping applied if mixed amplitude exceeds 1.0

### SNR Levels

- **20 dB**: Quiet noise (very high SNR, barely noticeable noise)
- **10 dB**: Clean audio with background noise
- **0 dB**: Equal speech and noise power
- **-5 dB**: Noise slightly louder than speech
- **-10 dB**: Significant noise (challenging task)
- **-20 dB**: Severe noise (robustness test)

## Sensor Data Processing

SKAB sensor CSV files contain 9 columns:

1. **datetime** - timestamp (skipped)
2. **Accelerometer1RMS** - vibration sensor 1
3. **Accelerometer2RMS** - vibration sensor 2
4. **Current** - electrical current
5. **Pressure** - system pressure
6. **Temperature** - ambient temperature
7. **Thermocouple** - additional temperature
8. **Voltage** - electrical voltage
9. **Volume Flow RateRMS** - flow rate

Features 2-9 are normalized independently to [0, 1] using min-max scaling:

```
feature_normalized = (feature - min(feature)) / (max(feature) - min(feature))
```

A random contiguous 128-timestep window is extracted from each CSV file.

## Sample Generation Statistics Example

```
======================================================================
GENERATION REPORT
======================================================================

TOTAL SAMPLES: 100

PER-LABEL DISTRIBUTION:
    activate safety lock          :    7 (  7.0%)
    all clear                     :   12 ( 12.0%)
    call supervisor               :    5 (  5.0%)
    ...
    stop the machine              :    4 (  4.0%)

PER-SNR DISTRIBUTION:
    SNR -20 dB:   20 ( 20.0%)
    SNR -10 dB:   16 ( 16.0%)
    SNR  -5 dB:   16 ( 16.0%)
    SNR  +0 dB:   16 ( 16.0%)
    SNR +10 dB:   16 ( 16.0%)
    SNR +20 dB:   16 ( 16.0%)

TRAIN/VAL/TEST SPLIT:
    train:   74 ( 74.0%)
    val  :   17 ( 17.0%)
    test :    9 (  9.0%)

======================================================================
```

## Requirements

The script requires these packages (already installed in `.venv`):

- torch >= 2.0.0
- torchaudio >= 2.0.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- tqdm

## Performance

Typical performance on NVIDIA RTX 3090:

- **Throughput**: ~50-80 samples per second
- **1000 samples**: ~15-20 seconds total
- **10,000 samples**: ~2-3 minutes total

Memory usage: ~500 MB for loading datasets, ~1-2 MB per sample in memory.

## Reproducibility

Use the `--seed` parameter to ensure reproducible results:

```bash
.venv/bin/python scripts/prepare_dataset.py --n_samples 1000 --seed 42
```

Same seed produces identical sample selection, audio mixing, sensor window extraction, and train/val/test splits across multiple runs.

## Troubleshooting

### "No speech files found" Error

**Cause**: `data/speech_commands/` directory empty or malformed

**Solution**: Run data organization step first:
```bash
.venv/bin/python scripts/download_datasets.py  # or your download script
```

### "No noise files found" Error

**Cause**: `data/noise/` directory empty

**Solution**: Ensure MS-SNSD noise files copied to `data/noise/`

### "No sensor files found" Error

**Cause**: `data/sensors/` directory empty

**Solution**: Ensure SKAB CSV files copied to `data/sensors/`

### Audio files have incorrect properties

**Check**: Verify source data
- Speech: 16 kHz sample rate, mono or stereo (script handles both)
- Noise: Any sample rate (resampled automatically)

### Sensor data all zeros

**Cause**: CSV file has insufficient variation

**Solution**: Script pads with zeros if needed; legitimate edge case

## Advanced Usage

### Loading Generated Data in Training

```python
import pandas as pd
import numpy as np
import torchaudio

# Load metadata
metadata = pd.read_csv("data/mixed/metadata.csv")

# Filter for training split
train_metadata = metadata[metadata["split"] == "train"]

# Load a single sample
for idx, row in train_metadata.iterrows():
    # Load audio
    audio, sr = torchaudio.load(f"data/mixed/{row['audio_file']}")
    
    # Load sensor
    sensor = np.load(f"data/mixed/{row['sensor_file']}")
    
    # Get command label info
    command_label = row["command_label"]
    command_idx = row["command_idx"]
    snr_db = row["snr_db"]
    
    # ... your training code ...
```

### Generating Multiple Datasets with Different Seeds

```bash
# Generate diverse training datasets
for seed in 42 56 99 123 456; do
    .venv/bin/python scripts/prepare_dataset.py \
        --n_samples 1000 \
        --seed $seed \
        --project_root /path/to/project
done
```

## Citation

If you use this script in research, please cite:

```
@software{athenai_prepare_dataset,
  title={Data Preparation Script for AthenAI Multimodal Intent Recognition},
  author={AthenAI Team},
  year={2026},
  url={https://github.com/your-repo/scripts/prepare_dataset.py}
}
```

## License

Same as the main AthenAI project.
