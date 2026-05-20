# AthenAI Research Guide
## Robust Safety Command Inference via Uncertainty-Aware Multimodal Fusion

> Last updated: April 2026
> Status: Prototype architecture complete — training and data pipeline pending

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Current Implementation Status](#3-current-implementation-status)
4. [Critical Finding: The Audio-JEPA Is Not JEPA](#4-critical-finding-the-audio-jepa-is-not-jepa)
5. [Audio Encoder Options Evaluated](#5-audio-encoder-options-evaluated)
6. [Recommended Encoder: WavJEPA-Nat-Base](#6-recommended-encoder-wavjepa-nat-base)
7. [Dataset Strategy](#7-dataset-strategy)
8. [Training Roadmap](#8-training-roadmap)
9. [Known Gaps and Next Steps](#9-known-gaps-and-next-steps)

---

## 1. Project Overview

**Problem:** Industrial workers communicate safety commands (e.g., "emergency shutdown", "evacuate immediately") in environments with sustained noise levels of 100+ dB. At these levels, conventional Automatic Speech Recognition (ASR) fails completely. A misheard or missed safety command can cause injury or death.

**Solution (AthenAI):** A multimodal system that combines:
- **Noisy audio** — what was heard, even if mostly noise
- **Industrial sensor readings** — temperature, pressure, vibration; contextual clues about machine state

These two signals are fused to infer the most likely intended safety command, paired with a calibrated confidence score and a structured safety alert.

**Key Design Principle:** Never guess silently. Every prediction comes with calibrated confidence and uncertainty. Low confidence triggers human-in-the-loop verification.

---

## 2. System Architecture

The system has two operating modes:

### Base Mode (Audio-Only)

```
Noisy Waveform
    → Audio Encoder → [B, N_patches, 768]
    → Mean Pooling  → [B, 768]
    → Command Classifier (MC Dropout)
    → Command + Confidence + Uncertainty
```

### Full Mode (Multimodal)

```
Noisy Waveform
    → Audio Encoder  → [B, N_patches, 768] ─┐
                                              ├─→ Cross-Attention Fusion → [B, 512]
Sensor Time-Series                            │       → Command Classifier
    → Sensor Encoder → [B, 256] ─────────────┘           → MC Dropout Inference
                                                      → Command + Confidence + Uncertainty
                                                      → Clean Audio Synthesis (AudioGen)
                                                      → LLM Safety Alert
```

### Module Map

| Module | File | Purpose |
|---|---|---|
| Audio Encoder | `src/encoders/audio_jepa.py` | Encode noisy speech into patch embeddings |
| Sensor Encoder | `src/encoders/sensor_encoder.py` | Encode multivariate time-series sensor data |
| Fusion | `src/fusion/cross_attention_fusion.py` | Cross-modal attention: speech as query, sensors as context |
| Classifier | `src/classification/command_classifier.py` | 2-layer MLP + temperature scaling + MC Dropout |
| Synthesizer | `src/synthesis/clean_audio_synthesizer.py` | AudioGen text-to-audio for operator verification |
| Alert | `src/alert/llm_alert.py` | LLM-based structured safety alert generation |
| Preprocessing | `src/utils/preprocessing.py` | Waveform → log-mel spectrogram (used by current ViT encoder) |
| Vocabulary | `src/utils/vocab.py` | 20 safety command strings + index mappings |
| Pipeline | `src/pipeline.py` | `AthenAISystem` — orchestrates full inference |
| CLI | `run_inference.py` | Command-line inference demo |

### Command Vocabulary (20 Commands)

```
 0: stop the machine          10: call supervisor
 1: emergency shutdown        11: check sensor
 2: evacuate immediately      12: restart system
 3: reduce speed              13: halt conveyor
 4: increase pressure         14: start conveyor
 5: decrease pressure         15: fire alarm
 6: open valve                16: chemical leak alert
 7: close valve               17: electrical hazard
 8: activate safety lock      18: all clear
 9: release safety lock       19: unknown command
```

---

## 3. Current Implementation Status

| Component | Status | Notes |
|---|---|---|
| Audio encoder (skeleton) | ✅ Exists | Uses ImageNet ViT — NOT true JEPA. See Section 4. |
| Sensor encoder | ✅ Exists | Transformer + positional encoding, solid design |
| Cross-attention fusion | ✅ Exists | 8-head attention, speech-as-query pattern |
| Command classifier + MC Dropout | ✅ Exists | Temperature scaling + entropy-based uncertainty |
| Clean audio synthesis | ✅ Exists | Uses `facebook/audiogen-medium` |
| LLM safety alert | ✅ Exists | OpenAI-compatible client interface |
| Inference script | ✅ Exists | `run_inference.py`, CLI-ready |
| Dataset | ❌ Missing | Strategy documented in `Datasets/README.md`, no data downloaded |
| Data mixing pipeline | ❌ Missing | Speech + industrial noise mixing code needed |
| Training script (supervised) | ❌ Missing | Fine-tuning fusion + classifier |
| Self-supervised pretraining loop | ❌ Missing | JEPA pretraining entirely absent |
| Evaluation / metrics code | ❌ Missing | Accuracy, F1, ECE not implemented |
| True JEPA audio encoder | ❌ Not implemented | Recommend replacing. See Sections 4–6. |

---

## 4. Critical Finding: The Audio-JEPA Is Not JEPA

### What Is Currently in `src/encoders/audio_jepa.py`

The `AudioJEPAEncoder` class:

- Loads `google/vit-base-patch16-224` — an **image** Vision Transformer pretrained on **ImageNet**
- Uses a `Conv2d` patch embedding layer to convert mel-spectrograms into patch tokens
- Creates a frozen EMA copy of the ViT named `target_encoder`
- `forward()` only calls the context encoder — **`target_encoder` is never called during inference**
- `update_target()` is defined but **never called anywhere in the pipeline**

In short: it is a ViT feature extractor with an unused EMA copy attached. It will produce reasonable embeddings (ViT is powerful), but it is not trained with a JEPA objective and does not have JEPA's noise robustness properties.

### What a True Audio-JEPA Actually Is

A JEPA (Joint-Embedding Predictive Architecture) has four essential components:

1. **Masking strategy** — randomly mask regions of the input (for audio: mask temporal segments of the spectrogram or waveform)
2. **Context encoder** — encodes only the unmasked (visible) regions
3. **Target encoder** — encodes the full unmasked input; updated via EMA, no gradient flow
4. **Predictor** — takes context encoder output and predicts the target encoder's representations of the **masked regions**

**Training objective:** minimize distance (L2 or cosine) between predictor output and target encoder output in embedding space. No pixel/frame reconstruction, no contrastive negatives.

The current code has a partial version of component 3 but is missing: masked context encoding, and entirely missing the predictor (component 4). JEPA's noise robustness comes precisely from training the context encoder to infer full semantic embeddings from partial, corrupted observations — exactly the industrial noise scenario.

---

## 5. Audio Encoder Options Evaluated

### Option A: Sony CSL — audio-representations

- **Repository:** https://github.com/SonyCSLParis/audio-representations
- **True JEPA:** Yes — masked patch prediction on mel-spectrograms, ICASSP 2024
- **Pretrained weights:** ❌ Author accidentally deleted them and has permanently left the project
- **Maintenance:** ❌ Inactive — README states "I no longer work on this topic"
- **Verdict:** Architecture is a valid reference, but nothing is usable. Do not use.

### Option B: WavJEPA — labhamlet

- **Repository:** https://github.com/labhamlet/wavjepa
- **Paper:** arXiv:2509.23238
- **True JEPA:** Yes — full JEPA on raw waveforms
- **Architecture:** Wav2Vec 2.0 feature encoder → ViT context encoder + ViT target encoder (EMA) + ViT predictor
- **Pretrained weights:** ✅ Available on HuggingFace
- **Maintenance:** ✅ Active — updated November 2025
- **License:** MIT

Two variants available:

| Model | Training Data | Use Case |
|---|---|---|
| `labhamlet/wavjepa-base` | AudioSet (1.74M clips) | General audio |
| `labhamlet/wavjepa-nat-base` | AudioSet + 85k noisy/reverberant simulations | Noisy/reverberant environments ← **recommended** |

---

## 6. Recommended Encoder: WavJEPA-Nat-Base

**HuggingFace model ID:** `labhamlet/wavjepa-nat-base`

### Why This Model

| Property | Detail |
|---|---|
| Noise robustness | Trained with WHAMR! noise at SNR 5–40 dB and 85k simulated binaural room impulse responses |
| True JEPA | Learns semantic embeddings via masked prediction, not reconstruction |
| Ready to use | Pretrained weights on HuggingFace, Transformers-compatible |
| Input format | Raw 16kHz waveforms — no mel-spectrogram preprocessing needed |
| Size | 0.2B parameters — suitable for research compute budgets |
| License | MIT |

### What Changes in the Pipeline

Replacing `AudioJEPAEncoder` with WavJEPA-Nat-Base requires three targeted changes:

1. **Input format:** Remove the `waveform_to_mel()` preprocessing step. WavJEPA takes raw waveforms directly.
2. **Output shape:** WavJEPA produces `[B, 200, 768]` for 2-second audio at 16kHz. The fusion module already expects `[B, N_patches, 768]` — fully compatible, no fusion changes needed.
3. **Encoder loading:** Replace `ViTModel.from_pretrained('google/vit-base-patch16-224')` with `AutoModel.from_pretrained('labhamlet/wavjepa-nat-base')`.

All other modules — `SensorEncoder`, `CrossAttentionFusion`, `CommandClassifier`, and everything downstream — remain unchanged.

---

## 7. Dataset Strategy

### Clean Speech Commands

| Dataset | Type | Access |
|---|---|---|
| Google Speech Commands v2 | Short command words (~105k clips, 35 classes) | HuggingFace: `google/speech_commands` |
| Fluent Speech Corpus | Phrase-level intent commands (~30k utterances) | Kaggle: `tommyngx/fluent-speech-corpus` |

> Note: Neither dataset contains the exact 20-command vocabulary used in AthenAI. Closest mappings: "stop", "go", "yes", "no" from Speech Commands; intent-mapped phrases from Fluent Speech. Custom recordings of all 20 safety commands may be necessary for best results.

### Industrial Background Noise

| Dataset | Type | Access |
|---|---|---|
| MS-SNSD | Factory, machinery, HVAC noise files | GitHub: `microsoft/MS-SNSD` |
| MIMII | Real malfunctioning industrial machinery audio | GitHub: `MIMII-hitachi/mimii_baseline` |
| ESC-50 | Environmental sounds (filter: industrial categories) | GitHub: `karolpiczak/ESC-50` |

### Industrial Sensor Time-Series

| Dataset | Type | Access |
|---|---|---|
| SKAB | Real water pump sensor readings (8 channels) | GitHub: `waico/SKAB` |
| C-MAPSS | Turbofan engine degradation sensor streams | NASA PCOE data repository |

### Data Mixing Strategy

1. Select one clean speech command clip
2. Select one industrial noise clip
3. Mix at target SNR: {20, 10, 0, −5, −10, −20} dB using standard SNR mixing formula
4. Select a 128-timestep sensor window from SKAB aligned to a plausible machine state for that command
5. Save as `(noisy_audio.wav, sensor_window.npy, command_label, snr_db)` tuple
6. Record in `metadata.csv`: `audio_file, sensor_file, command_label, command_idx, snr_db, split`

**Target dataset size:** 10,000 labeled pairs minimum for initial training; 50,000+ for meaningful generalization.

---

## 8. Training Roadmap

Each step is described with a ready-to-use agent prompt. Copy the prompt and give it to an AI coding agent to implement that step. Do not implement steps out of order — each step depends on the previous.

---

### Step 0: Environment Setup

**Goal:** Install all dependencies and verify GPU availability and WavJEPA model loading.

**Agent prompt:**
```
Set up the Python environment for the AthenAI project at:
/home/naveen/wso2/Multimodal_Intent_Reconstruction_for_Safety_Critical_Communication_in_Extreme_Acoustic_Environments

Tasks:
1. Install all packages from requirements.txt
2. Additionally install: huggingface-hub, datasets, accelerate
3. Verify CUDA availability via torch.cuda.is_available()
4. Attempt to load the WavJEPA-Nat-Base model from HuggingFace:
   - AutoModel.from_pretrained('labhamlet/wavjepa-nat-base')
   - AutoFeatureExtractor.from_pretrained('labhamlet/wavjepa-nat-base')
5. Report: Python version, PyTorch version, CUDA status, GPU name (if available), model load status
```

---

### Step 1: Download and Organize Datasets

**Goal:** Download Google Speech Commands v2, MS-SNSD noise files, and SKAB sensor data.

**Agent prompt:**
```
Download and organize training datasets for the AthenAI project at:
/home/naveen/wso2/Multimodal_Intent_Reconstruction_for_Safety_Critical_Communication_in_Extreme_Acoustic_Environments

Create this directory structure under the project root:
  data/
    speech_commands/   (audio files organized by label subfolder)
    noise/             (all noise .wav files, flat)
    sensors/           (SKAB CSV files)
    mixed/             (empty — will be populated in Step 2)

Download:
1. Google Speech Commands v2 — use HuggingFace datasets library:
   load_dataset('google/speech_commands', 'v0.02')
   Save audio files into data/speech_commands/<label>/ as .wav

2. MS-SNSD noise files — clone https://github.com/microsoft/MS-SNSD
   Copy all .wav files from noise_train/ into data/noise/

3. SKAB sensor dataset — clone https://github.com/waico/SKAB
   Copy all .csv files into data/sensors/

Report: file counts per speech label, total noise files, total sensor CSV files.
```

---

### Step 2: Build Data Mixing Pipeline

**Goal:** Generate (noisy audio, sensor window, label, SNR) training pairs from the raw data.

**Agent prompt:**
```
Create a data preparation script for the AthenAI project at:
/home/naveen/wso2/Multimodal_Intent_Reconstruction_for_Safety_Critical_Communication_in_Extreme_Acoustic_Environments

Place the script at: scripts/prepare_dataset.py

Requirements:
1. Load clean speech .wav files from data/speech_commands/ — resample to 16kHz, convert to mono using torchaudio
2. Load noise .wav files from data/noise/ — resample to 16kHz, convert to mono
3. Mix speech + noise at SNR levels [20, 10, 0, -5, -10, -20] dB using the standard SNR formula:
   noise_scaled = noise * (rms(speech) / (rms(noise) * 10^(snr/20)))
   mixed = speech + noise_scaled
4. Load a random contiguous 128-timestep window from a random SKAB CSV file — use all available numeric columns up to 8; normalize each column independently to [0, 1]
5. Save mixed audio to data/mixed/ as <id>_audio.wav
6. Save sensor window to data/mixed/ as <id>_sensor.npy (shape [128, 8])
7. Build metadata.csv at data/mixed/metadata.csv with columns:
   audio_file, sensor_file, command_label, command_idx, snr_db, split
8. Split: 80% train, 10% val, 10% test (random, stratified by command label)
9. Map speech command labels to AthenAI vocabulary in src/utils/vocab.py (COMMAND_VOCAB list)
   Commands not in vocabulary → label "unknown command" (index 19)
10. Generate at least 1000 mixed samples total across all SNR levels

Accept command-line arguments: --n_samples (default 1000), --seed (default 42)
Report: total samples generated, per-label distribution, per-SNR distribution.
```

---

### Step 3: Replace Audio Encoder with WavJEPA-Nat-Base

**Goal:** Swap the current ImageNet ViT encoder for the pretrained WavJEPA-Nat-Base. Adjust the pipeline to pass raw waveforms directly.

**Agent prompt:**
```
Modify the audio encoder in the AthenAI project at:
/home/naveen/wso2/Multimodal_Intent_Reconstruction_for_Safety_Critical_Communication_in_Extreme_Acoustic_Environments

File to modify: src/encoders/audio_jepa.py
File to modify: src/encoders/__init__.py
File to modify: src/pipeline.py

Changes required:

1. In src/encoders/audio_jepa.py:
   - Replace the AudioJEPAEncoder class with a new class named WavJEPAEncoder
   - The new class must:
     a. Load AutoModel.from_pretrained('labhamlet/wavjepa-nat-base') as self.model
     b. Load AutoFeatureExtractor.from_pretrained('labhamlet/wavjepa-nat-base') as self.feature_extractor
     c. Freeze all model parameters (requires_grad = False)
     d. Accept forward(waveform: torch.Tensor) where waveform is [B, T] at 16kHz
     e. Return last_hidden_state: shape [B, N, 768]
   - Keep the file at src/encoders/audio_jepa.py

2. In src/encoders/__init__.py:
   - Export WavJEPAEncoder instead of AudioJEPAEncoder

3. In src/pipeline.py:
   - Import WavJEPAEncoder instead of AudioJEPAEncoder
   - In AthenAISystem.__init__: instantiate WavJEPAEncoder
   - In AthenAISystem.infer: remove the waveform_to_mel() call; pass noisy_waveform directly to the audio encoder

Do not modify any other files. Do not change SensorEncoder, CrossAttentionFusion, CommandClassifier, or downstream modules.
```

---

### Step 4: Write the Training Script

**Goal:** Train the fusion + classifier layers while keeping the WavJEPA encoder frozen (transfer learning).

**Agent prompt:**
```
Create a supervised training script for the AthenAI project at:
/home/naveen/wso2/Multimodal_Intent_Reconstruction_for_Safety_Critical_Communication_in_Extreme_Acoustic_Environments

Place it at: scripts/train.py

Requirements:

Dataset:
- Read data/mixed/metadata.csv
- Build a PyTorch Dataset class (SafetyCommandDataset) that:
  - Loads audio .wav with torchaudio, resamples to 16kHz, converts to mono, pads/trims to 2 seconds (32000 samples)
  - Loads sensor .npy as float32 tensor [128, 8]
  - Returns (audio_tensor [32000], sensor_tensor [128, 8], command_idx int)
- DataLoader: batch_size=32, shuffle=True for train, shuffle=False for val/test, num_workers=4

Model (full mode):
- WavJEPAEncoder from src/encoders/audio_jepa.py — FREEZE all parameters
- SensorEncoder from src/encoders/sensor_encoder.py — trainable
- CrossAttentionFusion from src/fusion/cross_attention_fusion.py — trainable
- CommandClassifier from src/classification/command_classifier.py — trainable

Model (base mode):
- WavJEPAEncoder — FREEZE all parameters
- CommandClassifier with input_dim=768 — trainable
- Use mean pooling over WavJEPA output before classifier

Training:
- Optimizer: AdamW on trainable parameters only, lr=3e-4, weight_decay=1e-4
- Loss: CrossEntropyLoss
- Train for up to 50 epochs
- Early stopping: patience=10 on validation loss
- Save best checkpoint (lowest val loss) to checkpoints/best_model.pt

Logging per epoch: train_loss, val_loss, val_accuracy

At end of training, load best checkpoint and report:
- Test accuracy
- Per-class F1 score for all 20 commands (use sklearn.metrics.classification_report)

CLI arguments: --epochs (default 50), --batch_size (default 32), --lr (default 3e-4), --mode (base|full, default full)

Use pure PyTorch — no Lightning or other training frameworks.
```

---

### Step 5: Evaluate Calibration and Robustness

**Goal:** Measure prediction accuracy, per-SNR robustness, and calibration quality of confidence scores.

**Agent prompt:**
```
Create an evaluation script for the AthenAI project at:
/home/naveen/wso2/Multimodal_Intent_Reconstruction_for_Safety_Critical_Communication_in_Extreme_Acoustic_Environments

Place it at: scripts/evaluate.py

Requirements:
1. Load the trained model checkpoint from checkpoints/best_model.pt
2. Load the test split from data/mixed/metadata.csv (split == "test")
3. Run mc_dropout_inference (n_mc=20) from src/classification/command_classifier.py on all test samples
4. Compute and print:
   a. Overall top-1 accuracy
   b. Per-class accuracy and F1 score for all 20 commands (sklearn classification_report)
   c. Accuracy and mean confidence per SNR level (20, 10, 0, -5, -10, -20 dB)
   d. Expected Calibration Error (ECE):
      - Bin predictions into 10 equal-width confidence bins [0,0.1), [0.1,0.2), ..., [0.9,1.0]
      - Per bin: |mean_confidence - accuracy|, weighted by bin size
      - ECE = weighted average across bins
   e. Mean entropy (uncertainty) per SNR level
5. Save all numeric metrics to eval_results.json
6. Save two plots:
   - eval_reliability.png: reliability diagram (confidence bins vs accuracy, with diagonal reference line)
   - eval_uncertainty.png: histogram of entropy scores across test set, colored by SNR

CLI arguments: --checkpoint (default checkpoints/best_model.pt), --mode (base|full, default full)
```

---

## 9. Known Gaps and Next Steps

### Gaps Blocking Training

| Gap | Required Step |
|---|---|
| No training data | Steps 1 and 2 |
| Audio encoder lacks JEPA properties | Step 3 |
| No training script | Step 4 |
| No evaluation code | Step 5 |

### Research Quality Improvements (Post Initial Training)

- **Domain-adaptive JEPA fine-tuning:** If custom industrial audio recordings become available, fine-tune WavJEPA-Nat-Base further using the JEPA training loop from the [wavjepa repository](https://github.com/labhamlet/wavjepa). This requires the `train.py` and config files from that repo.

- **Pretrained sensor encoder:** Replace the custom `SensorEncoder` with a pretrained time-series foundation model. TimeMAE and TS2Vec are already referenced in `Resources/SensorEmbedding/` — both have open-source implementations and pretrained weights.

- **SNR conditioning:** Pass the estimated SNR value as an additional input feature to the classifier, allowing the model to modulate uncertainty based on how noisy it knows the signal is.

- **Hard negative vocabulary pairs:** Some command pairs are acoustically or semantically similar (e.g., "open valve" / "close valve", "increase pressure" / "decrease pressure"). Once an initial model is trained, add hard negative augmentation to explicitly train the classifier on these confusable pairs.

- **Sensor-modality ablation study:** Systematically compare base mode vs. full mode performance at each SNR level. This quantifies how much sensor data helps and at which noise levels it matters most — important for the research paper.

### Architectural Improvements to Consider

- **Ensemble uncertainty:** The current uncertainty is MC Dropout entropy only. Combining it with temperature-scaled softmax entropy provides a richer epistemic vs. aleatoric uncertainty decomposition.

- **Confidence thresholds are heuristics:** The `< 0.4` and `< 0.5` thresholds in the alert generator and synthesizer are arbitrary. Calibrate them from the ECE evaluation: set thresholds at the confidence level where the reliability diagram shows the model starts to be overconfident.

- **Deterministic alerts for safety:** LLM-generated alerts can hallucinate. For a production safety system, replace `llm_alert.py` with a deterministic template-based alert generator keyed on `(command, severity_level)`. LLM generation is appropriate for research demos, not deployment.

### Safety Considerations

- Always maintain a human-in-the-loop verification step for any predicted command that would trigger an irreversible or high-consequence action
- Design the system to fail safe: when confidence is low or uncertainty is high, the default action should be to alert a supervisor, not to execute the command
- Log all predictions with their confidence scores and sensor readings for post-incident forensic analysis

---

*This document was generated as part of the AthenAI research planning process. Implementation prompts in Section 8 are designed to be executed sequentially by an AI coding agent.*
