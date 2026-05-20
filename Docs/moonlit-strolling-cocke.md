# Research Planning: Robust Safety Command Inference via Uncertainty-Aware Multimodal Fusion

## Context
Full codebase analysis completed. This plan contains:
1. My analysis of the current state
2. Audio-JEPA resource evaluation
3. The full draft of RESEARCH_GUIDE.md to be created in the project root after user approval

After approval, I will create one file:
**`/home/naveen/wso2/Multimodal_Intent_Reconstruction_for_Safety_Critical_Communication_in_Extreme_Acoustic_Environments/RESEARCH_GUIDE.md`**

---

## Critical Findings (For User Review)

### Finding 1: The Audio-JEPA Is Not JEPA
`src/encoders/audio_jepa.py` is labeled Audio-JEPA but is **not** a true JEPA. It is a `google/vit-base-patch16-224` ViT (pretrained on ImageNet) with a Conv2d patch embedding for mel-spectrograms. An EMA target encoder exists in code but is never called during inference. There is no predictor network, no masking strategy, and no JEPA training objective. It will produce valid audio embeddings (ViT is powerful), but is not the self-supervised audio foundation model the architecture intends.

### Finding 2: No Training Infrastructure
There are no training scripts, no data downloading utilities, and no data mixing pipeline. The project is a well-designed inference-ready skeleton, not a trained system.

### Finding 3: WavJEPA-Nat-Base is the Right Encoder
- `labhamlet/wavjepa-nat-base` on HuggingFace
- True JEPA architecture trained on AudioSet + noisy/reverberant simulated scenes
- Specifically designed for noisy and reverberant environments (SNR 5–40 dB)
- 0.2B parameters, MIT license, active maintenance, Transformers-compatible
- Sony CSL's Audio-JEPA has no checkpoints (deleted by author, project abandoned)

---

## Draft: RESEARCH_GUIDE.md

This is the full content of the markdown file to be created. Review and approve before I write it.

---

```markdown
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
- **Noisy audio** (what was heard, even if mostly noise)
- **Industrial sensor readings** (temperature, pressure, vibration — contextual clues about machine state)

These two signals are fused to infer the most likely intended safety command, paired with a confidence score and a structured safety alert.

**Key Design Principle:** Never guess silently. Every prediction comes with calibrated confidence and uncertainty. Low confidence triggers human-in-the-loop verification.

---

## 2. System Architecture

The system has two operating modes:

### Base Mode (Audio-Only)
```
Noisy Waveform
    → Preprocessing (mel-spectrogram or raw waveform)
    → Audio Encoder
    → Mean Pooling → [B, embed_dim]
    → Command Classifier (MC Dropout)
    → Command + Confidence + Uncertainty
```

### Full Mode (Multimodal)
```
Noisy Waveform
    → Audio Encoder → [B, N_patches, embed_dim]  ─┐
                                                    ├─→ Cross-Attention Fusion → [B, 512]
Sensor Time-Series                                  │       → Command Classifier
    → Sensor Encoder → [B, 256] ───────────────────┘           → MC Dropout Inference
                                                            → Command + Confidence + Uncertainty
                                                            → Clean Audio Synthesis (AudioGen)
                                                            → LLM Safety Alert (OpenAI)
```

### Module Map

| Module | File | Purpose |
|---|---|---|
| Audio Encoder | `src/encoders/audio_jepa.py` | Encode noisy speech |
| Sensor Encoder | `src/encoders/sensor_encoder.py` | Encode time-series sensor data |
| Fusion | `src/fusion/cross_attention_fusion.py` | Cross-modal attention (speech as query, sensors as context) |
| Classifier | `src/classification/command_classifier.py` | 2-layer MLP + MC Dropout |
| Synthesizer | `src/synthesis/clean_audio_synthesizer.py` | AudioGen text-to-audio |
| Alert | `src/alert/llm_alert.py` | LLM-based structured safety alert |
| Preprocessing | `src/utils/preprocessing.py` | Waveform → log-mel spectrogram |
| Vocabulary | `src/utils/vocab.py` | 20 safety command strings |
| Pipeline | `src/pipeline.py` | Orchestrates full inference |

### Command Vocabulary (20 Commands)

```
0: stop the machine        10: call supervisor
1: emergency shutdown      11: check sensor
2: evacuate immediately    12: restart system
3: reduce speed            13: halt conveyor
4: increase pressure       14: start conveyor
5: decrease pressure       15: fire alarm
6: open valve              16: chemical leak alert
7: close valve             17: electrical hazard
8: activate safety lock    18: all clear
9: release safety lock     19: unknown command
```

---

## 3. Current Implementation Status

| Component | Status | Notes |
|---|---|---|
| Audio encoder (skeleton) | ✅ Exists | Uses ViT, NOT true JEPA — see Section 4 |
| Sensor encoder | ✅ Exists | Transformer with positional encoding, solid design |
| Cross-attention fusion | ✅ Exists | 8-head attention, speech-as-query pattern |
| Command classifier + MC Dropout | ✅ Exists | Temperature scaling + entropy uncertainty |
| Clean audio synthesis | ✅ Exists | Uses `facebook/audiogen-medium` |
| LLM safety alert | ✅ Exists | OpenAI-compatible interface |
| Inference script | ✅ Exists | `run_inference.py`, CLI-ready |
| Dataset | ❌ Missing | Strategy documented, no data downloaded |
| Data mixing pipeline | ❌ Missing | Speech + noise mixing code needed |
| Training script (supervised) | ❌ Missing | Fine-tuning fusion + classifier |
| Self-supervised pretraining | ❌ Missing | JEPA pretraining loop absent |
| Evaluation/metrics code | ❌ Missing | Accuracy, F1, ECE not implemented |
| True JEPA audio encoder | ❌ Not implemented | Recommend replacing — see Sections 4–6 |

---

## 4. Critical Finding: The Audio-JEPA Is Not JEPA

### What Is Currently in `src/encoders/audio_jepa.py`

The `AudioJEPAEncoder` class:
- Loads `google/vit-base-patch16-224` — an image Vision Transformer pretrained on ImageNet
- Uses a `Conv2d` patch embedding layer to convert mel-spectrograms into patch tokens
- Creates an EMA copy of the ViT called `target_encoder` (frozen)
- The `forward()` method only calls the context encoder (the main ViT)
- `target_encoder` is **never called during inference**
- `update_target()` is defined but never called in the pipeline

**In short:** It is a ViT feature extractor with an unused EMA copy attached. It will produce reasonable embeddings (ViT is capable), but it is not trained as JEPA and does not have JEPA's noise robustness properties.

### What a True Audio-JEPA Actually Is

A JEPA (Joint-Embedding Predictive Architecture) has four essential components:

1. **Masking strategy** — randomly mask regions of the input (for audio: mask temporal segments of the spectrogram or waveform)
2. **Context encoder** — encodes the unmasked (visible) regions
3. **Target encoder** — encodes the full input with EMA (no gradients, momentum-updated)
4. **Predictor** — takes context encoder output and predicts the target encoder's representation of the masked regions

Training objective: minimize the distance (e.g., L2 or cosine) between predictor output and target encoder output **in embedding space**. No pixel reconstruction, no contrastive negatives.

The current code has components 3 (partially) but is missing components 1, 2 in the masked sense, and entirely missing component 4 (the predictor).

### Why This Matters

JEPA's strength for noisy audio comes from:
- Learning **semantic** representations, not acoustic reconstruction
- The context encoder sees only partial information (like a noisy partial signal) and must predict the full semantic embedding
- This trains the model to be robust to missing/corrupted information — exactly the scenario of 100+ dB industrial noise

Without true JEPA training, the encoder has ImageNet-ViT biases and no noise robustness training specific to audio.

---

## 5. Audio Encoder Options Evaluated

### Option A: Sony CSL audio-representations
- **Repository:** [github.com/SonyCSLParis/audio-representations](https://github.com/SonyCSLParis/audio-representations)
- **True JEPA:** Yes — masked patch prediction on mel-spectrograms
- **Pretrained weights:** ❌ Author accidentally deleted them and has left the project
- **Maintenance:** ❌ Inactive — author states "I no longer work on this topic"
- **Verdict:** Use as a reference for architecture design only. Cannot be used for production research.

### Option B: WavJEPA (labhamlet)
- **Repository:** [github.com/labhamlet/wavjepa](https://github.com/labhamlet/wavjepa)
- **True JEPA:** Yes — raw waveform-based JEPA
- **Architecture:** Wav2Vec 2.0 feature encoder + ViT context encoder + ViT target encoder (EMA) + ViT predictor
- **Pretrained weights:** ✅ Available on HuggingFace (`labhamlet/wavjepa-base`)
- **Maintenance:** ✅ Active — updated November 2025
- **Paper:** arXiv:2509.23238

Two variants:
- `labhamlet/wavjepa-base` — 0.2B, trained on AudioSet only
- `labhamlet/wavjepa-nat-base` — 0.2B, trained on AudioSet + noisy/reverberant scenes ← recommended

---

## 6. Recommended Encoder: WavJEPA-Nat-Base

**HuggingFace model ID:** `labhamlet/wavjepa-nat-base`

### Why This Model

1. **Built for noise robustness** — trained on 85,000 simulated binaural room impulse responses (RIRs) from SoundSpaces2.0 and MatterPort3D environments, overlaid with WHAMR! noise at SNR 5–40 dB. This is the closest training condition to an industrial acoustic environment.

2. **True JEPA architecture** — the model learns semantic embeddings without reconstruction loss, making it robust to partial/corrupted information

3. **Pretrained weights available** — unlike Sony CSL's repo, this model is directly downloadable and usable with HuggingFace Transformers

4. **Raw waveform input** — no mel-spectrogram preprocessing needed; simplifies the pipeline and removes a potential information bottleneck

5. **Efficient** — 0.2B parameters; suitable for research compute budgets

6. **MIT licensed** — compatible with research and commercial use

### What Changes in the Pipeline

Replacing `AudioJEPAEncoder` with WavJEPA-Nat-Base requires three pipeline adjustments:

1. **Input format change:** Remove `waveform_to_mel()` preprocessing step. WavJEPA takes raw 16kHz waveforms directly.
2. **Output dimension change:** WavJEPA produces `[B, 200, 768]` for 2-second audio (200 time steps × 768 features). The fusion module already expects `[B, N_patches, 768]`, so it is compatible.
3. **Encoder loading:** Replace `ViTModel.from_pretrained('google/vit-base-patch16-224')` with `AutoModel.from_pretrained('labhamlet/wavjepa-nat-base')`.

The `SensorEncoder`, `CrossAttentionFusion`, `CommandClassifier`, and downstream modules remain unchanged.

---

## 7. Dataset Strategy

### Required Data Sources

#### Clean Speech Commands
| Dataset | Type | Size | Access |
|---|---|---|---|
| Google Speech Commands v2 | Short command words | ~105k clips, 35 classes | HuggingFace: `google/speech_commands` |
| Fluent Speech Corpus | Phrase-level commands | ~30k utterances | Kaggle: `tommyngx/fluent-speech-corpus` |

**Note:** Neither dataset contains exact safety command vocabulary. Closest matches: "stop", "go", "yes", "no" from Speech Commands; intent-mapped phrases from Fluent Speech. Custom recordings of the 20 safety commands may be necessary for best performance.

#### Industrial Background Noise
| Dataset | Type | Access |
|---|---|---|
| MS-SNSD | Factory, machinery, HVAC noise files | GitHub: `microsoft/MS-SNSD` |
| MIMII | Real malfunctioning industrial machinery audio | GitHub: `MIMII-hitachi/mimii_baseline` |
| ESC-50 | Environmental sounds (filter industrial categories) | GitHub: `karolpiczak/ESC-50` |

#### Industrial Sensor Time-Series
| Dataset | Type | Access |
|---|---|---|
| SKAB | Real physical sensor readings (water pump) | GitHub: `waico/SKAB` |
| C-MAPSS | Turbofan engine degradation sensor streams | NASA PCOE data repository |

### Data Mixing Strategy

1. Take one clean speech command clip
2. Take one sensor segment aligned to a plausible machine state for that command
3. Mix speech + industrial noise at target SNR: {20, 10, 0, -5, -10, -20} dB
4. Optionally apply room impulse response convolution for reverberation
5. Store as `(noisy_audio, sensor_window, command_label, snr_db)` tuple
6. Record metadata: `audio_file, sensor_file, snr_db, command_label, scenario_type`

Target dataset size for meaningful training: ~10,000 labeled pairs minimum; 50,000+ preferred.

---

## 8. Training Roadmap

This section describes what needs to be built and in what order. Each step includes an agent prompt you can use to have an AI agent implement that step.

### Step 0: Environment Setup

**What:** Install all dependencies and verify GPU access.

**Agent prompt:**
```
Set up the Python environment for the AthenAI project at /home/naveen/wso2/Multimodal_Intent_Reconstruction_for_Safety_Critical_Communication_in_Extreme_Acoustic_Environments.

Install requirements from requirements.txt. Also install: huggingface-hub, datasets, accelerate.
Verify that torch is installed with CUDA support by checking torch.cuda.is_available().
Download the WavJEPA-Nat-Base model card from HuggingFace (labhamlet/wavjepa-nat-base) to confirm the model loads correctly.
Report: Python version, torch version, CUDA availability, GPU name if available, and model load status.
```

---

### Step 1: Download and Prepare Datasets

**What:** Download Google Speech Commands v2, MS-SNSD noise files, and SKAB sensor data. Organize into a consistent directory structure.

**Agent prompt:**
```
Download and organize training datasets for the AthenAI safety command recognition project.

Target directory: /home/naveen/wso2/Multimodal_Intent_Reconstruction_for_Safety_Critical_Communication_in_Extreme_Acoustic_Environments/data/

Download the following:
1. Google Speech Commands v2 from HuggingFace (google/speech_commands, version v0.02) — use the HuggingFace datasets library. Save audio files organized by label.
2. MS-SNSD noise dataset from GitHub (microsoft/MS-SNSD) — download just the noise audio files in the noise_train/ folder.
3. SKAB sensor dataset from GitHub (waico/SKAB) — download the CSV files.

Create this directory structure:
  data/
    speech_commands/   (organized by command label)
    noise/             (all noise .wav files flat)
    sensors/           (SKAB CSV files)
    mixed/             (empty — for the generated training pairs)
    metadata.csv       (empty — to be populated)

Report: number of audio files per speech class, number of noise files, number of sensor CSV files.
```

---

### Step 2: Build Data Mixing Pipeline

**What:** Create a script that combines clean speech + noise at variable SNRs and aligns with sensor windows.

**Agent prompt:**
```
Create a data preparation script for the AthenAI project at /home/naveen/wso2/Multimodal_Intent_Reconstruction_for_Safety_Critical_Communication_in_Extreme_Acoustic_Environments.

The script should be placed at: scripts/prepare_dataset.py

What it must do:
1. Load clean speech audio clips from data/speech_commands/ (use torchaudio, resample to 16kHz, mono)
2. Load noise clips from data/noise/ (resample to 16kHz, mono)
3. Mix speech + noise at SNR levels: [20, 10, 0, -5, -10, -20] dB using standard SNR mixing formula
4. Load a random 128-timestep window from a SKAB CSV file as the sensor input (8 channels: normalize each column to [0,1])
5. Save mixed audio to data/mixed/ as .wav files
6. Save sensor windows to data/mixed/ as .npy files
7. Write metadata.csv with columns: audio_file, sensor_file, command_label, command_idx, snr_db, split (train/val/test)

Use 80/10/10 train/val/test split.
Map speech command labels to the AthenAI vocabulary in src/utils/vocab.py — use "unknown command" (idx 19) for commands not in the vocabulary.
Generate at least 1000 mixed samples for initial testing.

The vocabulary is defined in src/utils/vocab.py as COMMAND_VOCAB (a list of 20 strings).
```

---

### Step 3: Replace Audio Encoder with WavJEPA-Nat-Base

**What:** Swap the current `AudioJEPAEncoder` (ImageNet ViT) for the pretrained `labhamlet/wavjepa-nat-base`. Adjust the pipeline to skip mel-spectrogram preprocessing when using WavJEPA.

**Agent prompt:**
```
Modify the audio encoder in the AthenAI project at /home/naveen/wso2/Multimodal_Intent_Reconstruction_for_Safety_Critical_Communication_in_Extreme_Acoustic_Environments.

Current file: src/encoders/audio_jepa.py
Current class: AudioJEPAEncoder

Replace the encoder with WavJEPA-Nat-Base from HuggingFace (labhamlet/wavjepa-nat-base).
The new encoder should:
1. Load the model using AutoModel.from_pretrained('labhamlet/wavjepa-nat-base')
2. Load the feature extractor using AutoFeatureExtractor.from_pretrained('labhamlet/wavjepa-nat-base')
3. Accept raw waveforms at 16kHz as input: shape [B, T]
4. Return patch-level embeddings: shape [B, N, 768] where N is the number of time steps
5. Be named WavJEPAEncoder and placed in src/encoders/audio_jepa.py (replacing the old class)

Also update src/pipeline.py to:
- Import WavJEPAEncoder instead of AudioJEPAEncoder
- Skip the waveform_to_mel() preprocessing step before the audio encoder
- Pass raw waveforms directly to the encoder

Also update src/encoders/__init__.py to export WavJEPAEncoder.

Do not change SensorEncoder, CrossAttentionFusion, CommandClassifier, or any other modules.
```

---

### Step 4: Write Training Script

**What:** Create a supervised training script that trains the fusion layer and classifier while keeping the audio encoder frozen.

**Agent prompt:**
```
Create a training script for the AthenAI project at /home/naveen/wso2/Multimodal_Intent_Reconstruction_for_Safety_Critical_Communication_in_Extreme_Acoustic_Environments.

Place it at: scripts/train.py

The script must:
1. Load the metadata.csv from data/mixed/metadata.csv
2. Build a PyTorch Dataset class that:
   - Loads audio (.wav) with torchaudio, resamples to 16kHz, converts to mono
   - Loads sensor windows (.npy) — shape [128, 8]
   - Returns (audio_tensor, sensor_tensor, command_idx) tuples
3. Use DataLoader with batch_size=32, shuffle for train, no shuffle for val/test
4. Build the model:
   - WavJEPAEncoder from src/encoders/audio_jepa.py — FREEZE all parameters
   - SensorEncoder from src/encoders/sensor_encoder.py — trainable
   - CrossAttentionFusion from src/fusion/cross_attention_fusion.py — trainable
   - CommandClassifier from src/classification/command_classifier.py — trainable
5. Optimizer: AdamW with lr=3e-4 on trainable parameters only
6. Loss: CrossEntropyLoss
7. Train for 50 epochs with early stopping (patience=10 on val loss)
8. Log: train loss, val loss, val accuracy every epoch
9. Save best model checkpoint to checkpoints/best_model.pt
10. At end of training, print test accuracy and per-class F1 score

Use command line arguments: --epochs, --batch_size, --lr, --mode (base or full)
In base mode: skip sensor encoder and fusion, use mean-pooled WavJEPA output directly
In full mode: use full multimodal pipeline

The project uses PyTorch — do not use any training framework like Lightning unless needed.
```

---

### Step 5: Evaluate Uncertainty Calibration

**What:** Measure how well the confidence scores match actual accuracy (calibration). A well-calibrated model that says 80% confident should be right ~80% of the time.

**Agent prompt:**
```
Create an evaluation script for the AthenAI project at /home/naveen/wso2/Multimodal_Intent_Reconstruction_for_Safety_Critical_Communication_in_Extreme_Acoustic_Environments.

Place it at: scripts/evaluate.py

The script must:
1. Load the trained model from checkpoints/best_model.pt
2. Load the test split from data/mixed/metadata.csv
3. Run inference with MC Dropout (n_mc=20) on all test samples
4. Compute and report:
   a. Overall accuracy
   b. Per-class accuracy and F1 score (for all 20 commands)
   c. Accuracy broken down by SNR level (20, 10, 0, -5, -10, -20 dB)
   d. Expected Calibration Error (ECE) — reliability of confidence scores
   e. Reliability diagram (confidence vs accuracy plot, save as eval_reliability.png)
   f. Uncertainty histogram (distribution of entropy scores, save as eval_uncertainty.png)
5. Save all metrics to eval_results.json

For ECE calculation: bin predictions into 10 equal-width confidence bins, compute |avg_confidence - accuracy| per bin, weight by bin size.

Use the mc_dropout_inference function from src/classification/command_classifier.py for predictions.
```

---

## 9. Known Gaps and Next Steps

### Immediate Gaps (Blocking Training)
- No training data exists yet (Step 1 + 2 above)
- Audio encoder must be replaced before training is meaningful (Step 3)
- No training script exists (Step 4)

### Research Quality Improvements (After Initial Training)
- **Domain-adaptive fine-tuning of WavJEPA:** If custom industrial audio recordings are available, fine-tune WavJEPA-Nat-Base on them using the JEPA objective from the wavjepa repository
- **Larger command vocabulary:** The 20-command vocabulary may not cover all real industrial scenarios; consider expanding with domain expert input
- **Sensor modality ablation:** Systematically measure how much each SNR level benefits from sensor fusion vs. audio-only
- **Hard negative mining:** As training matures, focus the classifier on easily confused command pairs (e.g., "open valve" vs "close valve")

### Architectural Improvements to Consider
- **TimeMAE or TS2Vec for sensor encoding:** The current sensor encoder is a custom Transformer; replacing it with a pretrained time-series foundation model (TimeMAE is already referenced in Resources/) could improve performance
- **Learnable SNR conditioning:** Pass estimated SNR as an input feature to the classifier to allow the model to modulate its uncertainty based on known noise level
- **Ensemble uncertainty:** Combine MC Dropout entropy with temperature-scaled softmax entropy for more reliable uncertainty estimates

### Safety Considerations
- All confidence thresholds used in alert generation (`< 0.4`, `< 0.5`) are heuristics — calibrate them empirically from the ECE evaluation
- For deployment, deterministic templated alerts are safer than LLM-generated alerts (LLMs can hallucinate)
- Always maintain a human-in-the-loop verification step for any predicted command that triggers irreversible actions

---

*This document was generated as part of the AthenAI research planning process. Implementation prompts in Section 8 are designed to be executed by an AI coding agent after research review and approval.*
```

---

## Implementation Action (After User Approval)

Write the above markdown content to:
`/home/naveen/wso2/Multimodal_Intent_Reconstruction_for_Safety_Critical_Communication_in_Extreme_Acoustic_Environments/RESEARCH_GUIDE.md`
