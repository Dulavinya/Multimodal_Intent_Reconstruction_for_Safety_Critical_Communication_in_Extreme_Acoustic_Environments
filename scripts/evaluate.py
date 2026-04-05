"""Evaluation script for AthenAI.

Loads the best checkpoint, evaluates on the test split, computes accuracy,
per-class metrics, calibration, and uncertainty summaries, then saves metrics
and plots under the project root.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from torch.utils.data import DataLoader, Dataset

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_symbol(module_name: str, file_path: Path, symbol_name: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {symbol_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, symbol_name)


CommandClassifier = load_symbol(
    "athenai_command_classifier_eval",
    PROJECT_ROOT / "src" / "classification" / "command_classifier.py",
    "CommandClassifier",
)
mc_dropout_inference = load_symbol(
    "athenai_command_classifier_eval",
    PROJECT_ROOT / "src" / "classification" / "command_classifier.py",
    "mc_dropout_inference",
)
IntentQueryDecoder = load_symbol(
    "athenai_intent_query_decoder_eval",
    PROJECT_ROOT / "src" / "classification" / "intent_query_decoder.py",
    "IntentQueryDecoder",
)
mc_dropout_inference_iqd = load_symbol(
    "athenai_intent_query_decoder_eval",
    PROJECT_ROOT / "src" / "classification" / "intent_query_decoder.py",
    "mc_dropout_inference_iqd",
)
WavJEPAEncoder = load_symbol(
    "athenai_audio_jepa_eval",
    PROJECT_ROOT / "src" / "encoders" / "audio_jepa.py",
    "WavJEPAEncoder",
)
SensorEncoder = load_symbol(
    "athenai_sensor_encoder_eval",
    PROJECT_ROOT / "src" / "encoders" / "sensor_encoder.py",
    "SensorEncoder",
)
CrossAttentionFusion = load_symbol(
    "athenai_cross_attention_fusion_eval",
    PROJECT_ROOT / "src" / "fusion" / "cross_attention_fusion.py",
    "CrossAttentionFusion",
)
COMMAND_VOCAB = load_symbol(
    "athenai_vocab_eval",
    PROJECT_ROOT / "src" / "utils" / "vocab.py",
    "COMMAND_VOCAB",
)


TARGET_SAMPLE_RATE = 16000
TARGET_NUM_SAMPLES = 32000
NUM_SENSORS = 8
SNR_LEVELS = [20, 10, 0, -5, -10, -20]
N_BINS = 10


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pad_or_trim_waveform(waveform: torch.Tensor, target_length: int = TARGET_NUM_SAMPLES) -> torch.Tensor:
    waveform = waveform.flatten()
    current_length = waveform.shape[-1]
    if current_length > target_length:
        waveform = waveform[:target_length]
    elif current_length < target_length:
        waveform = F.pad(waveform, (0, target_length - current_length))
    return waveform.contiguous()


class SafetyCommandDataset(Dataset):
    def __init__(self, metadata_path: Path, split: str, data_dir: Path):
        self.data_dir = data_dir
        self.rows: List[Dict[str, str]] = []

        with metadata_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row["split"] == split:
                    self.rows.append(row)

        if not self.rows:
            raise ValueError(f"No samples found for split '{split}' in {metadata_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        row = self.rows[index]

        audio_path = self.data_dir / row["audio_file"]
        sensor_path = self.data_dir / row["sensor_file"]

        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0)
        if sample_rate != TARGET_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform.unsqueeze(0), sample_rate, TARGET_SAMPLE_RATE).squeeze(0)
        waveform = pad_or_trim_waveform(waveform, TARGET_NUM_SAMPLES).to(torch.float32)

        sensor = np.load(sensor_path).astype(np.float32)
        if sensor.shape != (128, NUM_SENSORS):
            raise ValueError(f"Expected sensor shape (128, {NUM_SENSORS}), got {sensor.shape} for {sensor_path}")
        sensor_tensor = torch.from_numpy(sensor)

        command_idx = int(row["command_idx"])
        snr_db = int(round(float(row["snr_db"])))
        return waveform, sensor_tensor, command_idx, snr_db


class AthenAIModel(nn.Module):
    def __init__(self, mode: str, decoder_type: str = "fc"):
        super().__init__()
        self.mode = mode
        self.decoder_type = decoder_type
        self.audio_encoder = WavJEPAEncoder()
        for parameter in self.audio_encoder.parameters():
            parameter.requires_grad = False

        if mode == "full":
            self.sensor_encoder = SensorEncoder(n_sensors=NUM_SENSORS)
            self.fusion = CrossAttentionFusion()
            self.classifier = CommandClassifier(input_dim=512)
        else:
            if decoder_type == "iqd":
                self.classifier = IntentQueryDecoder(
                    n_commands=len(COMMAND_VOCAB),
                    query_dim=768,
                    n_heads=8,
                    dropout=0.3,
                )
            else:
                self.classifier = CommandClassifier(input_dim=768)

    def forward_features(self, audio: torch.Tensor, sensor: torch.Tensor | None = None) -> torch.Tensor:
        with torch.no_grad():
            speech_emb = self.audio_encoder(audio)

        if self.mode == "full":
            if sensor is None:
                raise ValueError("sensor tensor is required in full mode")
            sensor_emb = self.sensor_encoder(sensor)
            return self.fusion(speech_emb, sensor_emb)

        if self.decoder_type == "iqd":
            # Return full sequence for IQD — do NOT mean-pool
            return speech_emb

        return speech_emb.mean(dim=1)


def build_test_loader(data_dir: Path, batch_size: int) -> DataLoader:
    metadata_path = data_dir / "metadata.csv"
    test_dataset = SafetyCommandDataset(metadata_path=metadata_path, split="test", data_dir=data_dir)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())


def load_checkpoint(path: Path, model: AthenAIModel, device: torch.device) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def compute_ece(confidences: np.ndarray, predictions: np.ndarray, targets: np.ndarray, n_bins: int = N_BINS):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(confidences)
    bin_stats = []
    ece = 0.0

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)

        bin_count = int(mask.sum())
        if bin_count == 0:
            bin_stats.append(
                {
                    "bin_lower": float(lower),
                    "bin_upper": float(upper),
                    "count": 0,
                    "accuracy": None,
                    "mean_confidence": None,
                    "gap": None,
                    "weighted_gap": 0.0,
                }
            )
            continue

        bin_accuracy = float((predictions[mask] == targets[mask]).mean())
        bin_confidence = float(confidences[mask].mean())
        gap = abs(bin_confidence - bin_accuracy)
        weighted_gap = (bin_count / total) * gap
        ece += weighted_gap
        bin_stats.append(
            {
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "count": bin_count,
                "accuracy": bin_accuracy,
                "mean_confidence": bin_confidence,
                "gap": gap,
                "weighted_gap": weighted_gap,
            }
        )

    return ece, bin_stats


def plot_reliability(bin_stats: List[Dict[str, object]], output_path: Path) -> None:
    bin_centers = []
    accuracies = []
    counts = []
    for stats in bin_stats:
        if stats["count"] == 0:
            continue
        lower = float(stats["bin_lower"])
        upper = float(stats["bin_upper"])
        bin_centers.append((lower + upper) / 2.0)
        accuracies.append(float(stats["accuracy"]))
        counts.append(int(stats["count"]))

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.plot(bin_centers, accuracies, marker="o", linewidth=2, label="Model accuracy")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_uncertainty_histogram(entropies_by_snr: Dict[int, List[float]], output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    colors = {
        20: "tab:green",
        10: "tab:blue",
        0: "tab:orange",
        -5: "tab:red",
        -10: "tab:purple",
        -20: "tab:brown",
    }

    bins = 20
    for snr in SNR_LEVELS:
        values = entropies_by_snr.get(snr, [])
        if not values:
            continue
        plt.hist(values, bins=bins, alpha=0.45, label=f"SNR {snr} dB", color=colors.get(snr))

    plt.xlabel("Entropy")
    plt.ylabel("Count")
    plt.title("Entropy Distribution by SNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AthenAI on the test split")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "best_model.pt")
    parser.add_argument("--mode", type=str, choices=("base", "full"), default="full")
    parser.add_argument("--decoder", type=str, choices=("fc", "iqd"), default="fc",
                        help="'iqd' for Intent Query Decoder, 'fc' for FC baseline")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    set_seed(42)

    data_dir = PROJECT_ROOT / "data" / "mixed"
    test_loader = build_test_loader(data_dir, args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AthenAIModel(mode=args.mode, decoder_type=args.decoder).to(device)
    load_checkpoint(args.checkpoint, model, device)
    model.eval()

    all_targets: List[int] = []
    all_predictions: List[int] = []
    all_confidences: List[float] = []
    all_entropies: List[float] = []
    all_snrs: List[int] = []

    with torch.no_grad():
        for audio, sensor, targets, snr_db in test_loader:
            audio = audio.to(device, non_blocking=True)
            sensor = sensor.to(device, non_blocking=True)

            if model.mode == "full":
                features = model.forward_features(audio, sensor)
            else:
                features = model.forward_features(audio)

            if model.decoder_type == "iqd":
                pred_cmd, confidence, uncertainty = mc_dropout_inference_iqd(
                    model.classifier,
                    features,
                    n_passes=20,
                )
            else:
                pred_cmd, confidence, uncertainty = mc_dropout_inference(
                    model.classifier,
                    features,
                    n_passes=20,
                )

            all_targets.extend(targets.tolist())
            all_predictions.extend(pred_cmd.detach().cpu().tolist())
            all_confidences.extend(confidence.detach().cpu().tolist())
            all_entropies.extend(uncertainty.detach().cpu().tolist())
            all_snrs.extend(snr_db.tolist())

    targets_np = np.asarray(all_targets, dtype=np.int64)
    predictions_np = np.asarray(all_predictions, dtype=np.int64)
    confidences_np = np.asarray(all_confidences, dtype=np.float64)
    entropies_np = np.asarray(all_entropies, dtype=np.float64)
    snrs_np = np.asarray(all_snrs, dtype=np.int64)

    overall_accuracy = float(accuracy_score(targets_np, predictions_np))
    report_dict = classification_report(
        targets_np,
        predictions_np,
        labels=list(range(len(COMMAND_VOCAB))),
        target_names=COMMAND_VOCAB,
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        targets_np,
        predictions_np,
        labels=list(range(len(COMMAND_VOCAB))),
        target_names=COMMAND_VOCAB,
        zero_division=0,
    )

    ece, bin_stats = compute_ece(confidences_np, predictions_np, targets_np, n_bins=N_BINS)

    snr_metrics: Dict[str, Dict[str, float]] = {}
    entropies_by_snr: Dict[int, List[float]] = {snr: [] for snr in SNR_LEVELS}
    for snr in SNR_LEVELS:
        mask = snrs_np == snr
        if mask.any():
            snr_accuracy = float((predictions_np[mask] == targets_np[mask]).mean())
            snr_confidence = float(confidences_np[mask].mean())
            snr_entropy = float(entropies_np[mask].mean())
            snr_metrics[str(snr)] = {
                "accuracy": snr_accuracy,
                "mean_confidence": snr_confidence,
                "mean_entropy": snr_entropy,
                "count": int(mask.sum()),
            }
            entropies_by_snr[snr] = entropies_np[mask].tolist()
        else:
            snr_metrics[str(snr)] = {
                "accuracy": None,
                "mean_confidence": None,
                "mean_entropy": None,
                "count": 0,
            }

    metrics = {
        "checkpoint": str(args.checkpoint),
        "mode": args.mode,
        "num_test_samples": int(len(targets_np)),
        "overall_top1_accuracy": overall_accuracy,
        "ece": float(ece),
        "per_class": {
            name: {
                "precision": float(values["precision"]),
                "recall": float(values["recall"]),
                "f1_score": float(values["f1-score"]),
                "support": int(values["support"]),
            }
            for name, values in report_dict.items()
            if name in COMMAND_VOCAB
        },
        "snr_metrics": snr_metrics,
        "ece_bins": bin_stats,
    }

    results_path = PROJECT_ROOT / "eval_results.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    plot_reliability(bin_stats, PROJECT_ROOT / "eval_reliability.png")
    plot_uncertainty_histogram(entropies_by_snr, PROJECT_ROOT / "eval_uncertainty.png")

    print("Overall top-1 accuracy:", overall_accuracy)
    print("\nClassification report:\n")
    print(report_text)
    print("\nAccuracy and confidence by SNR:")
    for snr in SNR_LEVELS:
        stats = snr_metrics[str(snr)]
        print(
            f"  SNR {snr:+3d} dB: accuracy={stats['accuracy']}, mean_confidence={stats['mean_confidence']}, mean_entropy={stats['mean_entropy']}"
        )
    print(f"\nECE: {ece:.6f}")
    print(f"Saved metrics to {results_path}")
    print(f"Saved reliability plot to {PROJECT_ROOT / 'eval_reliability.png'}")
    print(f"Saved uncertainty plot to {PROJECT_ROOT / 'eval_uncertainty.png'}")


if __name__ == "__main__":
    main()