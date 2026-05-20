"""
Ablation comparison script for AthenAI — FC baseline vs Intent Query Decoder (IQD).

Loads both checkpoints, evaluates on the test split, and produces a side-by-side
comparison table across all SNR levels with accuracy, ECE, and uncertainty metrics.

Usage:
    python scripts/ablate.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Dynamic symbol loading (mirrors train.py) ─────────────────────────────────

def load_symbol(module_name: str, file_path: Path, symbol_name: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, symbol_name)


CommandClassifier   = load_symbol("cc",  PROJECT_ROOT / "src/classification/command_classifier.py",   "CommandClassifier")
IntentQueryDecoder  = load_symbol("iqd", PROJECT_ROOT / "src/classification/intent_query_decoder.py", "IntentQueryDecoder")
mc_dropout_inference= load_symbol("cc",  PROJECT_ROOT / "src/classification/command_classifier.py",   "mc_dropout_inference")
mc_dropout_iqd      = load_symbol("iqd", PROJECT_ROOT / "src/classification/intent_query_decoder.py", "mc_dropout_inference_iqd")
WavJEPAEncoder      = load_symbol("ae",  PROJECT_ROOT / "src/encoders/audio_jepa.py",                 "WavJEPAEncoder")
COMMAND_VOCAB       = load_symbol("v",   PROJECT_ROOT / "src/utils/vocab.py",                          "COMMAND_VOCAB")

TARGET_SR   = 16000
TARGET_LEN  = 32000
NUM_SENSORS = 8
SNR_LEVELS  = [20, 10, 0, -5, -10, -20]
N_BINS      = 10
N_MC        = 20


# ── Dataset ───────────────────────────────────────────────────────────────────

def pad_or_trim(wav: torch.Tensor, n: int = TARGET_LEN) -> torch.Tensor:
    wav = wav.flatten()
    if wav.size(0) > n:
        return wav[:n].contiguous()
    return F.pad(wav, (0, n - wav.size(0))).contiguous()


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.rows: List[Dict] = []
        with (data_dir / "metadata.csv").open(newline="") as f:
            for row in csv.DictReader(f):
                if row["split"] == "test":
                    self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        wav, sr = torchaudio.load(self.data_dir / row["audio_file"])
        wav = wav.mean(0)
        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, TARGET_SR).squeeze(0)
        wav = pad_or_trim(wav).float()
        sensor = torch.from_numpy(np.load(self.data_dir / row["sensor_file"]).astype(np.float32))
        return wav, sensor, int(row["command_idx"]), int(round(float(row["snr_db"])))


# ── Model wrappers ────────────────────────────────────────────────────────────

class FCModel(torch.nn.Module):
    """Baseline: WavJEPA → mean-pool → FC classifier."""
    def __init__(self):
        super().__init__()
        self.audio_encoder = WavJEPAEncoder(freeze_encoder=True)
        self.classifier = CommandClassifier(input_dim=768)
        for p in self.audio_encoder.parameters():
            p.requires_grad = False

    def encode(self, wav): return self.audio_encoder(wav).mean(1)   # [B, 768]


class IQDModel(torch.nn.Module):
    """Intent Query Decoder: WavJEPA → full sequence → cross-attention classifier."""
    def __init__(self):
        super().__init__()
        self.audio_encoder = WavJEPAEncoder(freeze_encoder=True)
        self.classifier = IntentQueryDecoder(n_commands=len(COMMAND_VOCAB), query_dim=768, n_heads=8, dropout=0.3)
        for p in self.audio_encoder.parameters():
            p.requires_grad = False

    def encode(self, wav): return self.audio_encoder(wav)           # [B, N, 768]


# ── ECE ───────────────────────────────────────────────────────────────────────

def compute_ece(conf: np.ndarray, pred: np.ndarray, tgt: np.ndarray, n_bins: int = N_BINS) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(conf)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc = float((pred[mask] == tgt[mask]).mean())
        avg_c = float(conf[mask].mean())
        ece += (mask.sum() / total) * abs(avg_c - acc)
    return float(ece)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, decoder_type: str, loader, device) -> Dict:
    model.eval()
    all_targets, all_preds, all_conf, all_ent, all_snrs = [], [], [], [], []

    for wav, sensor, targets, snrs in loader:
        wav = wav.to(device)
        features = model.encode(wav)  # [B,768] or [B,N,768]

        if decoder_type == "fc":
            pred, conf, unc = mc_dropout_inference(model.classifier, features, n_passes=N_MC)
        else:
            pred, conf, unc = mc_dropout_iqd(model.classifier, features, n_passes=N_MC)

        all_targets.extend(targets.tolist())
        all_preds.extend(pred.cpu().tolist())
        all_conf.extend(conf.cpu().tolist())
        all_ent.extend(unc.cpu().tolist())
        all_snrs.extend(snrs.tolist())

    tgt = np.array(all_targets)
    prd = np.array(all_preds)
    cof = np.array(all_conf)
    ent = np.array(all_ent)
    snr = np.array(all_snrs)

    overall_acc = float(accuracy_score(tgt, prd))
    ece = compute_ece(cof, prd, tgt)

    snr_metrics = {}
    for s in SNR_LEVELS:
        mask = snr == s
        if mask.any():
            snr_metrics[s] = {
                "accuracy":        float((prd[mask] == tgt[mask]).mean()),
                "mean_confidence": float(cof[mask].mean()),
                "mean_entropy":    float(ent[mask].mean()),
                "count":           int(mask.sum()),
            }
        else:
            snr_metrics[s] = {"accuracy": None, "mean_confidence": None, "mean_entropy": None, "count": 0}

    return {"overall_accuracy": overall_acc, "ece": ece, "snr_metrics": snr_metrics}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = PROJECT_ROOT / "data" / "mixed"
    ckpt_dir = PROJECT_ROOT / "checkpoints"

    loader = torch.utils.data.DataLoader(
        TestDataset(data_dir), batch_size=32, shuffle=False, num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    results = {}

    # ── FC Baseline ──
    fc_ckpt = ckpt_dir / "best_fc.pt"
    if fc_ckpt.exists():
        print("\n── Evaluating FC Baseline ────────────────────────")
        fc_model = FCModel().to(device)
        ckpt = torch.load(fc_ckpt, map_location=device)
        fc_model.load_state_dict(ckpt["model_state_dict"])
        results["fc"] = evaluate(fc_model, "fc", loader, device)
        print(f"  Overall accuracy: {results['fc']['overall_accuracy']:.4f}")
        print(f"  ECE:              {results['fc']['ece']:.6f}")
    else:
        print(f"  [WARN] FC checkpoint not found at {fc_ckpt}, skipping.")

    # ── IQD ──
    iqd_ckpt = ckpt_dir / "best_iqd.pt"
    if iqd_ckpt.exists():
        print("\n── Evaluating Intent Query Decoder ───────────────")
        iqd_model = IQDModel().to(device)
        ckpt = torch.load(iqd_ckpt, map_location=device)
        iqd_model.load_state_dict(ckpt["model_state_dict"])
        results["iqd"] = evaluate(iqd_model, "iqd", loader, device)
        print(f"  Overall accuracy: {results['iqd']['overall_accuracy']:.4f}")
        print(f"  ECE:              {results['iqd']['ece']:.6f}")
    else:
        print(f"  [WARN] IQD checkpoint not found at {iqd_ckpt}, skipping.")

    # ── Comparison Table ──
    if "fc" in results and "iqd" in results:
        fc, iqd = results["fc"], results["iqd"]
        delta_overall = iqd["overall_accuracy"] - fc["overall_accuracy"]
        print("\n" + "═" * 70)
        print("  ABLATION COMPARISON: FC Baseline  vs  Intent Query Decoder (IQD)")
        print("═" * 70)
        print(f"  {'Metric':<30} {'FC':>10} {'IQD':>10} {'Δ (IQD-FC)':>12}")
        print(f"  {'-'*60}")
        print(f"  {'Overall Accuracy':<30} {fc['overall_accuracy']:>10.4f} {iqd['overall_accuracy']:>10.4f} {delta_overall:>+12.4f}")
        print(f"  {'ECE':<30} {fc['ece']:>10.6f} {iqd['ece']:>10.6f} {iqd['ece']-fc['ece']:>+12.6f}")
        print(f"  {'-'*60}")
        print(f"  {'SNR':<10} {'FC Acc':>12} {'IQD Acc':>12} {'Δ Acc':>12} {'IQD Entropy':>14}")
        for s in SNR_LEVELS:
            fc_s  = fc["snr_metrics"].get(s, {})
            iqd_s = iqd["snr_metrics"].get(s, {})
            fc_acc  = fc_s.get("accuracy")
            iqd_acc = iqd_s.get("accuracy")
            iqd_ent = iqd_s.get("mean_entropy")
            if fc_acc is None or iqd_acc is None:
                print(f"  {s:>+4} dB {'N/A':>12} {'N/A':>12}")
                continue
            delta = iqd_acc - fc_acc
            print(f"  {s:>+4} dB {fc_acc:>12.4f} {iqd_acc:>12.4f} {delta:>+12.4f} {iqd_ent:>14.4f}")
        print("═" * 70)

    # ── Save JSON ──
    out_path = PROJECT_ROOT / "ablation_results.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
