"""Supervised training script for AthenAI.

Supports two modes:
- base: WavJEPA -> mean pool -> classifier
- full: WavJEPA + SensorEncoder + CrossAttentionFusion -> classifier

The script expects data/mixed/metadata.csv and the generated audio/sensor files
under data/mixed/.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

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
    "athenai_command_classifier",
    PROJECT_ROOT / "src" / "classification" / "command_classifier.py",
    "CommandClassifier",
)
WavJEPAEncoder = load_symbol(
    "athenai_audio_jepa",
    PROJECT_ROOT / "src" / "encoders" / "audio_jepa.py",
    "WavJEPAEncoder",
)
SensorEncoder = load_symbol(
    "athenai_sensor_encoder",
    PROJECT_ROOT / "src" / "encoders" / "sensor_encoder.py",
    "SensorEncoder",
)
CrossAttentionFusion = load_symbol(
    "athenai_cross_attention_fusion",
    PROJECT_ROOT / "src" / "fusion" / "cross_attention_fusion.py",
    "CrossAttentionFusion",
)
COMMAND_VOCAB = load_symbol(
    "athenai_vocab",
    PROJECT_ROOT / "src" / "utils" / "vocab.py",
    "COMMAND_VOCAB",
)


TARGET_SAMPLE_RATE = 16000
TARGET_NUM_SAMPLES = 32000
NUM_SENSORS = 8


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

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        row = self.rows[index]

        audio_path = self.data_dir / row["audio_file"]
        sensor_path = self.data_dir / row["sensor_file"]

        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.ndim != 2:
            raise ValueError(f"Expected waveform with shape [C, T], got {tuple(waveform.shape)}")

        waveform = waveform.mean(dim=0)
        if sample_rate != TARGET_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform.unsqueeze(0), sample_rate, TARGET_SAMPLE_RATE).squeeze(0)
        waveform = pad_or_trim_waveform(waveform, TARGET_NUM_SAMPLES).to(torch.float32)

        sensor = np.load(sensor_path).astype(np.float32)
        if sensor.shape != (128, NUM_SENSORS):
            raise ValueError(f"Expected sensor shape (128, {NUM_SENSORS}), got {sensor.shape} for {sensor_path}")
        sensor_tensor = torch.from_numpy(sensor)

        command_idx = int(row["command_idx"])
        return waveform, sensor_tensor, command_idx


class AthenAIModel(nn.Module):
    def __init__(self, mode: str, freeze_audio: bool = True):
        super().__init__()
        self.mode = mode
        self.audio_encoder = WavJEPAEncoder(freeze_encoder=freeze_audio)

        if mode == "full":
            self.sensor_encoder = SensorEncoder(n_sensors=NUM_SENSORS)
            self.fusion = CrossAttentionFusion()
            self.classifier = CommandClassifier(input_dim=512)
        else:
            self.classifier = CommandClassifier(input_dim=768)

    def train(self, mode: bool = True):
        super().train(mode)
        # Even if unfreezing, keeping audio encoder batchnorm/dropout in eval can sometimes be useful,
        # but for true fine-tuning we should respect the freeze_encoder state.
        if hasattr(self.audio_encoder, "freeze_encoder") and self.audio_encoder.freeze_encoder:
            self.audio_encoder.eval()
        return self

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            speech_emb = self.audio_encoder(audio)
        return speech_emb

    def logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.classifier.fc(features)
        return logits / self.classifier.temperature

    def forward(self, audio: torch.Tensor, sensor: torch.Tensor | None = None) -> torch.Tensor:
        speech_emb = self.encode_audio(audio)

        if self.mode == "full":
            if sensor is None:
                raise ValueError("sensor tensor is required in full mode")
            sensor_emb = self.sensor_encoder(sensor)
            features = self.fusion(speech_emb, sensor_emb)
        else:
            features = speech_emb.mean(dim=1)

        return self.logits_from_features(features)


@dataclass
class EpochResult:
    loss: float
    accuracy: float


def build_dataloaders(data_dir: Path, batch_size: int) -> Dict[str, DataLoader]:
    metadata_path = data_dir / "metadata.csv"
    datasets = {
        split: SafetyCommandDataset(metadata_path=metadata_path, split=split, data_dir=data_dir)
        for split in ("train", "val", "test")
    }

    return {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available()),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available()),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available()),
    }


def run_epoch(
    model: AthenAIModel,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    desc: str = "",
) -> Tuple[float, float, List[int], List[int]]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    progress = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for audio, sensor, targets in progress:
        audio = audio.to(device, non_blocking=True)
        sensor = sensor.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            if model.mode == "full":
                logits = model(audio, sensor)
            else:
                logits = model(audio)
            loss = criterion(logits, targets)

            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(targets.detach().cpu().tolist())

        progress.set_postfix(loss=f"{loss.item():.4f}")

    average_loss = total_loss / max(1, len(all_targets))
    accuracy = float(accuracy_score(all_targets, all_preds))
    return average_loss, accuracy, all_preds, all_targets


def save_checkpoint(path: Path, model: AthenAIModel, optimizer: torch.optim.Optimizer, epoch: int, val_loss: float, mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "mode": mode,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(path: Path, model: AthenAIModel, device: torch.device) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised training for AthenAI")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--mode", type=str, choices=("base", "full"), default="full")
    parser.add_argument("--unfreeze_audio", action="store_true", help="Unfreeze the audio encoder for fine-tuning")
    args = parser.parse_args()

    warnings.filterwarnings(
        "ignore",
        message="The PyTorch API of MaskedTensors is in prototype stage and will change in the near future.*",
        category=UserWarning,
    )

    set_seed(42)

    project_root = PROJECT_ROOT
    data_dir = project_root / "data" / "mixed"
    checkpoint_path = project_root / "checkpoints" / "best_model.pt"

    dataloaders = build_dataloaders(data_dir, args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AthenAIModel(mode=args.mode, freeze_audio=not args.unfreeze_audio).to(device)

    if args.unfreeze_audio:
        audio_params = [p for n, p in model.named_parameters() if p.requires_grad and "audio_encoder" in n]
        head_params = [p for n, p in model.named_parameters() if p.requires_grad and "audio_encoder" not in n]
        optimizer = torch.optim.AdamW([
            {"params": audio_params, "lr": 1e-5},    # Ultra-slow LR for massive backbone
            {"params": head_params, "lr": args.lr}   # Normal LR for heads
        ], weight_decay=1e-4)
    else:
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=1e-4)
        
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_epoch = -1
    patience = 10
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy, _, _ = run_epoch(
            model=model,
            loader=dataloaders["train"],
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            desc=f"Epoch {epoch:03d} Train",
        )

        val_loss, val_accuracy, _, _ = run_epoch(
            model=model,
            loader=dataloaders["val"],
            device=device,
            criterion=criterion,
            optimizer=None,
            desc=f"Epoch {epoch:03d} Val",
        )

        summary = (
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | train_acc={train_accuracy:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_accuracy:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(checkpoint_path, model, optimizer, epoch, val_loss, args.mode)
            print(summary + " | best=updated")
        else:
            patience_counter += 1
            print(summary + f" | patience={patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}; best epoch was {best_epoch}")
                break

    load_checkpoint(checkpoint_path, model, device)

    test_loss, test_accuracy, test_preds, test_targets = run_epoch(
        model=model,
        loader=dataloaders["test"],
        device=device,
        criterion=criterion,
        optimizer=None,
    )

    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("\nPer-class report:")
    report = classification_report(
        test_targets,
        test_preds,
        labels=list(range(len(COMMAND_VOCAB))),
        target_names=COMMAND_VOCAB,
        zero_division=0,
    )
    print(report)


if __name__ == "__main__":
    main()