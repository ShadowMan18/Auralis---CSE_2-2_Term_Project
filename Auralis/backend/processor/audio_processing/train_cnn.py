"""
train_cnn.py

Trains a small CNN classifier over log-mel spectrograms, on the dataset
produced by dataset_producer.py.

THIS IS A PHASE-1 PIPELINE CHECK, not a production model: with a single
real source recording per class (just heavily augmented), the model can
easily reach high train/val accuracy while still overfitting to that one
recording's timbre/mic/background-noise fingerprint rather than learning
the species in general. High val accuracy here mainly tells you the
pipeline -- data loading, feature extraction, training loop, checkpointing,
inference -- actually works end to end, not that the model will
generalize to a genuinely different recording of the same species. Swap
in a real multi-recording dataset per class (via dataset_producer.py
--references-dir pointing at that folder) before trusting accuracy
numbers.

Usage:
    python train_cnn.py
    python train_cnn.py --epochs 40 --dataset-dir dataset
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import ml_config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_logmel(path):
    y, _sr = librosa.load(str(path), sr=cfg.SAMPLE_RATE, mono=True)
    target_len = int(cfg.CLIP_SECONDS * cfg.SAMPLE_RATE)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    mel = librosa.feature.melspectrogram(
        y=y, sr=cfg.SAMPLE_RATE, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS, fmin=cfg.FMIN, fmax=cfg.FMAX,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    # fix frame count exactly (rounding in STFT framing can be off by one)
    if logmel.shape[1] < cfg.N_FRAMES:
        logmel = np.pad(logmel, ((0, 0), (0, cfg.N_FRAMES - logmel.shape[1])), constant_values=logmel.min())
    else:
        logmel = logmel[:, :cfg.N_FRAMES]
    # normalize to roughly [-1, 1] -- dB values are typically in [-80, 0]
    logmel = (logmel + 40.0) / 40.0
    return logmel.astype(np.float32)


class SpectrogramDataset(Dataset):
    def __init__(self, split_dir: Path, class_to_idx: dict):
        self.paths = []
        self.labels = []
        for label_dir in sorted(split_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            if label not in class_to_idx:
                continue
            for wav_path in sorted(label_dir.glob("*.wav")):
                self.paths.append(wav_path)
                self.labels.append(class_to_idx[label])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        logmel = compute_logmel(self.paths[idx])
        x = torch.from_numpy(logmel).unsqueeze(0)  # (1, n_mels, n_frames)
        y = self.labels[idx]
        return x, y


class SmallAudioCNN(nn.Module):
    """Deliberately small -- this dataset (one real clip per class,
    augmented) doesn't have enough genuine diversity to justify or
    benefit from a bigger network; a bigger model would just overfit
    faster without learning anything more general."""

    def __init__(self, num_classes, n_mels=cfg.N_MELS, n_frames=cfg.N_FRAMES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def build_class_index(dataset_dir: Path):
    train_dir = dataset_dir / "train"
    labels = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    # background last, for a slightly nicer printed order; index values don't matter
    if cfg.BACKGROUND_LABEL in labels:
        labels.remove(cfg.BACKGROUND_LABEL)
        labels.append(cfg.BACKGROUND_LABEL)
    return {label: i for i, label in enumerate(labels)}


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def train(dataset_dir: Path, model_path: Path, epochs: int, batch_size: int, lr: float):
    class_to_idx = build_class_index(dataset_dir)
    logger.info("Classes: %s", class_to_idx)

    train_ds = SpectrogramDataset(dataset_dir / "train", class_to_idx)
    val_ds = SpectrogramDataset(dataset_dir / "val", class_to_idx)
    logger.info("train=%d val=%d examples", len(train_ds), len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallAudioCNN(num_classes=len(class_to_idx)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        train_loss = total_loss / max(len(train_ds), 1)
        val_acc = evaluate(model, val_loader, device)
        logger.info("epoch %3d/%d  train_loss=%.4f  val_acc=%.3f", epoch, epochs, train_loss, val_acc)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            with open(model_path.with_suffix(".labels.json"), "w") as f:
                json.dump(class_to_idx, f, indent=2)

    logger.info("Best val_acc=%.3f -- saved to %s", best_val_acc, model_path)
    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=cfg.DATASET_DIR)
    parser.add_argument("--model-path", type=Path, default=cfg.MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=cfg.LEARNING_RATE)
    args = parser.parse_args()

    train(args.dataset_dir, args.model_path, args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
