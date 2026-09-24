"""
train_cnn_v2.py

Trains the phase-2 species classifier on the window manifest produced by
build_window_manifest.py (real multi-recording iNatSounds data, not
augmented copies of one clip). Implements the recipe upgrades from the
workflow summary's "next steps" #4-5:
    - real capacity: torchvision resnet18/resnet50/mobilenet_v3_large
      (ImageNet-pretrained), not the 3-conv-block SmallAudioCNN phase-1
      used for ~9 classes
    - 128 mel bins / 11025Hz fmax, 3s windows strided 1.5s (via the
      manifest -- this file just reads whatever window each manifest row
      specifies)
    - SpecAugment (frequency + time masking), train-time only
    - Mixup on spectrograms
    - switchable CE vs assume-negative multilabel BCE loss (ml_config_v2.LOSS)

Usage:
    python train_cnn_v2.py --train-manifest manifests/train.jsonl --val-manifest manifests/val.jsonl
    python train_cnn_v2.py --train-manifest manifests/train.jsonl --val-manifest manifests/val.jsonl \\
        --model-arch mobilenet_v3_large --loss ce --epochs 20
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

try:  # package import (Flask) and direct-script import both work
    from . import ml_config_v2 as cfg
except ImportError:
    import ml_config_v2 as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Feature extraction (must match predict_v2.py exactly)
# --------------------------------------------------------------------------

def load_window(path: str, start_s: float, window_s: float = cfg.WINDOW_SECONDS, sr: int = cfg.SAMPLE_RATE):
    """Reads exactly one window directly out of the source file at the
    given offset (a seek + short decode, not a full-file decode) -- this
    is what makes the manifest-based approach cheap even though the
    routed corpus itself can be very large."""
    info = sf.info(path)
    native_sr = info.samplerate
    start_frame = int(round(start_s * native_sr))
    n_frames = int(round(window_s * native_sr))
    y, _ = sf.read(path, start=start_frame, frames=n_frames, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if native_sr != sr:
        y = librosa.resample(y, orig_sr=native_sr, target_sr=sr)
    target_len = int(round(window_s * sr))
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y.astype(np.float32)


def compute_logmel(y: np.ndarray, sr: int = cfg.SAMPLE_RATE):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS, fmin=cfg.FMIN, fmax=cfg.FMAX,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    if logmel.shape[1] < cfg.N_FRAMES:
        logmel = np.pad(logmel, ((0, 0), (0, cfg.N_FRAMES - logmel.shape[1])), constant_values=logmel.min())
    else:
        logmel = logmel[:, :cfg.N_FRAMES]
    logmel = (logmel + 40.0) / 40.0  # roughly [-1, 1]
    return logmel.astype(np.float32)


def spec_augment(logmel: np.ndarray):
    """Frequency + time masking, applied in place. Mirrors the official
    iNatSounds dataset.py's freq_masking/time_masking (mask value 0 in
    their [0,1]-normalized space; here we mask to the array's own min,
    since our normalization range differs)."""
    out = logmel.copy()
    fill = out.min()
    n_mels, n_frames = out.shape
    for _ in range(cfg.SPECAUG_N_FREQ_MASKS):
        w = np.random.randint(0, cfg.SPECAUG_FREQ_MASK_MAX + 1)
        if w == 0 or w >= n_mels:
            continue
        f0 = np.random.randint(0, n_mels - w)
        out[f0:f0 + w, :] = fill
    for _ in range(cfg.SPECAUG_N_TIME_MASKS):
        w = np.random.randint(0, cfg.SPECAUG_TIME_MASK_MAX + 1)
        if w == 0 or w >= n_frames:
            continue
        t0 = np.random.randint(0, n_frames - w)
        out[:, t0:t0 + w] = fill
    return out


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

class WindowManifestDataset(Dataset):
    def __init__(self, manifest_path: Path, class_to_idx: dict, train: bool):
        self.rows = []
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row["label"] not in class_to_idx:
                    continue
                self.rows.append(row)
        self.class_to_idx = class_to_idx
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        y = load_window(row["path"], row["start_s"])
        logmel = compute_logmel(y)
        if self.train and cfg.SPECAUGMENT:
            logmel = spec_augment(logmel)
        x = torch.from_numpy(logmel).unsqueeze(0).repeat(3, 1, 1)  # 1ch -> 3ch, for ImageNet backbones
        label = self.class_to_idx[row["label"]]
        return x, label


def build_class_index(train_manifest: Path):
    labels = set()
    with open(train_manifest) as f:
        for line in f:
            line = line.strip()
            if line:
                labels.add(json.loads(line)["label"])
    labels = sorted(labels)
    if cfg.BACKGROUND_LABEL in labels:
        labels.remove(cfg.BACKGROUND_LABEL)
        labels.append(cfg.BACKGROUND_LABEL)
    return {label: i for i, label in enumerate(labels)}


def build_balanced_sampler(dataset: WindowManifestDataset, power: float):
    """Down-weight dominant taxonomy-rollup classes without discarding data.

    iNatSounds contains very large order-level rollups alongside classes with
    only a handful of recordings. Raw shuffled minibatches would otherwise
    be overwhelmingly Passeriformes-like examples. Sampling with inverse
    count**power is intentionally softer than a fully uniform sampler.
    """
    if power <= 0:
        return None
    counts = np.bincount(
        [dataset.class_to_idx[row["label"]] for row in dataset.rows],
        minlength=len(dataset.class_to_idx),
    )
    weights = [float(counts[dataset.class_to_idx[row["label"]]]) ** -power for row in dataset.rows]
    logger.info("Balanced sampling enabled (power=%.2f; windows/class min=%d max=%d)",
                power, int(counts.min()), int(counts.max()))
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def build_model(arch: str, num_classes: int, pretrained: bool):
    if arch == "resnet18":
        model = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "resnet50":
        model = torchvision.models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "mobilenet_v3_large":
        model = torchvision.models.mobilenet_v3_large(weights="IMAGENET1K_V2" if pretrained else None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"unknown arch '{arch}'")
    return model


# --------------------------------------------------------------------------
# Mixup
# --------------------------------------------------------------------------

def mixup_batch(x, y_onehot, alpha):
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1 - lam) * x[perm]
    y_mixed = lam * y_onehot + (1 - lam) * y_onehot[perm]
    return x_mixed, y_mixed


# --------------------------------------------------------------------------
# Train / eval
# --------------------------------------------------------------------------

def compute_loss(logits, y_onehot, loss_kind: str):
    if loss_kind == "ce":
        # y_onehot may be soft (post-mixup) -- use soft cross-entropy either way
        log_probs = F.log_softmax(logits, dim=1)
        return -(y_onehot * log_probs).sum(dim=1).mean()
    elif loss_kind == "assume_negative_bce":
        return F.binary_cross_entropy_with_logits(logits, y_onehot)
    else:
        raise ValueError(f"unknown loss '{loss_kind}'")


def evaluate(model, loader, device, num_classes):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def train(train_manifest: Path, val_manifest: Path, model_path: Path,
          epochs: int, batch_size: int, lr: float, weight_decay: float,
          arch: str, pretrained: bool, loss_kind: str, mixup: bool, mixup_alpha: float,
          num_workers: int, class_balance_power: float, resume_path: Path | None = None):
    class_to_idx = build_class_index(train_manifest)
    num_classes = len(class_to_idx)
    logger.info("num_classes=%d", num_classes)

    train_ds = WindowManifestDataset(train_manifest, class_to_idx, train=True)
    val_ds = WindowManifestDataset(val_manifest, class_to_idx, train=False)
    logger.info("train windows=%d  val windows=%d", len(train_ds), len(val_ds))

    if not train_ds.rows or not val_ds.rows:
        raise ValueError("Training and validation manifests must both contain at least one readable window.")
    sampler = build_balanced_sampler(train_ds, class_balance_power)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=sampler is None,
                              sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(arch, num_classes, pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = -1.0
    start_epoch = 1
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device)
        if checkpoint.get("arch") != arch:
            raise ValueError(f"Resume checkpoint architecture is {checkpoint.get('arch')}, not requested {arch}")
        if checkpoint.get("class_to_idx") and checkpoint["class_to_idx"] != class_to_idx:
            raise ValueError("Resume checkpoint labels do not match the current training manifest.")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_acc = float(checkpoint.get("best_val_acc", -1.0))
        logger.info("Resuming %s after epoch %d (best val_acc=%.3f)",
                    resume_path, start_epoch - 1, best_val_acc)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = model_path.with_name(f"{model_path.stem}.last{model_path.suffix}")
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            y_onehot = F.one_hot(y, num_classes=num_classes).float()
            if mixup:
                x, y_onehot = mixup_batch(x, y_onehot, mixup_alpha)

            optimizer.zero_grad()
            logits = model(x)
            loss = compute_loss(logits, y_onehot, loss_kind)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        scheduler.step()
        train_loss = total_loss / max(len(train_ds), 1)
        val_acc = evaluate(model, val_loader, device, num_classes)
        logger.info("epoch %3d/%d  train_loss=%.4f  val_acc=%.3f  lr=%.2e",
                     epoch, epochs, train_loss, val_acc, scheduler.get_last_lr()[0])

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({"state_dict": model.state_dict(), "arch": arch, "loss_kind": loss_kind,
                        "class_to_idx": class_to_idx},
                       model_path)
            with open(model_path.with_suffix(".labels.json"), "w") as f:
                json.dump(class_to_idx, f, indent=2)

        # This checkpoint is deliberately separate from the best inference
        # checkpoint. Put --model-path on mounted Drive in Colab so an idle
        # runtime disconnect can resume the optimizer and LR schedule.
        torch.save({
            "state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(), "class_to_idx": class_to_idx,
            "arch": arch, "loss_kind": loss_kind, "epoch": epoch,
            "best_val_acc": best_val_acc,
        }, last_checkpoint_path)

    logger.info("Best val_acc=%.3f -- saved to %s", best_val_acc, model_path)
    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-manifest", type=Path, default=cfg.MANIFEST_DIR / "train.jsonl")
    parser.add_argument("--val-manifest", type=Path, default=cfg.MANIFEST_DIR / "val.jsonl")
    parser.add_argument("--model-path", type=Path, default=cfg.MODEL_PATH)
    parser.add_argument("--model-arch", default=cfg.MODEL_ARCH,
                         choices=["resnet18", "resnet50", "mobilenet_v3_large"])
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false", default=cfg.PRETRAINED)
    parser.add_argument("--loss", default=cfg.LOSS, choices=["ce", "assume_negative_bce"])
    parser.add_argument("--mixup", dest="mixup", action="store_true", default=cfg.MIXUP)
    parser.add_argument("--no-mixup", dest="mixup", action="store_false")
    parser.add_argument("--mixup-alpha", type=float, default=cfg.MIXUP_ALPHA)
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=cfg.LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=cfg.WEIGHT_DECAY)
    parser.add_argument("--num-workers", type=int, default=cfg.NUM_WORKERS)
    parser.add_argument("--class-balance-power", type=float, default=cfg.CLASS_BALANCE_POWER,
                        help="0 disables balancing; 0.5 softens iNatSounds' extreme long tail")
    parser.add_argument("--resume", type=Path, default=None,
                        help="path to the .last.pt checkpoint written after each epoch")
    args = parser.parse_args()

    train(
        args.train_manifest, args.val_manifest, args.model_path,
        args.epochs, args.batch_size, args.lr, args.weight_decay,
        args.model_arch, args.pretrained, args.loss, args.mixup, args.mixup_alpha,
        args.num_workers, args.class_balance_power, args.resume,
    )


if __name__ == "__main__":
    main()
