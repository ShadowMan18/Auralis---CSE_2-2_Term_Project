"""
train_cnn.py

Trains a small CNN classifier over log-mel spectrograms, on the dataset
produced by dataset_producer.py.

VALIDATION IS ONLY AS HONEST AS THE DATA. With >= 3 recordings of a class,
dataset_producer.py holds out whole recordings for validation and val
accuracy is a real generalisation estimate. With fewer, validation clips are
augmentations of the recording(s) the model trained on, so val accuracy is
optimistic (this script warns per class). More real recordings per species
is the biggest accuracy lever there is.

Short-clip padding (waveform_to_logmel) uses the same noise_bed.py generator
as dataset_producer.py's training clips -- see _pad_with_noise_bed for why.

MASKED POOLING: matching the padding's statistics to real background noise
(above) still leaves the padded region able to influence the prediction --
it is real numbers flowing through real conv/BN layers into the same global
average every other frame contributes to. For a short upload, most of that
average could be synthetic. SmallAudioCNN.forward() takes an optional
`valid_frames` argument: when given, the final pooling step averages ONLY
over the frames that came from the actual uploaded audio, and the padded
frames get exactly zero weight -- not "low weight", zero. This is a
provable fix, not a statistical one: it no longer matters what the padding
contains, because it can no longer reach the classifier at all. Training
clips (dataset_producer.py) are NOT masked -- their whole 3 s canvas is
deliberately real content + realistic background by construction (the event
is placed at a random position on purpose, so the model learns to find a
call anywhere in the window), so there is no "this part doesn't count"
region to mask there. Masking only applies where it's actually needed: a
real upload shorter than the model's 3 s window.

Training details: SpecAugment (random frequency/time masks, fresh every
epoch), label smoothing, AdamW with cosine learning-rate decay, dropout. The
best checkpoint is the one with the highest val accuracy, ties broken by
lower val loss. Per-class val accuracy is printed at the end.

Usage:
    python train_cnn.py
    python train_cnn.py --epochs 40 --dataset-dir dataset
"""

import argparse
import json
import hashlib
import logging
from pathlib import Path

import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:  # imported as part of the processor.audio_processing package (Flask service)
    from . import ml_config as cfg
    from . import noise_bed as nb
except ImportError:  # run directly as a script from this folder
    import ml_config as cfg
    import noise_bed as nb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _pad_with_noise_bed(y, target_len):
    """Extend a short clip to target_len with a REALISTIC noise bed (white /
    pink / brown / mains hum / wind, at a random SNR against the clip's own
    level) instead of digital zeros -- the same noise_bed.py generator that
    dataset_producer.py uses under every training clip. Earlier this padded
    with flat low-level Gaussian noise instead; that was closer to real audio
    than zeros, but still a different noise distribution from what the model
    trained on. Using the SAME generator on both sides is what ml_config.py's
    "train and inference must match" rule actually requires here. Tiling/
    repeating the clip to fill the space was tested and rejected: it creates
    near-perfectly periodic audio (autocorrelation ~0.9 vs ~0 for real audio)
    that the model could learn to detect instead of the sound itself.

    Deterministic (fixed seed) so the same upload always gives the same
    prediction."""
    pad_len = target_len - len(y)
    if pad_len <= 0:
        return y
    # Seed from the clip's own content (not a fixed constant): same upload -> same
    # padding -> same prediction every time, but DIFFERENT uploads get different
    # bed kinds, matching the spread of kinds (white/pink/brown/hum/wind/quiet)
    # dataset_producer.py trains on. A fixed seed would deterministically pick the
    # SAME bed kind for every upload, which is a narrower distribution than training.
    seed = int(hashlib.blake2b(y.tobytes(), digest_size=4).hexdigest(), 16) if len(y) else 0
    rng = np.random.default_rng(seed)
    signal_rms = nb.rms(y) if len(y) else 1e-3
    bed = nb.bed_at_snr(pad_len, cfg.SAMPLE_RATE, rng, signal_rms)
    return np.concatenate([y, bed]).astype(np.float32)


def waveform_to_logmel(y):
    """Feature extraction for ONE clip, given a mono float waveform that is
    already at cfg.SAMPLE_RATE. Padded (with noise, see _pad_with_noise_floor)
    or cropped to exactly cfg.CLIP_SECONDS.

    Split out of compute_logmel() so the API can run the identical feature
    code on sliding windows of a longer upload -- training and inference
    must never use two different copies of this."""
    target_len = int(cfg.CLIP_SECONDS * cfg.SAMPLE_RATE)
    if len(y) < target_len:
        y = _pad_with_noise_bed(y, target_len)
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


def valid_frame_count(n_samples):
    """How many of cfg.N_FRAMES output columns are REAL audio for a clip of
    `n_samples` raw samples (before any padding). Matches librosa's own
    framing (1 + n_samples // hop_length), capped at N_FRAMES -- a clip that
    fills or exceeds the window has none of it masked."""
    return int(min(cfg.N_FRAMES, 1 + n_samples // cfg.HOP_LENGTH))


def waveform_to_logmel_with_length(y):
    """Same as waveform_to_logmel, but also returns valid_frame_count(len(y))
    -- the number of output columns that are this clip's real audio, before
    padding was added. Pass this to SmallAudioCNN.forward's `valid_frames`
    so padded frames are excluded from pooling. Training doesn't use this
    (see the MASKED POOLING note above); only short-upload inference does."""
    return waveform_to_logmel(y), valid_frame_count(len(y))


def compute_logmel_with_length(path):
    """Load a file and return (log-mel, valid_frame_count) -- the file-path version
    of waveform_to_logmel_with_length, for predict.py."""
    y, _sr = librosa.load(str(path), sr=cfg.SAMPLE_RATE, mono=True)
    return waveform_to_logmel(y), valid_frame_count(len(y))


def compute_logmel(path):
    """Load a file and return the log-mel of its first cfg.CLIP_SECONDS."""
    y, _sr = librosa.load(str(path), sr=cfg.SAMPLE_RATE, mono=True)
    return waveform_to_logmel(y)


# SpecAugment (training only -- never applied at inference). The time mask is
# kept short (<= ~0.12 s) so it can't erase a whole short call such as a bark.
SPEC_AUG_FREQ_MASKS = 2        # 0..2 frequency masks per example
SPEC_AUG_FREQ_WIDTH = 6        # mel bins (of N_MELS)
SPEC_AUG_TIME_PROB = 0.5       # chance of one time mask
SPEC_AUG_TIME_WIDTH = 10       # frames (~0.12 s at HOP_LENGTH 256)


def spec_augment(logmel, rng=None):
    """Mask random mel bands / a short time span of a (n_mels, n_frames)
    log-mel with the clip's mean value. Returns a modified copy."""
    rng = rng if rng is not None else np.random.default_rng()
    out = logmel.copy()
    fill = float(out.mean())
    n_mels, n_frames = out.shape
    for _ in range(int(rng.integers(0, SPEC_AUG_FREQ_MASKS + 1))):
        w = int(rng.integers(1, SPEC_AUG_FREQ_WIDTH + 1))
        f0 = int(rng.integers(0, n_mels - w + 1))
        out[f0:f0 + w, :] = fill
    if rng.random() < SPEC_AUG_TIME_PROB:
        w = int(rng.integers(1, SPEC_AUG_TIME_WIDTH + 1))
        t0 = int(rng.integers(0, n_frames - w + 1))
        out[:, t0:t0 + w] = fill
    return out


class SpectrogramDataset(Dataset):
    def __init__(self, split_dir: Path, class_to_idx: dict, augment: bool = False):
        self.augment = augment
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
        if self.augment:
            logmel = spec_augment(logmel)
        x = torch.from_numpy(logmel).unsqueeze(0)  # (1, n_mels, n_frames)
        y = self.labels[idx]
        return x, y


class SmallAudioCNN(nn.Module):
    """Deliberately small -- this dataset (one real clip per class,
    augmented) doesn't have enough genuine diversity to justify or
    benefit from a bigger network; a bigger model would just overfit
    faster without learning anything more general."""

    # Two MaxPool2d(2) layers in `features`, each halving the time axis (floor
    # division, matching PyTorch's default MaxPool2d behaviour exactly).
    _TIME_DOWNSAMPLE = 4  # 2 * 2

    def __init__(self, num_classes, n_mels=cfg.N_MELS, n_frames=cfg.N_FRAMES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            # Global pooling used to live here as a no-op AdaptiveAvgPool2d(1) layer.
            # It's done manually in forward() now (optionally masked) instead --
            # AdaptiveAvgPool2d and Dropout both have zero learnable parameters, so
            # this is NOT a breaking change: state_dict keys are identical, and a
            # checkpoint saved before this change still loads here without issue.
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x, valid_frames=None):
        """valid_frames: optional (N,) int tensor/array, one value per example in the
        batch, giving how many columns of the ORIGINAL (n_mels, n_frames) input are
        real audio (see valid_frame_count) rather than padding. When given, pooling
        averages only over those columns -- padded columns get exactly zero weight,
        so their content (whatever it is) cannot influence the prediction. Omit it
        (the default) for a plain global average over the whole input, which is what
        training uses -- see this file's MASKED POOLING docstring note for why."""
        feat = self.features(x)  # (N, 64, H', W')
        if valid_frames is None:
            pooled = feat.mean(dim=(2, 3))
        else:
            n, _c, h, w = feat.shape
            valid_frames = torch.as_tensor(valid_frames, device=feat.device, dtype=torch.float32)
            # same two floor-halvings the two MaxPool2d(2) layers just applied to the
            # real spectrogram width, so the mask lines up with `feat`'s actual columns
            valid_w = torch.clamp((valid_frames.long() // 2) // 2, min=1, max=w)
            col = torch.arange(w, device=feat.device).unsqueeze(0)          # (1, W')
            mask = (col < valid_w.unsqueeze(1)).to(feat.dtype)              # (N, W')
            mask = mask.view(n, 1, 1, w)
            pooled = (feat * mask).sum(dim=(2, 3)) / (h * valid_w.view(n, 1).to(feat.dtype))
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


def build_class_index(dataset_dir: Path):
    train_dir = dataset_dir / "train"
    labels = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    # background last, for a slightly nicer printed order; index values don't matter
    if cfg.BACKGROUND_LABEL in labels:
        labels.remove(cfg.BACKGROUND_LABEL)
        labels.append(cfg.BACKGROUND_LABEL)
    return {label: i for i, label in enumerate(labels)}


def evaluate(model, loader, device, criterion, n_classes):
    """Returns (accuracy, mean loss, correct-per-class, total-per-class)."""
    model.eval()
    loss_sum = 0.0
    correct = np.zeros(n_classes)
    total = np.zeros(n_classes)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += criterion(logits, y).item() * y.size(0)
            y_np = y.cpu().numpy()
            np.add.at(total, y_np, 1)
            np.add.at(correct, y_np, (logits.argmax(dim=1).cpu().numpy() == y_np).astype(float))
    n = max(total.sum(), 1)
    return correct.sum() / n, loss_sum / n, correct, total


def _warn_about_val_honesty(dataset_dir: Path):
    info_path = dataset_dir / "dataset_info.json"
    if not info_path.is_file():
        logger.warning("No dataset_info.json in %s: this dataset was made by the OLD dataset_producer.py "
                       "(zero-padded clips). Regenerate it with the current one.", dataset_dir)
        return
    with open(info_path) as f:
        classes = json.load(f).get("classes", {})
    leaky = sorted(c for c, v in classes.items() if v.get("val_mode") != "source")
    if leaky:
        logger.warning("Validation is OPTIMISTIC for %s: too few recordings, so val clips are augmentations "
                       "of training recordings. Add recordings (%s_001.wav, ...) for an honest number.",
                       leaky, leaky[0])


def train(dataset_dir: Path, model_path: Path, epochs: int, batch_size: int, lr: float,
          num_workers: int = 0, label_smoothing: float = 0.1, weight_decay: float = 1e-2,
          spec_aug: bool = True):
    class_to_idx = build_class_index(dataset_dir)
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    logger.info("Classes: %s", class_to_idx)
    _warn_about_val_honesty(dataset_dir)

    train_ds = SpectrogramDataset(dataset_dir / "train", class_to_idx, augment=spec_aug)
    val_ds = SpectrogramDataset(dataset_dir / "val", class_to_idx)
    logger.info("train=%d val=%d examples (SpecAugment %s)", len(train_ds), len(val_ds),
                "on" if spec_aug else "off")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(class_to_idx)
    model = SmallAudioCNN(num_classes=n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()  # plain CE so val loss is comparable across runs

    best_acc, best_loss, best_report = -1.0, float("inf"), None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        scheduler.step()

        train_loss = total_loss / max(len(train_ds), 1)
        val_acc, val_loss, correct, total = evaluate(model, val_loader, device, eval_criterion, n_classes)
        logger.info("epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.3f",
                    epoch, epochs, train_loss, val_loss, val_acc)

        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
            best_acc, best_loss, best_report = val_acc, val_loss, (correct.copy(), total.copy())
            torch.save(model.state_dict(), model_path)
            with open(model_path.with_suffix(".labels.json"), "w") as f:
                json.dump(class_to_idx, f, indent=2)

    logger.info("Best val_acc=%.3f (val_loss=%.4f) -- saved to %s", best_acc, best_loss, model_path)
    per_class = {}
    if best_report is not None:
        correct, total = best_report
        for i in range(n_classes):
            per_class[idx_to_class[i]] = {"correct": int(correct[i]), "total": int(total[i])}
            logger.info("  %-12s %5.1f%%  (%d/%d)", idx_to_class[i],
                        100.0 * correct[i] / max(total[i], 1), correct[i], total[i])
    # per_class is additional (appended, not inserted) so existing callers that only
    # unpack/use the first return value are unaffected; cross_validate.py uses both.
    return best_acc, per_class


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=cfg.DATASET_DIR)
    parser.add_argument("--model-path", type=Path, default=cfg.MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=cfg.LEARNING_RATE)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (0 is safest on Windows; try 2-4 to speed up)")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--no-spec-augment", action="store_true")
    args = parser.parse_args()

    train(args.dataset_dir, args.model_path, args.epochs, args.batch_size, args.lr,
          args.num_workers, args.label_smoothing, args.weight_decay, not args.no_spec_augment)


if __name__ == "__main__":
    main()