"""
dataset_producer.py

Generates an augmented training dataset from a folder of reference audio
clips (one file per class, filename stem = class label), for training the
phase-1 CNN classifier.

THIS IS A PHASE-1 TOOL: it stretches ONE real recording per class into
many distorted copies. That's enough to validate the pipeline end to
end -- data generation, feature extraction, training loop, inference --
but it teaches the model to recognize augmented versions of THAT
recording, not the general acoustic category (same limitation the DSP
matcher had, just relocated). When you have a real multi-recording
dataset per species, point this script at that folder instead -- pass
--references-dir pointing at a folder with MULTIPLE files per class
(e.g. cat_001.wav, cat_002.wav, ...; the class is taken from the part of
the filename before the first "_", falling back to the whole stem if
there's no "_") and augmentation still helps on top of that real
diversity, it's just not a substitute for it.

Also generates a "_background" class: silence, white noise, and pink-ish
noise at varying levels. Without an explicit background/negative class, a
closed-set classifier will always confidently pick one of your species
labels even for silence or an unrelated sound -- there's no "none of the
above" otherwise.

Output layout:
    <output_dir>/
        train/
            <species>/  *.wav
            _background/  *.wav
        val/
            <species>/  *.wav
            _background/  *.wav

Usage:
    python dataset_producer.py --variants-per-clip 60
    python dataset_producer.py --references-dir /path/to/real/dataset --output-dir dataset
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

import ml_config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

# Augmentation ranges -- deliberately broader than the DSP pipeline's small
# build-time grid (pipeline_config.py), since this produces a *training
# set* (breadth of examples matters) rather than a handful of matched
# reference variants for exact hashing.
PITCH_STEPS_RANGE = (-4.0, 4.0)       # semitones
TEMPO_RATE_RANGE = (0.75, 1.35)       # multiplier
NOISE_SNR_DB_RANGE = (0.0, 30.0)      # lower = noisier; applied with p=0.8
GAIN_DB_RANGE = (-12.0, 6.0)
SILENCE_PAD_SECONDS_RANGE = (0.0, 1.5)
TONE_PROBABILITY = 0.15
TONE_FREQ_RANGE = (200.0, 4000.0)
TONE_GAIN_RANGE = (0.05, 0.25)        # relative to signal peak

N_BACKGROUND_VARIANTS = 80


def _load_mono(path, sr):
    y, _sr = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def _add_noise(y, snr_db):
    sig_power = np.mean(y ** 2) + 1e-12
    noise = np.random.normal(0, 1, len(y)).astype(np.float32)
    noise_power = np.mean(noise ** 2) + 1e-12
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / noise_power)
    return y + noise


def _add_tone(y, sr):
    freq = random.uniform(*TONE_FREQ_RANGE)
    gain = random.uniform(*TONE_GAIN_RANGE)
    t = np.arange(len(y)) / sr
    peak = np.max(np.abs(y)) or 1.0
    tone = (gain * peak * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return y + tone


def _augment(y, sr):
    """Apply a random subset/combination of distortions. Each call
    produces a different combination -- that's the point of stochastic
    per-sample augmentation over a fixed pre-baked grid: many more
    effective combinations for the same number of output files."""
    if random.random() < 0.7:
        steps = random.uniform(*PITCH_STEPS_RANGE)
        try:
            y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=steps)
        except Exception:
            pass
    if random.random() < 0.7:
        rate = random.uniform(*TEMPO_RATE_RANGE)
        try:
            y = librosa.effects.time_stretch(y=y, rate=rate)
        except Exception:
            pass

    gain_db = random.uniform(*GAIN_DB_RANGE)
    y = y * (10 ** (gain_db / 20))

    if random.random() < 0.8:
        snr_db = random.uniform(*NOISE_SNR_DB_RANGE)
        y = _add_noise(y, snr_db)

    if random.random() < TONE_PROBABILITY:
        y = _add_tone(y, sr)

    pad_s = random.uniform(*SILENCE_PAD_SECONDS_RANGE)
    if pad_s > 0:
        pad_n = int(pad_s * sr)
        left = random.randint(0, pad_n)
        y = np.pad(y, (left, pad_n - left))

    return y.astype(np.float32)


def _fix_length(y, sr, seconds=cfg.CLIP_SECONDS):
    """CNN needs a fixed-size input -- crop (randomly, for variety) or
    pad (randomly positioned) to exactly `seconds`."""
    target_len = int(seconds * sr)
    if len(y) >= target_len:
        start = random.randint(0, len(y) - target_len)
        return y[start:start + target_len]
    pad_total = target_len - len(y)
    pad_left = random.randint(0, pad_total)
    pad_right = pad_total - pad_left
    return np.pad(y, (pad_left, pad_right))


def _normalize_peak(y, target=0.9):
    peak = np.max(np.abs(y))
    if peak < 1e-6:
        return y
    return (y / peak * target).astype(np.float32)


def _make_background_clip(sr, seconds=cfg.CLIP_SECONDS):
    kind = random.choice(["silence", "white", "pink", "quiet_white"])
    n = int(seconds * sr)
    if kind == "silence":
        y = np.zeros(n, dtype=np.float32)
        # a touch of very quiet noise so it's not literally all-zero
        y += np.random.normal(0, 0.001, n).astype(np.float32)
    elif kind == "white":
        y = np.random.normal(0, random.uniform(0.05, 0.3), n).astype(np.float32)
    elif kind == "quiet_white":
        y = np.random.normal(0, random.uniform(0.01, 0.05), n).astype(np.float32)
    else:  # pink-ish: filtered white noise, cheap 1/f approximation via cumsum+normalize
        white = np.random.normal(0, 1, n).astype(np.float32)
        pink = np.cumsum(white)
        pink = pink - np.mean(pink)
        pink = pink / (np.max(np.abs(pink)) + 1e-9) * random.uniform(0.05, 0.3)
        y = pink.astype(np.float32)
    return y


def _class_label_for(path: Path):
    """species from filename: 'cat.mp3' -> 'cat', 'cat_003.wav' -> 'cat'."""
    stem = path.stem
    return stem.split("_")[0] if "_" in stem else stem


def build_dataset(references_dir: Path, output_dir: Path, variants_per_clip: int,
                   val_fraction: float = cfg.VAL_FRACTION, seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)

    reference_files = sorted(
        p for p in references_dir.iterdir()
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
    )
    if not reference_files:
        raise FileNotFoundError(f"No audio files found in {references_dir}")

    by_class = {}
    for path in reference_files:
        by_class.setdefault(_class_label_for(path), []).append(path)

    for split in ("train", "val"):
        for label in list(by_class.keys()) + [cfg.BACKGROUND_LABEL]:
            (output_dir / split / label).mkdir(parents=True, exist_ok=True)

    total_written = 0
    for label, paths in by_class.items():
        n_val = max(1, int(round(variants_per_clip * len(paths) * val_fraction)))
        variant_idx = 0
        for path in paths:
            y = _load_mono(path, cfg.SAMPLE_RATE)
            for _ in range(variants_per_clip):
                variant = _augment(y, cfg.SAMPLE_RATE)
                variant = _fix_length(variant, cfg.SAMPLE_RATE)
                variant = _normalize_peak(variant)
                split = "val" if variant_idx < n_val else "train"
                out_path = output_dir / split / label / f"{label}_{variant_idx:04d}.wav"
                sf.write(str(out_path), variant, cfg.SAMPLE_RATE)
                variant_idx += 1
                total_written += 1
        logger.info("'%s': wrote %d variants from %d source clip(s)", label, variant_idx, len(paths))

    n_val_bg = max(1, int(round(N_BACKGROUND_VARIANTS * val_fraction)))
    for i in range(N_BACKGROUND_VARIANTS):
        y = _make_background_clip(cfg.SAMPLE_RATE)
        y = _normalize_peak(y, target=random.uniform(0.1, 0.6))
        split = "val" if i < n_val_bg else "train"
        out_path = output_dir / split / cfg.BACKGROUND_LABEL / f"bg_{i:04d}.wav"
        sf.write(str(out_path), y, cfg.SAMPLE_RATE)
        total_written += 1
    logger.info("'%s': wrote %d background variants", cfg.BACKGROUND_LABEL, N_BACKGROUND_VARIANTS)

    logger.info("Done: %d total files written to %s", total_written, output_dir)
    return sorted(by_class.keys())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references-dir", type=Path, default=cfg.REFERENCES_DIR)
    parser.add_argument("--output-dir", type=Path, default=cfg.DATASET_DIR)
    parser.add_argument("--variants-per-clip", type=int, default=60)
    parser.add_argument("--val-fraction", type=float, default=cfg.VAL_FRACTION)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    classes = build_dataset(
        args.references_dir, args.output_dir, args.variants_per_clip,
        args.val_fraction, args.seed,
    )
    logger.info("Classes: %s + %s", classes, cfg.BACKGROUND_LABEL)


if __name__ == "__main__":
    main()
