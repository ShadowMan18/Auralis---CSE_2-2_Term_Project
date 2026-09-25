"""
noise_bed.py

Realistic background-noise generation, shared by dataset_producer.py (build
training clips) and train_cnn.py / predict.py (pad short uploads at
inference). This is the single source of truth for "what does silence /
background sound like in this pipeline" -- train and inference MUST agree on
this or the model sees a different input distribution than it trained on
(same reasoning as ml_config.py's docstring, applied to noise instead of
feature extraction).

Earlier versions padded short clips with either digital zeros or flat
Gaussian noise. Both are unlike real recordings (measured: a model trained
on zero-padded clips dropped from 88% to 48% accuracy on clips with a
realistic noise floor -- see chat history / PIPELINE_NOTES.md). Tiling the
event to fill the window was also tested and rejected: it creates a
near-perfectly periodic waveform (autocorrelation ~0.9, vs ~0 for real
audio) that a CNN can key on as a shortcut instead of learning the sound
itself.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfilt

BED_KINDS = ["white", "pink", "brown", "hum", "wind", "quiet"]
BED_WEIGHTS = [0.20, 0.25, 0.15, 0.10, 0.15, 0.15]
# SNR of a real call vs its background bed. "quiet" bypasses this (near-silent
# room) and uses a much higher SNR range instead -- see make_bed_for().
SNR_DB_RANGE = (3.0, 35.0)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))


def _pink(n, sr, rng):
    spec = np.fft.rfft(rng.standard_normal(n))
    spec /= np.sqrt(np.maximum(np.arange(len(spec)), 1))
    return np.fft.irfft(spec, n)


def _brown(n, sr, rng):
    x = np.cumsum(rng.standard_normal(n))
    x -= x.mean()
    return sosfilt(butter(2, 20, btype="highpass", fs=sr, output="sos"), x)  # remove DC drift


def make_bed(kind: str, n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-RMS background noise of the requested kind."""
    if kind in ("white", "quiet"):
        x = rng.standard_normal(n)
    elif kind == "pink":
        x = _pink(n, sr, rng)
    elif kind == "brown":
        x = _brown(n, sr, rng)
    elif kind == "hum":  # mains hum + harmonics
        t = np.arange(n) / sr
        f0 = float(rng.choice([50.0, 60.0]))
        x = sum((1.0 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 2 * np.pi))
                for k in range(1, int(rng.integers(3, 7))))
        x = x + 0.05 * rng.standard_normal(n)
    elif kind == "wind":  # low-passed noise with a slow swell
        x = sosfilt(butter(2, rng.uniform(150, 900), btype="lowpass", fs=sr, output="sos"),
                    rng.standard_normal(n))
        t = np.arange(n) / sr
        x = x * (1.0 + 0.7 * np.sin(2 * np.pi * rng.uniform(0.15, 0.8) * t + rng.uniform(0, 2 * np.pi)))
    else:
        raise ValueError(f"unknown bed kind {kind!r}")
    return (x / rms(x)).astype(np.float32)


def random_bed(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """A single random-kind unit-RMS bed (what dataset_producer uses for pure
    _background clips)."""
    kind = str(rng.choice(BED_KINDS, p=BED_WEIGHTS))
    return make_bed(kind, n, sr, rng)


def bed_at_snr(n: int, sr: int, rng: np.random.Generator, signal_rms: float,
              snr_db_range=SNR_DB_RANGE) -> np.ndarray:
    """A random-kind noise bed scaled so a signal with `signal_rms` sits at a
    random SNR drawn from `snr_db_range` above it -- this is what real
    padding/background looks like relative to a foreground call."""
    kind = str(rng.choice(BED_KINDS, p=BED_WEIGHTS))
    snr_db = rng.uniform(35.0, 50.0) if kind == "quiet" else rng.uniform(*snr_db_range)
    bed = make_bed(kind, n, sr, rng)
    return bed * (signal_rms / (10 ** (snr_db / 20)))