"""Create fixed-length waveform-augmented training audio for the species CNN.

Input is the species-only routed layout made by ``inatsounds_route.py``:
    <routed-dir>/train/<species_slug>/<recording>.wav

Output is deliberately split at the ORIGINAL RECORDING level before any
augmentation:
    <output-dir>/train/<species_slug>/<recording>_aug00.flac ... _aug04.flac
    <output-dir>/val/<species_slug>/<recording>_clean.flac

This prevents a source recording and its pitch/noise/speed variants appearing
on both sides of validation.  Training gets exactly five modified copies per
source by default; validation keeps a clean, fixed-length copy so its metric
still measures something useful.

Important: altered copies do NOT turn one recording into five independent
animal observations. iNatSounds has 1,951 one-recording species. Augmentation
helps tolerance to microphones/noise/pitch/tempo, but cannot make their
species-level predictions reliable by itself.

The full 137k-recording dataset produces roughly 685k three-second files and
needs substantial extra storage. Run ``--max-source-files 100`` first to test
the pipeline, preferably on Colab's local /content disk.

Examples:
    python dataset_generator_v2.py --routed-dir inat_routed_species \
        --output-dir inat_augmented_species --variants-per-file 5
    python dataset_generator_v2.py --routed-dir inat_routed_species \
        --output-dir inat_augmented_species --max-source-files 100
"""

import argparse
import hashlib
import logging
import random
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

try:
    from . import ml_config_v2 as cfg
except ImportError:
    import ml_config_v2 as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def stable_unit_interval(value: str, seed: int) -> float:
    """Stable per-recording split choice, independent of directory order."""
    digest = hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def fixed_length(y: np.ndarray, rng: random.Random, random_crop: bool) -> np.ndarray:
    target_samples = int(round(cfg.WINDOW_SECONDS * cfg.SAMPLE_RATE))
    if len(y) >= target_samples:
        offset = rng.randint(0, len(y) - target_samples) if random_crop else (len(y) - target_samples) // 2
        return y[offset:offset + target_samples]
    padding = target_samples - len(y)
    left = rng.randint(0, padding) if random_crop else padding // 2
    return np.pad(y, (left, padding - left))


def normalize(y: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    return (y / peak * 0.95).astype(np.float32) if peak > 1e-7 else y.astype(np.float32)


def add_noise(y: np.ndarray, rng: random.Random) -> np.ndarray:
    # Signal-to-noise ratio rather than fixed amplitude keeps quiet and loud
    # source files comparably augmented.
    snr_db = rng.uniform(8.0, 30.0)
    signal_power = float(np.mean(y ** 2)) + 1e-12
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, 1, len(y)).astype(np.float32)
    noise_power = float(np.mean(noise ** 2)) + 1e-12
    return y + noise * np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))


def augment(y: np.ndarray, rng: random.Random) -> np.ndarray:
    """Pitch shift, speed shift, gain, noise, and fixed three-second crop/pad."""
    if rng.random() < 0.85:
        y = librosa.effects.pitch_shift(y, sr=cfg.SAMPLE_RATE, n_steps=rng.uniform(-3.0, 3.0))
    if rng.random() < 0.85:
        y = librosa.effects.time_stretch(y, rate=rng.uniform(0.80, 1.20))
    y = y * (10 ** (rng.uniform(-8.0, 4.0) / 20.0))
    if rng.random() < 0.85:
        y = add_noise(y, rng)
    return normalize(fixed_length(y, rng, random_crop=True))


def clean_validation_copy(y: np.ndarray) -> np.ndarray:
    # A fixed deterministic seed matters only for the crop; no distortion.
    return normalize(fixed_length(y, random.Random(0), random_crop=False))


def source_files(routed_train_dir: Path):
    for class_dir in sorted(path for path in routed_train_dir.iterdir() if path.is_dir()):
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                yield class_dir.name, path


def split_sources(grouped: dict[str, list[Path]], val_fraction: float, seed: int):
    """Hold out an original recording only when that species has >=2 sources."""
    train, val = [], []
    for label, paths in grouped.items():
        # A one-recording species must remain train-only; using an altered
        # copy as validation would be leakage, not validation.
        label_val = [path for path in paths if len(paths) > 1 and stable_unit_interval(str(path), seed) < val_fraction]
        if len(paths) > 1 and not label_val:
            label_val = [min(paths, key=lambda path: stable_unit_interval(str(path), seed))]
        if len(label_val) == len(paths):
            # Random hashing can theoretically select every source. Keep one
            # training original so that species remains learnable.
            label_val.pop()
        label_val_set = set(label_val)
        val.extend((label, path) for path in label_val)
        train.extend((label, path) for path in paths if path not in label_val_set)
    return train, val


def write_audio(path: Path, y: np.ndarray, output_format: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    subtype = "PCM_16"
    sf.write(str(path.with_suffix(f".{output_format}")), y, cfg.SAMPLE_RATE, subtype=subtype)


def generate(routed_dir: Path, output_dir: Path, variants_per_file: int,
             val_fraction: float, seed: int, output_format: str, max_source_files: int | None,
             delete_source_after: bool):
    source_root = routed_dir / "train"
    if not source_root.is_dir():
        raise FileNotFoundError(f"{source_root} does not exist; route the training archive first")

    grouped = defaultdict(list)
    for label, path in source_files(source_root):
        grouped[label].append(path)
    if max_source_files is not None:
        kept = 0
        limited = defaultdict(list)
        for label in sorted(grouped):
            for path in grouped[label]:
                if kept >= max_source_files:
                    break
                limited[label].append(path)
                kept += 1
            if kept >= max_source_files:
                break
        grouped = limited
    if not grouped:
        raise FileNotFoundError(f"No readable audio found under {source_root}")

    train_sources, val_sources = split_sources(grouped, val_fraction, seed)
    logger.info("Original recording split: %d train / %d val across %d species", 
                len(train_sources), len(val_sources), len(grouped))
    written, failures = 0, 0

    for split, entries in (("train", train_sources), ("val", val_sources)):
        variants = variants_per_file if split == "train" else 1
        for index, (label, path) in enumerate(entries, start=1):
            try:
                y, _ = librosa.load(str(path), sr=cfg.SAMPLE_RATE, mono=True)
                if not len(y):
                    raise ValueError("empty audio")
                stem = path.stem
                for variant in range(variants):
                    out_stem = f"{stem}_{'aug' + str(variant).zfill(2) if split == 'train' else 'clean'}"
                    rng = random.Random(f"{seed}:{path}:{variant}")
                    clip = augment(y, rng) if split == "train" else clean_validation_copy(y)
                    write_audio(output_dir / split / label / out_stem, clip, output_format)
                    written += 1
                if delete_source_after:
                    # Only remove the routed input after every requested
                    # output for this source was written successfully. This
                    # is useful on disposable Colab storage, not on a local
                    # archive you intend to preserve or re-use.
                    path.unlink()
            except Exception as exc:
                failures += 1
                logger.warning("Skipping %s (%s)", path, exc)
            if index % 500 == 0:
                logger.info("%s: processed %d/%d source recordings", split, index, len(entries))

    logger.info("Done: wrote %d fixed-length files (%d source failures) to %s", written, failures, output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--routed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants-per-file", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=cfg.VAL_FRACTION)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--format", choices=["wav", "flac"], default="flac")
    parser.add_argument("--max-source-files", type=int, default=None,
                        help="small smoke test only; limits originals before the split")
    parser.add_argument("--delete-source-after", action="store_true",
                        help="delete each routed source only after its generated outputs succeed; "
                             "use only on disposable Colab storage to reduce peak disk usage")
    args = parser.parse_args()
    if args.variants_per_file < 1:
        parser.error("--variants-per-file must be at least 1")
    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction must be between 0 and 1")
    generate(args.routed_dir, args.output_dir, args.variants_per_file,
             args.val_fraction, args.seed, args.format, args.max_source_files,
             args.delete_source_after)


if __name__ == "__main__":
    main()
