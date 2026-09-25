"""Generate reproducible CNN/DSP test clips from the reference recordings.

For every species in ``samples/references``, this writes eight WAV files to
``samples/test``:

    <animal>_original.wav
    <animal>_padded_silence.wav
    <animal>_noised.wav
    <animal>_pitch_increased.wav
    <animal>_pitch_decreased.wav
    <animal>_slowed.wav
    <animal>_speeded.wav
    <animal>_toned.wav

It also writes five direct-concatenation mixtures and five deliberately
overlapping mixtures, choosing a random reference per participating species.
The first mixture of each type always includes ``gunshot``; the rest remain
random. This ensures both multi-animal test modes include at least one
non-animal event without making every mixture artificial.
``generated_manifest.json`` records the source file(s), transformations, and
mixture timing so predictions can be checked against known ground truth.

The generator only overwrites files that it generates. It leaves unrelated
files in the test folder alone. Use ``--clear-generated`` to remove files
listed in the previous manifest before regenerating.

Usage:
    python generate_test_samples.py
    python generate_test_samples.py --seed 7 --num-mixtures 10
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

try:
    from . import ml_config as cfg
except ImportError:
    import ml_config as cfg


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "samples" / "test"
MANIFEST_NAME = "generated_manifest.json"

# Fixed variant parameters make the files comparable between runs; the seed
# only decides which reference recording is used for each generated sample.
PAD_SECONDS = 1.0
NOISE_SNR_DB = 10.0
PITCH_STEPS = 3.0
SLOW_RATE = 0.8
SPEED_RATE = 1.25
TONE_FREQUENCY_HZ = 1000.0
TONE_LEVEL_DB = -12.0
MAX_PEAK = 0.95
REQUIRED_MULTI_SPECIES = "gunshot"


def species_for(path: Path) -> str:
    """Map reference names such as ``cat_003.mp3`` to ``cat``."""
    return path.stem.split("_", maxsplit=1)[0]


def discover_references(references_dir: Path):
    grouped = defaultdict(list)
    for path in sorted(references_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            grouped[species_for(path)].append(path)
    return dict(sorted(grouped.items()))


def load_audio(path: Path) -> np.ndarray:
    waveform, _ = librosa.load(str(path), sr=cfg.SAMPLE_RATE, mono=True)
    if waveform.size == 0:
        raise ValueError("contains no audio samples")
    return waveform.astype(np.float32)


def peak_limit(waveform: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > MAX_PEAK:
        waveform = waveform * (MAX_PEAK / peak)
    return waveform.astype(np.float32)


def add_noise(waveform: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add deterministic white noise at the configured signal-to-noise ratio."""
    noise = rng.normal(0.0, 1.0, size=len(waveform)).astype(np.float32)
    signal_rms = float(np.sqrt(np.mean(waveform ** 2)))
    noise_rms = float(np.sqrt(np.mean(noise ** 2)))
    if signal_rms == 0.0 or noise_rms == 0.0:
        return waveform.copy()
    target_noise_rms = signal_rms / (10 ** (NOISE_SNR_DB / 20.0))
    return peak_limit(waveform + noise * (target_noise_rms / noise_rms))


def pitch_shift(waveform: np.ndarray, steps: float) -> np.ndarray:
    try:
        shifted = librosa.effects.pitch_shift(
            y=waveform, sr=cfg.SAMPLE_RATE, n_steps=steps
        )
    except TypeError:  # Compatibility with older librosa versions.
        shifted = librosa.effects.pitch_shift(waveform, cfg.SAMPLE_RATE, steps)
    return peak_limit(np.asarray(shifted))


def time_stretch(waveform: np.ndarray, rate: float) -> np.ndarray:
    try:
        stretched = librosa.effects.time_stretch(y=waveform, rate=rate)
    except TypeError:  # Compatibility with older librosa versions.
        stretched = librosa.effects.time_stretch(waveform, rate)
    return peak_limit(np.asarray(stretched))


def add_fixed_tone(waveform: np.ndarray) -> np.ndarray:
    """Overlay a 1 kHz sine tone at a fixed -12 dB RMS relative level."""
    signal_rms = float(np.sqrt(np.mean(waveform ** 2)))
    if signal_rms == 0.0:
        return waveform.copy()
    tone_rms = signal_rms * (10 ** (TONE_LEVEL_DB / 20.0))
    tone_amplitude = tone_rms * np.sqrt(2.0)
    time = np.arange(len(waveform), dtype=np.float32) / cfg.SAMPLE_RATE
    tone = tone_amplitude * np.sin(2.0 * np.pi * TONE_FREQUENCY_HZ * time)
    return peak_limit(waveform + tone)


def generate_variant(waveform: np.ndarray, variant: str, rng: np.random.Generator):
    """Apply one named test transformation to a source recording."""
    pad = np.zeros(int(PAD_SECONDS * cfg.SAMPLE_RATE), dtype=np.float32)
    if variant == "original":
        return waveform.copy()
    if variant == "padded_silence":
        return np.concatenate((pad, waveform, pad))
    if variant == "noised":
        return add_noise(waveform, rng)
    if variant == "pitch_increased":
        return pitch_shift(waveform, PITCH_STEPS)
    if variant == "pitch_decreased":
        return pitch_shift(waveform, -PITCH_STEPS)
    if variant == "slowed":
        return time_stretch(waveform, SLOW_RATE)
    if variant == "speeded":
        return time_stretch(waveform, SPEED_RATE)
    if variant == "toned":
        return add_fixed_tone(waveform)
    raise ValueError(f"Unknown variant: {variant}")


def write_wav(path: Path, waveform: np.ndarray):
    """Use float WAV so the test signal is not silently quantized or clipped."""
    sf.write(str(path), waveform.astype(np.float32), cfg.SAMPLE_RATE, subtype="FLOAT")


def pick_events(
    grouped,
    rng: random.Random,
    events_per_mixture: int,
    required_species: str | None = None,
):
    """Choose distinct species, optionally forcing one into the mixture."""
    labels = list(grouped)
    count = min(events_per_mixture, len(labels))
    if required_species is None:
        selected_labels = rng.sample(labels, count)
    else:
        if required_species not in grouped:
            raise ValueError(
                f"Required multi-sample species '{required_species}' has no reference files"
            )
        other_labels = [label for label in labels if label != required_species]
        selected_labels = [required_species] + rng.sample(other_labels, count - 1)
    return [(label, rng.choice(grouped[label])) for label in selected_labels]


def load_events(events):
    return [(label, path, load_audio(path)) for label, path in events]


def build_concatenated_mixture(events):
    """Place source clips end-to-end without any overlap or inserted gap."""
    loaded_events = load_events(events)
    position = 0
    manifest_events = []
    waveforms = []
    for species, path, waveform in loaded_events:
        end = position + len(waveform)
        manifest_events.append({
            "species": species,
            "source": path.name,
            "start_seconds": round(position / cfg.SAMPLE_RATE, 4),
            "end_seconds": round(end / cfg.SAMPLE_RATE, 4),
        })
        waveforms.append(waveform)
        position = end
    return peak_limit(np.concatenate(waveforms)), manifest_events


def build_overlapping_mixture(events, rng: np.random.Generator):
    """Mix clips with starts confined to half the shortest clip's duration.

    Therefore every selected pair overlaps in time, while each mixture remains
    compact enough to exercise multi-animal window aggregation.
    """
    loaded_events = load_events(events)
    shortest = min(len(waveform) for _, _, waveform in loaded_events)
    latest_start = max(1, shortest // 2)
    starts = [int(rng.integers(0, latest_start + 1)) for _ in loaded_events]
    total_length = max(start + len(waveform) for start, (_, _, waveform) in zip(starts, loaded_events))
    mixture = np.zeros(total_length, dtype=np.float32)
    manifest_events = []

    for start, (species, path, waveform) in zip(starts, loaded_events):
        end = start + len(waveform)
        mixture[start:end] += waveform
        manifest_events.append({
            "species": species,
            "source": path.name,
            "start_seconds": round(start / cfg.SAMPLE_RATE, 4),
            "end_seconds": round(end / cfg.SAMPLE_RATE, 4),
        })
    return peak_limit(mixture), manifest_events


def remove_previous_generated_files(output_dir: Path):
    manifest_path = output_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        return 0
    try:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0

    removed = 0
    for name in previous.get("generated_files", []):
        path = output_dir / name
        if path.is_file():
            path.unlink()
            removed += 1
    manifest_path.unlink(missing_ok=True)
    return removed


def generate(references_dir: Path, output_dir: Path, seed: int, num_mixtures: int, events_per_mixture: int):
    if not references_dir.is_dir():
        raise FileNotFoundError(f"References folder not found: {references_dir}")
    if num_mixtures < 1:
        raise ValueError("num_mixtures must be at least 1")
    if events_per_mixture < 2:
        raise ValueError("events_per_mixture must be at least 2")

    grouped = discover_references(references_dir)
    if not grouped:
        raise FileNotFoundError(f"No supported audio files found in {references_dir}")
    if len(grouped) < 2:
        raise ValueError("At least two species are needed to create mixed samples")
    if REQUIRED_MULTI_SPECIES not in grouped:
        raise ValueError(
            f"References must include '{REQUIRED_MULTI_SPECIES}' so each multi-sample "
            "test mode has a guaranteed gunshot case"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    source_rng = random.Random(seed)
    noise_rng = np.random.default_rng(seed)
    manifest = {
        "sample_rate": cfg.SAMPLE_RATE,
        "seed": seed,
        "variant_parameters": {
            "pad_seconds": PAD_SECONDS,
            "noise_snr_db": NOISE_SNR_DB,
            "pitch_steps": PITCH_STEPS,
            "slow_rate": SLOW_RATE,
            "speed_rate": SPEED_RATE,
            "tone_frequency_hz": TONE_FREQUENCY_HZ,
            "tone_level_db": TONE_LEVEL_DB,
        },
        "samples": {},
        "generated_files": [],
    }

    for species, paths in grouped.items():
        for variant in (
            "original", "padded_silence", "noised", "pitch_increased",
            "pitch_decreased", "slowed", "speeded", "toned",
        ):
            source = source_rng.choice(paths)
            waveform = load_audio(source)
            generated = generate_variant(waveform, variant, noise_rng)
            filename = f"{species}_{variant}.wav"
            write_wav(output_dir / filename, generated)
            manifest["generated_files"].append(filename)
            manifest["samples"][filename] = {
                "kind": "single_species_variant",
                "species": [species],
                "source": source.name,
                "variant": variant,
                "duration_seconds": round(len(generated) / cfg.SAMPLE_RATE, 4),
            }
            print(f"wrote {filename:<35s} source={source.name}")

    for mixture_index in range(1, num_mixtures + 1):
        required_species = (
            REQUIRED_MULTI_SPECIES if mixture_index == 1 else None
        )
        events = pick_events(
            grouped, source_rng, events_per_mixture, required_species
        )
        waveform, mixture_events = build_concatenated_mixture(events)
        filename = f"multi_nonoverlapping_{mixture_index:02d}.wav"
        write_wav(output_dir / filename, waveform)
        manifest["generated_files"].append(filename)
        manifest["samples"][filename] = {
            "kind": "non_overlapping_concatenation",
            "species": [event["species"] for event in mixture_events],
            "events": mixture_events,
        }
        print(f"wrote {filename:<35s} species={manifest['samples'][filename]['species']}")

    for mixture_index in range(1, num_mixtures + 1):
        required_species = (
            REQUIRED_MULTI_SPECIES if mixture_index == 1 else None
        )
        events = pick_events(
            grouped, source_rng, events_per_mixture, required_species
        )
        waveform, mixture_events = build_overlapping_mixture(events, noise_rng)
        filename = f"multi_overlapping_{mixture_index:02d}.wav"
        write_wav(output_dir / filename, waveform)
        manifest["generated_files"].append(filename)
        manifest["samples"][filename] = {
            "kind": "overlapping_mix",
            "species": [event["species"] for event in mixture_events],
            "events": mixture_events,
        }
        print(f"wrote {filename:<35s} species={manifest['samples'][filename]['species']}")

    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {len(manifest['generated_files'])} audio files to {output_dir}")
    print(f"Ground truth manifest: {manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references-dir", type=Path, default=cfg.REFERENCES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-mixtures", type=int, default=5,
                        help="Number of non-overlapping AND overlapping mixtures to generate.")
    parser.add_argument("--events-per-mixture", type=int, default=3,
                        help="Number of different species in each mixture.")
    parser.add_argument("--clear-generated", action="store_true",
                        help="Remove files listed in the prior generated manifest first.")
    args = parser.parse_args()

    if args.clear_generated:
        removed = remove_previous_generated_files(args.output_dir)
        print(f"Removed {removed} previously generated audio file(s).")
    generate(
        args.references_dir,
        args.output_dir,
        args.seed,
        args.num_mixtures,
        args.events_per_mixture,
    )


if __name__ == "__main__":
    main()
