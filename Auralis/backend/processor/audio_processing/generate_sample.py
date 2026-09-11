"""
generate_test_samples.py

Builds synthetic "sampleN.{mp3,wav}" mixtures for test_species_detection.py
out of whatever's sitting in samples/references/ (any mix of .mp3 and .wav).

Each generated sample:
  - picks 1-3 reference species at random (no repeats within one sample)
  - applies a random pitch shift, time stretch, and gain to each one
  - places each modified clip at a random offset inside a longer canvas
    (so detection has to find the call inside a bigger mixture, not just
    match a whole clip 1:1)
  - adds white or pink background noise at a random SNR (or none)
  - peak-normalizes and exports at a randomly chosen sample rate, in
    either .mp3 or .wav -- so the pipeline's decode step (and your own
    resampler) actually gets exercised across formats/rates, not just
    the one you happened to develop against

Writes samples/ground_truth.json mapping each generated filename to the
species actually mixed into it (with offsets), and prints a ready-to-paste
EXPECTED dict for test_species_detection.py.

Requires: numpy, scipy, librosa (already project dependencies), and the
system `ffmpeg` binary on PATH for mp3 export. wav-only export still works
without ffmpeg.
"""

import argparse
import json
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import lfilter

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = PROJECT_ROOT / "samples"
REFERENCES_DIR = SAMPLES_DIR / "references"

REF_EXTENSIONS = (".mp3", ".wav")
OUTPUT_SAMPLE_RATES = [16000, 22050, 32000, 44100, 48000]
PITCH_STEPS_RANGE = (-4.0, 4.0)          # semitones
TIME_STRETCH_RANGE = (0.8, 1.25)         # rate multiplier
GAIN_RANGE = (0.6, 1.4)
SNR_DB_RANGE = (5.0, 25.0)               # lower = noisier
CANVAS_DURATION_RANGE = (6.0, 12.0)      # seconds
WORKING_SR = 22050                       # internal rate used while mixing


def discover_reference_files():
    files = [
        p for p in REFERENCES_DIR.iterdir()
        if p.suffix.lower() in REF_EXTENSIONS
    ]
    return sorted(files)


def load_mono(path: Path, sr: int) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def make_pink_noise(n_samples: int) -> np.ndarray:
    """Rough pink-noise approximation via an IIR filter over white noise
    (standard 1/f-ish approximation coefficients)."""
    white = np.random.normal(0, 1, n_samples + 2000).astype(np.float64)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    pink = lfilter(b, a, white)
    return pink[2000:2000 + n_samples].astype(np.float32)


def add_noise(signal: np.ndarray, snr_db: float, noise_type: str) -> np.ndarray:
    if noise_type == "none":
        return signal
    if noise_type == "pink":
        noise = make_pink_noise(len(signal))
    else:
        noise = np.random.normal(0, 1, len(signal)).astype(np.float32)

    signal_power = np.mean(signal ** 2) + 1e-12
    noise_power = np.mean(noise ** 2) + 1e-12
    target_noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = noise * np.sqrt(target_noise_power / noise_power)
    return signal + noise


def modify_clip(y: np.ndarray, sr: int) -> np.ndarray:
    n_steps = random.uniform(*PITCH_STEPS_RANGE)
    rate = random.uniform(*TIME_STRETCH_RANGE)
    gain = random.uniform(*GAIN_RANGE)

    try:
        y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
    except TypeError:
        # older librosa signature: pitch_shift(y, sr, n_steps)
        y = librosa.effects.pitch_shift(y, sr, n_steps)

    try:
        y = librosa.effects.time_stretch(y=y, rate=rate)
    except TypeError:
        y = librosa.effects.time_stretch(y, rate)

    return (y * gain).astype(np.float32)


def build_mixture(ref_files, working_sr: int):
    """Returns (mixed_signal, list of {species, start_sec, dur_sec})."""
    n_events = random.randint(1, min(3, len(ref_files)))
    chosen = random.sample(ref_files, k=n_events)

    canvas_dur = random.uniform(*CANVAS_DURATION_RANGE)
    canvas_len = int(round(canvas_dur * working_sr))
    canvas = np.zeros(canvas_len, dtype=np.float32)

    events = []
    for ref_path in chosen:
        species = ref_path.stem
        y = load_mono(ref_path, working_sr)
        y = modify_clip(y, working_sr)

        if len(y) >= canvas_len:
            # clip is longer than the canvas -- crop a random window of it
            start_in_clip = random.randint(0, len(y) - canvas_len)
            y = y[start_in_clip:start_in_clip + canvas_len]
            start_sample = 0
        else:
            start_sample = random.randint(0, canvas_len - len(y))

        canvas[start_sample:start_sample + len(y)] += y
        events.append({
            "species": species,
            "start_sec": round(start_sample / working_sr, 3),
            "duration_sec": round(len(y) / working_sr, 3),
        })

    noise_type = random.choice(["white", "pink", "none"])
    snr_db = random.uniform(*SNR_DB_RANGE)
    canvas = add_noise(canvas, snr_db, noise_type)

    peak = np.max(np.abs(canvas)) if len(canvas) else 0.0
    if peak > 0:
        canvas = canvas / peak * 0.95

    return canvas.astype(np.float32), events, noise_type, round(snr_db, 2)


def export_wav(signal: np.ndarray, sr: int, out_path: Path):
    sf.write(str(out_path), signal, sr)


def export_mp3(signal: np.ndarray, sr: int, out_path: Path):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH -- required for mp3 export. "
            "Install ffmpeg, or rerun with --formats wav."
        )
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        sf.write(str(tmp_path), signal, sr)
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(tmp_path),
                "-codec:a", "libmp3lame", "-qscale:a", "2",
                str(out_path),
            ],
            check=True,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--formats", nargs="+", choices=["mp3", "wav"], default=["mp3", "wav"],
        help="File formats to cycle through across the generated samples.",
    )
    parser.add_argument(
        "--clear-existing", action="store_true",
        help="Delete existing sample*.mp3/.wav in samples/ before generating.",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    ref_files = discover_reference_files()
    if not ref_files:
        raise SystemExit(
            f"No reference files found in {REFERENCES_DIR}. "
            f"Add some .mp3/.wav references there first."
        )
    print(f"Found {len(ref_files)} reference file(s): "
          f"{', '.join(p.name for p in ref_files)}")

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    if args.clear_existing:
        for p in SAMPLES_DIR.glob("sample*.mp3"):
            p.unlink()
        for p in SAMPLES_DIR.glob("sample*.wav"):
            p.unlink()

    ground_truth = {}

    for i in range(1, args.num_samples + 1):
        out_format = args.formats[(i - 1) % len(args.formats)]
        out_sr = random.choice(OUTPUT_SAMPLE_RATES)

        mixed_at_working_sr, events, noise_type, snr_db = build_mixture(
            ref_files, WORKING_SR
        )

        # Resample from the internal working rate to this sample's chosen
        # output rate so different samples exercise different native rates.
        if out_sr != WORKING_SR:
            mixed = librosa.resample(
                mixed_at_working_sr, orig_sr=WORKING_SR, target_sr=out_sr
            )
        else:
            mixed = mixed_at_working_sr

        stem = f"sample{i}"
        out_path = SAMPLES_DIR / f"{stem}.{out_format}"

        if out_format == "wav":
            export_wav(mixed, out_sr, out_path)
        else:
            export_mp3(mixed, out_sr, out_path)

        species_list = sorted({e["species"] for e in events})
        ground_truth[stem] = {
            "file": out_path.name,
            "sample_rate": out_sr,
            "noise_type": noise_type,
            "snr_db": snr_db,
            "events": events,
            "species": species_list,
        }

        print(f"  wrote {out_path.name:<16s} sr={out_sr:<6d} "
              f"noise={noise_type:<5s} snr={snr_db:5.1f}dB  "
              f"species={species_list}")

    gt_path = SAMPLES_DIR / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"\nWrote ground truth to {gt_path}")

    print("\nPaste this into EXPECTED in test_species_detection.py:\n")
    print("EXPECTED = {")
    for stem, info in ground_truth.items():
        if len(info["species"]) == 1:
            value = f'"{info["species"][0]}"'
        else:
            value = "{" + ", ".join(f'"{s}"' for s in info["species"]) + "}"
        print(f'    "{stem}": {value},')
    print("}")


if __name__ == "__main__":
    main()
