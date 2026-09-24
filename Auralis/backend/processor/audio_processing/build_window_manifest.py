"""
build_window_manifest.py

Turns <routed_dir>/<split>/<class_slug>/*.{wav,flac,mp3,ogg,...} (produced
by inatsounds_route.py) into a window manifest: one row per (source file,
start_time) pair, NOT one file per window.

WHY A MANIFEST INSTEAD OF WRITING OUT EVERY WINDOW AS ITS OWN FILE:
Striding 3s windows by 1.5s roughly doubles the number of examples per
recording (more for longer recordings). Materializing every window as a
new audio file would multiply disk usage on top of the already-large
routed corpus for no benefit -- train_cnn_v2.py's Dataset reads each
window directly out of the original file at the given offset
(soundfile supports partial reads via start/frames, so this is a cheap
seek + short decode, not a full-file decode per window).

RECORDING-LEVEL SPLIT, NOT WINDOW-LEVEL SPLIT:
val_fraction is applied by RECORDING (file), not by window. Two windows
from the same recording overlap by half and share the same mic/individual/
background-noise fingerprint, so if sibling windows of one recording were
allowed to land on both sides of the split, val accuracy would partly
measure "did the model memorize this exact recording" rather than
generalization -- the same failure mode the whole phase-1 -> phase-2
migration was meant to fix, just reintroduced at the window level instead
of the clip level.

Usage:
    # If only train.tar.gz is routed, create train+val without leakage:
    python build_window_manifest.py --routed-dir inat_routed --split train \\
        --also-split-val
    # If train.tar.gz and val.tar.gz are both routed, run once per split.
"""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import soundfile as sf

try:  # package import (Flask) and direct-script import both work
    from . import ml_config_v2 as cfg
except ImportError:
    import ml_config_v2 as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
N_BACKGROUND_RECORDINGS = 120   # synthesized negative-class clips (see make_background_clips)


def window_starts(duration_s: float, window_s: float, stride_s: float):
    """Start offsets (seconds) for windows across a recording. A recording
    shorter than one window still gets exactly one window starting at 0
    (train_cnn_v2.py pads it out); this only ever under- rather than
    over-covers the tail (last partial window past duration is dropped,
    matching the official iNatSounds dense_prediction convention of
    padding to fit whole windows rather than emitting a short one)."""
    if duration_s <= window_s:
        return [0.0]
    starts = []
    t = 0.0
    while t + window_s <= duration_s:
        starts.append(t)
        t += stride_s
    if not starts:
        starts = [0.0]
    return starts


def scan_class_dir(class_dir: Path):
    return sorted(p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)


def make_background_clips(out_dir: Path, n: int, seed: int = 0):
    """Synthesizes a `_background` negative class (silence / white / pink
    noise) the same way phase-1's dataset_producer.py did, so the phase-2
    classifier also has a "none of the above" option -- a closed-set
    classifier otherwise always confidently picks a species, even for
    silence, wind, or an unrelated sound. Real ambient/room-tone/wind
    field recordings are a better source than synthesized noise if you
    have them; point --background-dir at that folder instead of using
    this generator."""
    random.seed(seed)
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    sr = cfg.SAMPLE_RATE
    seconds = 6.0  # long enough to yield a couple of strided windows each
    written = []
    for i in range(n):
        n_samples = int(seconds * sr)
        kind = random.choice(["silence", "white", "pink", "quiet_white"])
        if kind == "silence":
            y = np.random.normal(0, 0.001, n_samples).astype(np.float32)
        elif kind == "white":
            y = np.random.normal(0, random.uniform(0.05, 0.3), n_samples).astype(np.float32)
        elif kind == "quiet_white":
            y = np.random.normal(0, random.uniform(0.01, 0.05), n_samples).astype(np.float32)
        else:
            white = np.random.normal(0, 1, n_samples).astype(np.float32)
            pink = np.cumsum(white)
            pink -= np.mean(pink)
            pink = pink / (np.max(np.abs(pink)) + 1e-9) * random.uniform(0.05, 0.3)
            y = pink.astype(np.float32)
        path = out_dir / f"bg_{i:04d}.wav"
        sf.write(str(path), y, sr)
        written.append(path)
    return written


def build_manifest(routed_split_dir: Path, background_dir: Path | None,
                    window_s: float, stride_s: float):
    rows = []
    class_dirs = sorted(p for p in routed_split_dir.iterdir() if p.is_dir())
    for class_dir in class_dirs:
        slug = class_dir.name
        files = scan_class_dir(class_dir)
        for path in files:
            try:
                info = sf.info(str(path))
            except Exception as e:
                logger.warning("skipping unreadable file %s (%s)", path, e)
                continue
            duration = info.frames / float(info.samplerate)
            for start in window_starts(duration, window_s, stride_s):
                rows.append({"path": str(path), "start_s": round(start, 3),
                             "label": slug, "recording_id": str(path)})
        logger.info("class '%s': %d recordings -> windows so far in class = %d",
                     slug, len(files), sum(1 for r in rows if r["label"] == slug))

    if background_dir is not None and background_dir.exists():
        files = scan_class_dir(background_dir)
        for path in files:
            info = sf.info(str(path))
            duration = info.frames / float(info.samplerate)
            for start in window_starts(duration, window_s, stride_s):
                rows.append({"path": str(path), "start_s": round(start, 3),
                             "label": cfg.BACKGROUND_LABEL, "recording_id": str(path)})
        logger.info("class '%s': %d recordings", cfg.BACKGROUND_LABEL, len(files))

    return rows


def split_by_recording(rows, val_fraction: float, seed: int = 0):
    """Splits at the recording (file) level -- see module docstring."""
    rng = random.Random(seed)
    by_class_recordings = {}
    for r in rows:
        by_class_recordings.setdefault(r["label"], set()).add(r["recording_id"])

    val_recordings = set()
    for label, recs in by_class_recordings.items():
        recs = sorted(recs)
        rng.shuffle(recs)
        n_val = max(1, int(round(len(recs) * val_fraction))) if len(recs) > 1 else 0
        val_recordings.update(recs[:n_val])

    train_rows = [r for r in rows if r["recording_id"] not in val_recordings]
    val_rows = [r for r in rows if r["recording_id"] in val_recordings]
    return train_rows, val_rows


def write_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--routed-dir", type=Path, default=cfg.ROUTED_AUDIO_DIR)
    parser.add_argument("--split", required=True, choices=["train", "val"],
                         help="which routed split folder to read (<routed-dir>/<split>)")
    parser.add_argument("--background-dir", type=Path, default=None,
                         help="folder of real background/negative recordings; if omitted, "
                              "synthesized noise/silence background clips are generated")
    parser.add_argument("--synthesize-background", action="store_true", default=True)
    parser.add_argument("--no-synthesize-background", dest="synthesize_background", action="store_false")
    parser.add_argument("--out-dir", type=Path, default=cfg.MANIFEST_DIR)
    parser.add_argument("--val-fraction", type=float, default=cfg.VAL_FRACTION,
                         help="only used when --split train and --also-split-val is set")
    parser.add_argument("--also-split-val", action="store_true",
                         help="carve a val manifest out of the train split by recording "
                              "(use this if you're only routing train.tar.gz and don't "
                              "also have val.tar.gz routed separately)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    split_dir = args.routed_dir / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"{split_dir} does not exist -- run inatsounds_route.py first")

    background_dir = args.background_dir
    if background_dir is None and args.synthesize_background:
        # A separately generated train/val background set is essential: using
        # the same synthetic files in both manifests is validation leakage.
        # In --also-split-val mode the recording-level split below keeps one
        # shared background pool safely partitioned instead.
        background_name = "shared" if args.split == "train" and args.also_split_val else args.split
        background_dir = args.out_dir / "_synth_background" / background_name
        if not background_dir.exists() or not any(background_dir.iterdir()):
            logger.info("Synthesizing %d background recordings into %s", N_BACKGROUND_RECORDINGS, background_dir)
            make_background_clips(background_dir, N_BACKGROUND_RECORDINGS, args.seed)

    rows = build_manifest(split_dir, background_dir, cfg.WINDOW_SECONDS, cfg.WINDOW_STRIDE_SECONDS)
    logger.info("Built %d window rows across %d classes", len(rows),
                len({r["label"] for r in rows}))

    if args.split == "train" and args.also_split_val:
        train_rows, val_rows = split_by_recording(rows, args.val_fraction, args.seed)
        write_jsonl(train_rows, args.out_dir / "train.jsonl")
        write_jsonl(val_rows, args.out_dir / "val.jsonl")
        logger.info("Wrote %d train / %d val window rows (split by recording)", len(train_rows), len(val_rows))
    else:
        out_path = args.out_dir / f"{args.split}.jsonl"
        write_jsonl(rows, out_path)
        logger.info("Wrote %s (%d window rows)", out_path, len(rows))


if __name__ == "__main__":
    main()
