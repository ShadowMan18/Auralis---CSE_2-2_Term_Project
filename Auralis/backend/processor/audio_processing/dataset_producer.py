"""
dataset_producer.py

Builds the training set for the phase-1 CNN from a folder of reference
recordings (filename stem = class label; several recordings per class are
named cat_001.wav, cat_002.wav, ... -- the label is the part before the first
"_").

WHAT CHANGED vs the first version, and why
------------------------------------------
Your references are very different lengths (0.6 s ... 8 s). The old generator
cropped/zero-padded every clip to 3 s, so a 0.6 s dog bark became ~80% exact
digital silence while an 8 s crow filled the whole window. The CNN could then
score well by learning "how much silence" instead of "which animal" -- and
real audio never contains digital silence. Measured on the old model: the
same sounds scored 88% zero-padded but only 48% on a mild noise floor (cat,
cow and dog fell to 0%).

Now every training clip is built the same way, whatever the reference length:

  1. TRIM + SPLIT: silence is trimmed from each reference; long recordings are
     split into individual calls ("events", max 2.4 s). Every class becomes a
     pool of events, so an 8 s recording no longer dominates a 0.6 s one.
  2. AUGMENT THE EVENT: pitch, tempo, EQ/band-limit and reverb (phone mics,
     rooms).
  3. COMPOSE ON A 3 s CANVAS at a random position -- sometimes cut off by the
     window edge (the API slides windows, so calls straddle boundaries) and
     short calls are sometimes repeated (dogs bark in series).
  4. ADD A REAL NOISE BED over the WHOLE 3 s (white / pink / brown / mains
     hum / wind at a random SNR) -- never digital zeros -- and sometimes a
     quiet call from ANOTHER species in the background.
  5. `_background` clips come from the same bed generator, so "nothing here"
     sounds like the padding of real clips, not like a different world.
  6. BALANCED: every class gets exactly --variants-per-class clips, no matter
     how long its references are.

CROSS-VALIDATION (recommended with only ~5 recordings per class): a single
train/val split means the reported accuracy depends heavily on which ONE
recording got held out. Pass --fold i (i = 0..k-1) to rotate which
recording is held out per class instead of picking randomly; run k folds and
average, or just use cross_validate.py which does this automatically end to
end (build -> train -> aggregate) and prints per-class mean +/- std accuracy.

Noise generation (the beds used both here and to pad short uploads at
inference) lives in noise_bed.py, not in this file -- single source of truth,
see that module's docstring.

HONEST VALIDATION
-----------------
With >= 3 recordings of a class, whole RECORDINGS are held out for validation
(the model never hears them in training), so val accuracy means something.
With fewer, validation clips are augmentations of the same recording(s) and
val accuracy is OPTIMISTIC; the script warns and records this in
dataset_info.json, and train_cnn.py repeats the warning. More real recordings
per species is the biggest accuracy lever there is.

Output layout (unchanged):
    <output_dir>/train/<class>/*.wav   <output_dir>/val/<class>/*.wav
    <output_dir>/dataset_info.json

Usage:
    python dataset_producer.py                       # 200 clips per class
    python dataset_producer.py --variants-per-class 400 --seed 1
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfilt

import ml_config as cfg
import noise_bed as nb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

SR = cfg.SAMPLE_RATE
WIN = int(cfg.CLIP_SECONDS * SR)

# --- event extraction -------------------------------------------------------
TRIM_TOP_DB = 35.0            # silence threshold when trimming a reference
SPLIT_TOP_DB = 30.0           # silence threshold when splitting into calls
MERGE_GAP_SECONDS = 0.30      # bursts closer than this are one call (e.g. a rooster's crow)
MIN_EVENT_SECONDS = 0.05
MAX_EVENT_SECONDS = 0.8 * cfg.CLIP_SECONDS   # always leaves some background around the call

# --- augmentation of the event itself --------------------------------------
PITCH_STEPS_RANGE = (-3.0, 3.0)   # semitones
TEMPO_RATE_RANGE = (0.8, 1.25)
P_PITCH, P_TEMPO, P_EQ, P_REVERB = 0.6, 0.6, 0.3, 0.3

# --- composition ------------------------------------------------------------
P_EDGE_CUT = 0.25                 # call partly outside the window (>= 50% stays inside)
REPEAT_BELOW_SECONDS = 1.0        # calls shorter than this may repeat 1-3x
SNR_DB_RANGE = (3.0, 35.0)        # call level vs noise bed (lower = noisier)
P_DISTRACTOR = 0.25               # quiet call from another species in the background
DISTRACTOR_DB_RANGE = (10.0, 25.0)  # how far below the target call


# ---------------------------------------------------------------------------
# small DSP helpers
# ---------------------------------------------------------------------------

def _load_mono(path):
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    return y.astype(np.float32)


def _normalize_peak(y, target):
    peak = float(np.max(np.abs(y)))
    return y if peak < 1e-9 else (y / peak * target).astype(np.float32)


def _random_eq(y, rng):
    """Band-limit like a phone mic / distance: random low-pass or high-pass."""
    if rng.random() < 0.5:
        sos = butter(2, rng.uniform(3500, 9500), btype="lowpass", fs=SR, output="sos")
    else:
        sos = butter(2, rng.uniform(100, 700), btype="highpass", fs=SR, output="sos")
    return sosfilt(sos, y)


def _reverb(y, rng):
    """Cheap synthetic room: exponentially decaying noise impulse response."""
    n_ir = max(8, int(rng.uniform(0.05, 0.4) * SR))
    ir = rng.standard_normal(n_ir) * np.exp(-np.arange(n_ir) / (n_ir / 5.0))
    ir[0] = 1.0
    wet = fftconvolve(y, ir / np.sqrt(np.sum(ir ** 2)))[:len(y)]
    mix = rng.uniform(0.2, 0.6)
    return (1 - mix) * y + mix * wet


def _augment_event(y, rng):
    if rng.random() < P_PITCH:
        try:
            y = librosa.effects.pitch_shift(y=y, sr=SR, n_steps=float(rng.uniform(*PITCH_STEPS_RANGE)))
        except Exception:
            pass
    if rng.random() < P_TEMPO:
        try:
            y = librosa.effects.time_stretch(y=y, rate=float(rng.uniform(*TEMPO_RATE_RANGE)))
        except Exception:
            pass
    if rng.random() < P_EQ:
        y = _random_eq(y, rng)
    if rng.random() < P_REVERB:
        y = _reverb(y, rng)
    y = np.asarray(y, dtype=np.float32)
    return y if len(y) >= 64 and np.all(np.isfinite(y)) else np.zeros(64, np.float32) + 1e-4


# ---------------------------------------------------------------------------
# reference -> events
# ---------------------------------------------------------------------------

def extract_events(y, trim_top_db=TRIM_TOP_DB, split_top_db=SPLIT_TOP_DB):
    """Trim a reference and split it into individual calls (each <= MAX_EVENT_SECONDS).

    trim_top_db: how much quieter than the loudest point counts as "silence" when
    cutting the ends of the file. split_top_db: same, but for finding gaps BETWEEN
    calls inside the file. Lower = more aggressive (cuts more); a quiet recording
    with soft calls or a noisy room may need these raised (less aggressive) to
    avoid cutting into the actual call. See trim_references.py to check this on
    your own files before committing to a threshold."""
    if len(y) == 0 or float(np.max(np.abs(y))) < 1e-4:
        return []  # empty / silent file: nothing to learn from (trim would keep all of it)
    y, _ = librosa.effects.trim(y, top_db=trim_top_db)
    if len(y) < int(MIN_EVENT_SECONDS * SR):
        return []
    max_len = int(MAX_EVENT_SECONDS * SR)
    if len(y) <= max_len:
        return [y]

    gap = int(MERGE_GAP_SECONDS * SR)
    bursts = []
    for s, e in librosa.effects.split(y, top_db=split_top_db, frame_length=1024, hop_length=256):
        if bursts and s - bursts[-1][1] <= gap:
            bursts[-1][1] = e
        else:
            bursts.append([int(s), int(e)])

    events, min_len = [], int(MIN_EVENT_SECONDS * SR)
    for s, e in bursts:
        pos = s
        while e - pos >= min_len:
            end = min(pos + max_len, e)
            events.append(y[pos:end])
            if end == e:
                break
            pos += max_len // 2          # long call: overlapping 2.4 s pieces
    return events or [y[:max_len]]


# ---------------------------------------------------------------------------
# composing a training clip
# ---------------------------------------------------------------------------

def _place(canvas, ev, start):
    """Add `ev` into `canvas` at `start`; parts outside the canvas are cut off."""
    a, b = max(start, 0), min(start + len(ev), len(canvas))
    if b > a:
        canvas[a:b] += ev[a - start:b - start]


def _event_track(ev, rng):
    """One 3 s track holding the (possibly repeated / edge-cut) event, no noise yet."""
    if len(ev) > WIN:
        s = int(rng.integers(0, len(ev) - WIN + 1))
        ev = ev[s:s + WIN]
    track = np.zeros(WIN, np.float32)
    n_rep = int(rng.choice([1, 1, 2, 3])) if len(ev) < REPEAT_BELOW_SECONDS * SR else 1
    if rng.random() < P_EDGE_CUT:
        start = int(rng.integers(-(len(ev) // 2), WIN - len(ev) // 2))
    else:
        start = int(rng.integers(0, WIN - len(ev) + 1))
    for k in range(n_rep):
        gain = 10 ** (rng.uniform(-3, 3) / 20) if k else 1.0
        _place(track, ev * gain, start)
        start += len(ev) + int(rng.uniform(0.1, 0.6) * SR)
        if start >= WIN:
            break
    return track


def make_species_clip(event, distractor_pool, rng):
    ev = _augment_event(event, rng)
    ev_rms = nb.rms(ev)
    track = _event_track(ev, rng)

    if distractor_pool and rng.random() < P_DISTRACTOR:
        d = _augment_event(distractor_pool[int(rng.integers(len(distractor_pool)))], rng)
        d_track = _event_track(d, rng) * (ev_rms / nb.rms(d)) * 10 ** (-rng.uniform(*DISTRACTOR_DB_RANGE) / 20)
        track = track + d_track

    bed = nb.bed_at_snr(WIN, SR, rng, ev_rms, SNR_DB_RANGE)
    return _normalize_peak(track + bed, rng.uniform(0.3, 0.95))


def make_background_clip(rng):
    x = nb.random_bed(WIN, SR, rng)
    if rng.random() < 0.3:  # two noise sources at once (e.g. hum + wind)
        x = x + rng.uniform(0.2, 1.0) * nb.random_bed(WIN, SR, rng)
    return _normalize_peak(x, rng.uniform(0.1, 0.9))


# ---------------------------------------------------------------------------
# dataset build
# ---------------------------------------------------------------------------

def _class_label_for(path: Path):
    """'cat.mp3' -> 'cat', 'cat_003.wav' -> 'cat'."""
    return path.stem.split("_")[0] if "_" in path.stem else path.stem


def _clear_generated(split_dir: Path):
    """Remove a previous run's clips so old (differently built) files can't mix in."""
    if not split_dir.exists():
        return
    files = [p for p in split_dir.rglob("*") if p.is_file()]
    if any(p.suffix.lower() != ".wav" for p in files):
        raise RuntimeError(f"{split_dir} contains non-wav files; refusing to delete it. "
                           f"Pick an empty --output-dir.")
    logger.warning("Removing %d old clips in %s", len(files), split_dir)
    shutil.rmtree(split_dir)


def _sample_event(sources, rng):
    """Pick a source recording uniformly, then one of its events uniformly, so a
    long recording with many calls doesn't outweigh a short one."""
    _name, events = sources[int(rng.integers(len(sources)))]
    return events[int(rng.integers(len(events)))]


def build_dataset(references_dir: Path, output_dir: Path, variants_per_class: int,
                  val_fraction: float = cfg.VAL_FRACTION, seed: int = 0, val_mode: str = "auto",
                  fold: int | None = None, trim_top_db: float = TRIM_TOP_DB,
                  split_top_db: float = SPLIT_TOP_DB):
    rng = np.random.default_rng(seed)

    files = sorted(p for p in references_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"No audio files found in {references_dir}")
    by_class = {}
    for p in files:
        by_class.setdefault(_class_label_for(p), []).append(p)

    # -- events per source recording, and the train/val split of SOURCES -------
    split_sources, info = {}, {}
    for label, paths in sorted(by_class.items()):
        loaded = []
        for p in paths:
            events = extract_events(_load_mono(p), trim_top_db, split_top_db)
            if not events:
                logger.warning("'%s': %s has no usable audio, skipped", label, p.name)
                continue
            loaded.append((p.name, events))
            logger.info("'%s': %s -> %d event(s), %.2f-%.2fs", label, p.name, len(events),
                        min(len(e) for e in events) / SR, max(len(e) for e in events) / SR)
        if not loaded:
            raise RuntimeError(f"Class '{label}' has no usable reference audio.")

        mode = val_mode if val_mode != "auto" else ("source" if len(loaded) >= 3 else "variant")
        if mode == "source" and len(loaded) < 2:
            mode = "variant"
        if mode == "source":
            if fold is not None:
                # deterministic rotation: recording (fold % n) is held out. Used by
                # cross_validate.py to run leave-one-recording-out over every recording
                # in turn, instead of one random (and possibly lucky/unlucky) split.
                held = fold % len(loaded)
                val_src = [loaded[held]]
                train_src = [loaded[i] for i in range(len(loaded)) if i != held]
            else:
                order = [int(i) for i in rng.permutation(len(loaded))]
                n_val_src = min(max(1, int(round(len(loaded) * val_fraction))), len(loaded) - 1)
                val_src = [loaded[i] for i in order[:n_val_src]]
                train_src = [loaded[i] for i in order[n_val_src:]]
        else:
            train_src = val_src = loaded
            logger.warning("'%s': %d recording(s) -> validation clips come from the SAME recording(s) as "
                           "training, so val accuracy is OPTIMISTIC for this class. Add more recordings "
                           "(%s_001.wav, %s_002.wav, ...) for an honest validation.",
                           label, len(loaded), label, label)
        split_sources[label] = {"train": train_src, "val": val_src}
        info[label] = {"val_mode": mode, "fold": fold,
                       "train_sources": [n for n, _ in train_src],
                       "val_sources": [n for n, _ in val_src]}

    n_val = max(1, int(round(variants_per_class * val_fraction)))
    counts = {"train": variants_per_class - n_val, "val": n_val}

    _clear_generated(output_dir / "train")
    _clear_generated(output_dir / "val")

    written = 0
    for split in ("train", "val"):
        for label in list(by_class) + [cfg.BACKGROUND_LABEL]:
            (output_dir / split / label).mkdir(parents=True, exist_ok=True)

        for label in sorted(by_class):
            own = split_sources[label][split]
            others = [ev for other, s in split_sources.items() if other != label
                      for _n, evs in s[split] for ev in evs]
            for i in range(counts[split]):
                clip = make_species_clip(_sample_event(own, rng), others, rng)
                sf.write(str(output_dir / split / label / f"{label}_{i:04d}.wav"), clip, SR)
                written += 1
            logger.info("%s/%s: %d clips", split, label, counts[split])

        for i in range(counts[split]):
            sf.write(str(output_dir / split / cfg.BACKGROUND_LABEL / f"bg_{i:04d}.wav"),
                     make_background_clip(rng), SR)
            written += 1
        logger.info("%s/%s: %d clips", split, cfg.BACKGROUND_LABEL, counts[split])

    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump({"seed": seed, "variants_per_class": variants_per_class,
                   "val_fraction": val_fraction, "fold": fold, "classes": info}, f, indent=2)
    logger.info("Done: %d clips written to %s", written, output_dir)
    return sorted(by_class)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--references-dir", type=Path, default=cfg.REFERENCES_DIR)
    parser.add_argument("--output-dir", type=Path, default=cfg.DATASET_DIR)
    parser.add_argument("--variants-per-class", "--variants-per-clip", dest="variants_per_class",
                        type=int, default=200,
                        help="clips generated per class (train+val), regardless of reference length. "
                             "(--variants-per-clip is the old name.)")
    parser.add_argument("--val-fraction", type=float, default=cfg.VAL_FRACTION)
    parser.add_argument("--val-mode", choices=["auto", "source", "variant"], default="auto",
                        help="source = hold out whole recordings; variant = hold out augmentations; "
                             "auto = source when a class has >= 3 recordings")
    parser.add_argument("--trim-top-db", type=float, default=TRIM_TOP_DB,
                        help=f"leading/trailing silence threshold, dB below peak (default "
                             f"{TRIM_TOP_DB}). Lower = trims more aggressively. Check the effect on "
                             f"your files first with trim_references.py.")
    parser.add_argument("--split-top-db", type=float, default=SPLIT_TOP_DB,
                        help=f"threshold for splitting a recording into separate calls, dB below peak "
                             f"(default {SPLIT_TOP_DB}). Lower = splits more readily / merges less.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, default=None,
                        help="deterministically hold out recording (fold %% n_recordings) per class "
                             "instead of a random split. For manual k-fold; cross_validate.py automates "
                             "this.")
    args = parser.parse_args()

    classes = build_dataset(args.references_dir, args.output_dir, args.variants_per_class,
                            args.val_fraction, args.seed, args.val_mode, args.fold,
                            args.trim_top_db, args.split_top_db)
    logger.info("Classes: %s + %s", classes, cfg.BACKGROUND_LABEL)


if __name__ == "__main__":
    main()