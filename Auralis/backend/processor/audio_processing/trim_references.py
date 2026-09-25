"""
trim_references.py

Trims LEADING and TRAILING silence from every reference recording in
--references-dir and saves each one back under its ORIGINAL filename (same
name, same extension where possible -- see the M4A note below). Internal
pauses inside a recording are left alone; only the dead air at the very
start and end is cut. (dataset_producer.py separately splits a recording
into individual calls when it builds the dataset -- that's a different step
from this one and unaffected by it.)

SAFETY: this overwrites your reference files, and they're often the only
copy you have of a given animal. Before the FIRST time a file is trimmed,
its untouched original is copied to --backup-dir (default: a sibling folder
next to --references-dir, so it's not mistaken for another reference by
dataset_producer.py). A file that already has a backup is treated as
already trimmed and is SKIPPED on later runs -- both so re-running this
script is a no-op by default, and because MP3 is lossy: decoding and
re-encoding an already-trimmed MP3 again would throw away a little more
audio quality each time for no reason. Use --force to reprocess a file
anyway (it restores from the backup first, so quality never compounds
across repeated --force runs either).

M4A NOTE: libsndfile (what this script writes audio with) can't encode
M4A/AAC. An .m4a reference is trimmed and saved as <name>.wav instead, and
the original .m4a is moved into the backup folder (not left in
--references-dir) so dataset_producer.py doesn't see both and count them as
two separate recordings of the same animal.

Run this BEFORE dataset_producer.py, especially the first time you add new
recordings. Standalone: only needs ml_config.py and silence_trim.py alongside
it, no other pipeline files.

Per-file warnings:A
  - "NOTHING KEPT": the whole file registered as silence at this threshold
    -- wrong file, or --trim-top-db needs to be higher (less aggressive).
  - ">90% removed": likely trimming into the actual call (soft/distant
    recording) -- listen to the saved file, or raise --trim-top-db.

Usage:
    python trim_references.py
    python trim_references.py --trim-top-db 30
    python trim_references.py --force --trim-top-db 30   # redo with a new threshold
"""

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

try:  # Allow both ``python trim_references.py`` and package imports.
    from . import ml_config as cfg
    from .silence_trim import (
        DEFAULT_TRIM_TOP_DB,
        SILENCE_PEAK_EPSILON,
        trim_leading_trailing_silence,
    )
except ImportError:
    import ml_config as cfg
    from silence_trim import (
        DEFAULT_TRIM_TOP_DB,
        SILENCE_PEAK_EPSILON,
        trim_leading_trailing_silence,
    )

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Standalone by design: this used to import ALLOWED_EXTENSIONS / TRIM_TOP_DB / SR /
# _class_label_for from dataset_producer.py, on the theory that if those ever changed
# there they'd automatically stay in sync here too. In practice that just meant this
# script broke with an ImportError whenever the two files' versions drifted even
# slightly apart. These settings change rarely; the actual trim operation and
# threshold now come from silence_trim.py so online preprocessing and reference
# preparation cannot drift apart. If you deliberately change ALLOWED_EXTENSIONS
# or the silence-threshold default in dataset_producer.py, update the copies below too.
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
TRIM_TOP_DB = DEFAULT_TRIM_TOP_DB
SR = cfg.SAMPLE_RATE


def _class_label_for(path: Path):
    """'cat.mp3' -> 'cat', 'cat_003.wav' -> 'cat'. Copy of dataset_producer.py's
    function of the same name -- see the note above on why this is duplicated."""
    return path.stem.split("_")[0] if "_" in path.stem else path.stem

# A file trimmed past this fraction gets flagged: possibly the threshold is too
# aggressive for this recording (quiet call, distant mic, soft fade), or the
# file genuinely contains almost no usable sound (wrong file, mostly silence).
WARN_TRIMMED_FRACTION = 0.90

# soundfile/libsndfile can write these; anything else (M4A/AAC) falls back to WAV.
_WRITABLE_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg"}


def _default_backup_dir(references_dir: Path) -> Path:
    return references_dir.parent / f"{references_dir.name}_untrimmed_backup"


def trim_one(path: Path, backup_dir: Path, trim_top_db: float, force: bool):
    """Returns a dict with the per-file report, or None if skipped (already
    backed up and not --force)."""
    backup_path = backup_dir / path.name
    already_done = backup_path.is_file()
    if already_done and not force:
        return None

    source = backup_path if (already_done and force) else path  # --force re-trims from the ORIGINAL
    y, _sr = librosa.load(str(source), sr=SR, mono=True)
    orig_secs = len(y) / SR

    if len(y) == 0 or float(np.max(np.abs(y))) < SILENCE_PEAK_EPSILON:
        return {"name": path.name, "orig_secs": orig_secs, "kept_secs": 0.0, "removed_frac": 1.0,
               "status": "NOTHING KEPT: file is silent at any threshold -- check it's the right file"}

    trimmed, index = trim_leading_trailing_silence(y, top_db=trim_top_db)
    kept_secs = len(trimmed) / SR
    removed_frac = 1.0 - (kept_secs / orig_secs) if orig_secs > 0 else 1.0
    lead_secs, trail_secs = index[0] / SR, (len(y) - index[1]) / SR

    status = ""
    if len(trimmed) < int(0.02 * SR):
        status = "NOTHING KEPT: check --trim-top-db (too aggressive) or the file itself"
    elif removed_frac > WARN_TRIMMED_FRACTION:
        status = f"{removed_frac:.0%} removed: may be cutting into a quiet call -- listen to it"

    if not already_done:
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_path)  # untouched original, kept before any overwrite

    if status.startswith("NOTHING KEPT"):
        # don't overwrite the reference with near-silence; leave the original in place for review
        return {"name": path.name, "orig_secs": orig_secs, "kept_secs": kept_secs,
               "removed_frac": removed_frac, "status": status, "written": False}

    write_path = path
    if path.suffix.lower() not in _WRITABLE_SUFFIXES:  # M4A etc: fall back to .wav, same stem
        write_path = path.with_suffix(".wav")
    sf.write(str(write_path), trimmed.astype(np.float32), SR)
    if write_path != path:
        path.unlink()  # remove the un-writable original so dataset_producer.py doesn't see both

    return {"name": path.name, "orig_secs": orig_secs, "kept_secs": kept_secs,
           "removed_frac": removed_frac, "lead_secs": lead_secs, "trail_secs": trail_secs,
           "status": status, "written": True, "written_as": write_path.name}


def run(references_dir: Path, backup_dir: Path, trim_top_db: float, force: bool):
    files = sorted(p for p in references_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"No audio files found in {references_dir}")

    logger.info(f"{'file':22s} {'class':10s} {'orig':>7s} {'kept':>7s} {'lead cut':>9s} "
               f"{'trail cut':>10s} {'removed':>8s}")
    logger.info("-" * 90)
    skipped, warnings, written = 0, [], 0
    for p in files:
        r = trim_one(p, backup_dir, trim_top_db, force)
        if r is None:
            skipped += 1
            continue
        label = _class_label_for(p)
        if not r.get("written"):
            logger.info(f"{r['name']:22s} {label:10s} {r['orig_secs']:6.2f}s {'--':>7s} "
                       f"{'--':>9s} {'--':>10s} {r['removed_frac']:7.1%}  <-- {r['status']}")
            warnings.append((r["name"], r["status"]))
            continue
        rename = f" -> {r['written_as']}" if r["written_as"] != r["name"] else ""
        flag = f"  <-- {r['status']}" if r["status"] else ""
        logger.info(f"{r['name']:22s} {label:10s} {r['orig_secs']:6.2f}s {r['kept_secs']:6.2f}s "
                   f"{r['lead_secs']:8.2f}s {r['trail_secs']:9.2f}s {r['removed_frac']:7.1%}"
                   f"{rename}{flag}")
        written += 1
        if r["status"]:
            warnings.append((r["name"], r["status"]))

    logger.info("-" * 90)
    logger.info(f"{written} file(s) trimmed and saved, {skipped} already done (skipped -- use --force "
               f"to redo), backups kept in {backup_dir}")
    if warnings:
        logger.info(f"{len(warnings)} file(s) worth a closer look: {warnings}")
    else:
        logger.info("No warnings.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--references-dir", type=Path, default=cfg.REFERENCES_DIR)
    parser.add_argument("--backup-dir", type=Path, default=None,
                        help="where untouched originals are kept (default: a sibling folder next to "
                             "--references-dir, e.g. references_untrimmed_backup)")
    parser.add_argument("--trim-top-db", type=float, default=TRIM_TOP_DB,
                        help=f"silence threshold, dB below the file's peak (default {TRIM_TOP_DB}). "
                             f"Lower = trims more aggressively.")
    parser.add_argument("--force", action="store_true",
                        help="reprocess files that already have a backup (re-trims from the backed-up "
                             "original, not from the possibly-already-trimmed current file)")
    args = parser.parse_args()

    backup_dir = args.backup_dir or _default_backup_dir(args.references_dir)
    run(args.references_dir, backup_dir, args.trim_top_db, args.force)


if __name__ == "__main__":
    main()
