"""
dsp_diagnostics.py — Two diagnostics for the DSP fingerprint matcher.

1. print_hash_counts()
   Lists every reference's total hash count, sorted ascending. A
   reference with a tiny count (single digits, low tens) lets pure
   noise win a match: score = raw_count / total, so a handful of
   coincidental hash collisions divided by a tiny denominator produces
   a deceptively large score fraction. This is the likely explanation
   for a species like rooster winning matches it has nothing to do
   with -- look for anything flagged below.

2. inspect_self_match(filename)
   For a query whose self-match score comes out above 1.0 (or just
   looks suspicious), this shows exactly which hashes are voting for
   the winning offset, and -- critically -- whether any hash repeats
   MORE THAN ONCE within the query's own fingerprint. If a hash h
   occurs n times in the query and m times in the matched reference,
   match_and_score compares every query occurrence against every
   reference occurrence for that hash (an n*m cross product landing in
   the same offset bucket, if the repeats are evenly spaced -- which a
   periodic call like a repeated bark or neigh often is). That cross
   product is what lets raw_count outgrow the reference's own total
   hash count, since reference_hash_counts only ever counts each
   fingerprint pair once, not the pairwise product against a
   self-similar query.

Run from anywhere inside the backend/ tree (same sys.path fix as
test_dsp.py).
"""

import sys
from collections import Counter
from pathlib import Path


def _find_backend_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "processor" / "services").is_dir():
            return candidate
    raise RuntimeError(
        f"Could not find the backend root (a folder containing "
        f"processor/services/) by walking up from {start}."
    )


sys.path.insert(0, str(_find_backend_root(Path(__file__).resolve().parent)))

from processor.services import processor as detector
from processor.audio_processing.audio_processing import AudioPreprocessor

# Same folder test_dsp.py points at -- adjust if your test files live
# somewhere else.
SAMPLES_TEST_DIR = Path(
    r"C:\Users\ramis\Documents\2-2\220\Auralis---CSE_2-2_Term_Project\Auralis\backend\processor\audio_processing\samples\test"
)


class LocalFile:
    """Duck-types werkzeug's FileStorage well enough for
    AudioPreprocessor.process()'s seek/read calls on a plain local path."""

    def __init__(self, path: Path):
        self.filename = path.name
        self._path = path
        self._fh = None

    def __getattr__(self, name):
        if self._fh is None:
            self._fh = open(self._path, "rb")
        return getattr(self._fh, name)


def print_hash_counts(flag_below=20):
    """Sorted ascending so degenerate references are at the top."""
    matcher = detector._get_matcher()
    counts = sorted(matcher.reference_hash_counts.items(), key=lambda kv: kv[1])

    print(f"{'ref_id':45s} hash_count")
    for ref_id, count in counts:
        flag = "  <-- suspiciously small, noise can win here" if count < flag_below else ""
        print(f"{ref_id:45s} {count:6d}{flag}")


def inspect_self_match(filename, top_k=3):
    """filename: a file under SAMPLES_TEST_DIR, e.g. 'dog_original.wav'."""
    path = SAMPLES_TEST_DIR / filename
    matcher = detector._get_matcher()
    preprocessor = AudioPreprocessor()

    windows = preprocessor.process(LocalFile(path), detector.QUERY_SPEC)
    spectrogram = detector._compute_log_spectrogram(windows[0], detector.QUERY_SPEC.target_sample_rate)

    keypoints = matcher.extract_keypoints(spectrogram, **detector.EXTRACT_PARAMS)
    fingerprint = matcher.build_fingerprint(keypoints, **matcher.fingerprint_kwargs)

    query_hash_counts = Counter(h for h, _t in fingerprint)
    repeated = {h: c for h, c in query_hash_counts.items() if c > 1}

    print(f"\n--- {filename} ---")
    print(f"Query fingerprint: {len(fingerprint)} pairs, {len(query_hash_counts)} distinct hashes.")
    if repeated:
        worst = max(repeated.values())
        print(f"{len(repeated)} hashes repeat WITHIN the query itself (max {worst}x for one hash) "
              f"-- these are the ones that can cause raw_count to exceed a reference's own total.")
    else:
        print("No internal repeats in this query's fingerprint -- a >1.0 score here would need "
              "a different explanation (e.g. duplicate reference registration).")

    results = matcher.match_and_score(fingerprint)
    ranked = matcher.rank_matches(results, top_k=top_k)
    print("Top matches:")
    for ref_id, info in ranked:
        total = matcher.reference_hash_counts.get(ref_id)
        print(f"  {ref_id:35s} score={info['score']:.4f}  raw_count={info['raw_count']:4d}  "
              f"total={total:4d}  offset={info['offset']}")


def correlate_with_duration(references_dir=None):
    """For each reference file, print its duration alongside the total
    hash count summed across all its augmentation variants, plus
    hashes-per-second. If hashes/sec is roughly consistent across files,
    the spread in raw hash counts is just explained by clip length being
    different (not a bug -- just uneven recording lengths). If
    hashes/sec varies wildly too, something about specific files
    themselves (near-silence, heavy trimming, clipping, wrong sample
    rate) is the actual problem, independent of length."""
    import soundfile as sf

    references_dir = Path(references_dir) if references_dir else Path(detector.REFERENCES_DIR)
    matcher = detector._get_matcher()

    # Sum hash counts per underlying recording (across all its ::variant entries)
    per_recording = {}
    for ref_id, count in matcher.reference_hash_counts.items():
        recording_id = ref_id.split(matcher.fingerprint_kwargs.get("variant_separator", "::"))[0] \
            if hasattr(matcher, "fingerprint_kwargs") else ref_id.split("::")[0]
        # fall back to splitting on the separator used when building, typically "::"
        recording_id = ref_id.split("::")[0]
        per_recording[recording_id] = per_recording.get(recording_id, 0) + count

    print(f"{'file':20s} {'duration_s':>10s} {'total_hashes':>12s} {'hashes/sec':>10s}")
    rows = []
    for path in sorted(references_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in detector.ALLOWED_EXTENSIONS:
            continue
        recording_id = path.stem
        total_hashes = per_recording.get(recording_id, 0)
        try:
            info = sf.info(str(path))
            duration = info.frames / info.samplerate
        except Exception:
            duration = float("nan")
        rate = total_hashes / duration if duration else float("nan")
        rows.append((path.name, duration, total_hashes, rate))

    for name, duration, total_hashes, rate in sorted(rows, key=lambda r: r[3]):
        flag = "  <-- very low density for its length" if rate < 5 else ""
        print(f"{name:20s} {duration:10.2f} {total_hashes:12d} {rate:10.2f}{flag}")


if __name__ == "__main__":
    print("=== Reference hash counts (ascending -- look for anything flagged) ===")
    print_hash_counts()

    print("\n=== Hash density vs. duration (is the spread just clip length, or a data problem?) ===")
    correlate_with_duration()

    print("\n=== Self-match inspection (edit the filenames below to check others) ===")
    for fname in ["dog_original.wav", "horse_original.wav", "rooster_original.wav"]:
        try:
            inspect_self_match(fname)
        except FileNotFoundError:
            print(f"\n--- {fname} --- (file not found, skipping)")