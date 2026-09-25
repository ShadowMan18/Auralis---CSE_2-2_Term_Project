"""
test_dsp.py — Batch-run the pure-DSP (AudioMatcher fingerprint) detection
path over every sample file in samples/test and chart the confidence
scores, without re-running inference each time you want to look at the
chart.

This exercises process_sample() (the fingerprint-matcher path), NOT
process_cnn_sample() (which is what test.py exercises and what the live
route currently calls) -- use this one specifically to evaluate the DSP
branch on its own, independent of the CNN.

Usage
-----
    python test_dsp.py                  # run inference on every file, cache to test_dsp_results.json, then plot
    python test_dsp.py --replot         # skip inference, just re-plot from the existing cache
    python test_dsp.py --threshold 0.05 # override the threshold line drawn on the chart

Run this from the `backend/` directory (the one containing the
`processor` package) so `from processor.services import processor`
resolves the same way it does for routes.py.

NOTE: the first call builds the AudioMatcher reference library from
samples/references (with the full pitch/tempo augmentation grid), which
can take a while -- that's expected, and it's cached in-process for the
rest of the run (see _get_matcher() in processor.py).

Requires matplotlib: pip install matplotlib
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def _find_backend_root(start: Path) -> Path:
    """Python puts the SCRIPT's own folder on sys.path, not your current
    working directory -- so running this from audio_processing\\, or even
    from backend\\ with a plain `python test_dsp.py`, never makes
    `processor.services` importable on its own. Walk up from this file's
    location until we find the directory that actually contains the
    `processor` package and use that, regardless of where the script
    lives or where you ran it from.

    Checks for the directory (processor/services/) rather than an
    __init__.py file, since this project's `processor` package has no
    __init__.py -- it relies on Python's implicit namespace packages,
    same reason the relative imports elsewhere (e.g. `from .resample
    import Resample`) work without one."""
    for candidate in (start, *start.parents):
        if (candidate / "processor" / "services").is_dir():
            return candidate
    raise RuntimeError(
        "Could not find the backend root (a folder containing "
        f"processor/services/) by walking up from {start}. If this "
        "script isn't somewhere inside the backend/ tree, hardcode the "
        "backend path here instead."
    )


sys.path.insert(0, str(_find_backend_root(Path(__file__).resolve().parent)))

# Adjust this import if your run config resolves packages differently.
from processor.services import processor as detector

SAMPLES_DIR = Path(
    r"C:\Users\ramis\Documents\2-2\220\Auralis---CSE_2-2_Term_Project\Auralis\backend\processor\audio_processing\samples\test"
)
RESULTS_CACHE = Path(__file__).parent / "test_dsp_results.json"
ALLOWED_EXTENSIONS = detector.ALLOWED_EXTENSIONS


class LocalFile:
    """Minimal duck-type of werkzeug's FileStorage -- just enough for
    process_sample()'s `sample.filename` / stream-based reads, so a plain
    local path can be fed in instead of a real upload."""

    def __init__(self, path: Path):
        self.filename = path.name
        self._path = path
        self._fh = None

    # AudioPreprocessor.process() -> AudioDecoder.decode() calls
    # file_obj.seek(0) then reads from it directly (no .save() call, since
    # process_sample doesn't route through a temp file the way the CNN
    # path does) -- so this shim needs to behave like an actual open file.
    def __getattr__(self, name):
        if self._fh is None:
            self._fh = open(self._path, "rb")
        return getattr(self._fh, name)


def guess_true_label(filename: str) -> str:
    """Infer a single-species label from the test filename.

    Test clips use names such as ``sample_cat2`` while references use
    ``cat_002``. A mixed-species clip has no single top-1 ground-truth
    label, so return None for it and leave it out of correctness coloring.
    """
    stem = Path(filename).stem
    known_species = {
        "cat", "cow", "crow", "dog", "goat", "horse", "monkey", "rooster"
    }
    tokens = stem.lower().replace("-", "_").replace(" ", "_").split("_")
    labels = [token for token in tokens if token in known_species]
    return labels[0] if len(labels) == 1 else None


def run_inference():
    if not SAMPLES_DIR.exists():
        sys.exit(f"Samples folder not found: {SAMPLES_DIR}")

    files = sorted(
        p for p in SAMPLES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
    )
    if not files:
        sys.exit(f"No audio files with extensions {sorted(ALLOWED_EXTENSIONS)} found in {SAMPLES_DIR}")

    results = []
    for path in files:
        print(f"Processing {path.name} ...")
        result, status = detector.process_sample(LocalFile(path))
        if status != 200:
            print(f"  -> error ({status}): {result.get('error')}")
            results.append({
                "file": path.name,
                "true_label": guess_true_label(path.name),
                "error": result.get("error"),
                "predictions": [],
            })
            continue

        # process_sample returns parallel lists, already sorted by
        # descending score -- reshape into the same {species, confidence}
        # dict-list shape test.py uses, so plot_results below is identical.
        predictions = [
            {"species": species, "confidence": confidence}
            for species, confidence in zip(result.get("species", []), result.get("confidence", []))
        ]
        print(f"  -> {predictions}")
        results.append({
            "file": path.name,
            "true_label": guess_true_label(path.name),
            "error": None,
            "predictions": predictions,
        })

    RESULTS_CACHE.write_text(json.dumps(results, indent=2))
    print(f"\nCached {len(results)} results to {RESULTS_CACHE}")
    return results


def load_cached_results():
    if not RESULTS_CACHE.exists():
        sys.exit(f"No cached results at {RESULTS_CACHE} -- run without --replot first.")
    return json.loads(RESULTS_CACHE.read_text())


def plot_results(results, threshold=None):
    threshold = threshold if threshold is not None else detector.CONFIDENCE_THRESHOLD

    files = [r["file"] for r in results]
    top_conf = []
    top_species = []
    correct = []

    for r in results:
        preds = r["predictions"]
        if preds:
            top_conf.append(preds[0]["confidence"])
            top_species.append(preds[0]["species"])
            label = r.get("true_label")
            correct.append(None if label is None else preds[0]["species"].split("_")[0] == label)
        else:
            top_conf.append(0.0)
            top_species.append("(no detection)")
            correct.append(False)

    colors = ["#2ca02c" if c is True else "#d62728" if c is False else "#7f7f7f" for c in correct]

    fig, ax = plt.subplots(figsize=(max(8, len(files) * 0.5), 6))
    bars = ax.bar(range(len(files)), top_conf, color=colors)
    ax.axhline(threshold, color="gray", linestyle="--", linewidth=1,
               label=f"confidence threshold ({threshold})")

    ax.set_xticks(range(len(files)))
    ax.set_xticklabels(files, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("Top match score")
    ax.set_title("Auralis DSP fingerprint matcher — top-1 score per test file")
    top = max(top_conf) if top_conf else 1.0
    ax.set_ylim(0, max(1.0, top * 1.15))
    ax.legend()

    for bar, species in zip(bars, top_species):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(1.0, top),
                species, rotation=90, fontsize=7, ha="center", va="bottom")

    fig.tight_layout()
    out_path = Path(__file__).parent / "test_dsp_results_chart.png"
    fig.savefig(out_path, dpi=150)
    print(f"Chart saved to {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot", action="store_true",
                         help="Skip inference; re-plot from the cached results.")
    parser.add_argument("--threshold", type=float, default=None,
                         help="Override the confidence threshold line on the chart.")
    args = parser.parse_args()

    results = load_cached_results() if args.replot else run_inference()
    plot_results(results, threshold=args.threshold)


if __name__ == "__main__":
    main()
