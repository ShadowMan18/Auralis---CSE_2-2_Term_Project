"""
test.py — Batch-run the CNN detection pipeline over every sample file in
samples/test and chart the confidence scores, without re-running
inference each time you want to look at the chart.

Usage
-----
    python test_cnn.py                  # cache to test_cnn_results.json, then plot
    python test_cnn.py --replot         # skip inference, just re-plot the CNN cache
    python test_cnn.py --top-k 5        # keep more than the top prediction per file
    python test_cnn.py --threshold 0.3  # override the threshold line on the chart

Run this from the `backend/` directory (the one containing the
`processor` package) so `from processor.services import processor`
resolves the same way it does for routes.py. If your project uses a
different PYTHONPATH root (see the import-style note in processor.py's
own docstring about bandpass.py vs routes.py), adjust the import below
to match.

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
    from backend\\ with a plain `python test.py`, never makes
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
    r"C:\Users\Shadman Sami Shanon\OneDrive\Desktop\Auralis---CSE_2-2_Term_Project\Auralis\backend\processor\audio_processing\samples\test"
)
RESULTS_CACHE = Path(__file__).parent / "test_cnn_results.json"
ALLOWED_EXTENSIONS = detector.ALLOWED_EXTENSIONS


class LocalFile:
    """Minimal duck-type of werkzeug's FileStorage -- just enough for
    process_cnn_sample()'s `sample.filename` / `sample.save(temp)` calls,
    so a plain local path can be fed in instead of a real upload."""

    def __init__(self, path: Path):
        self.filename = path.name
        self._path = path

    def save(self, dst):
        dst.write(self._path.read_bytes())


def guess_true_label(filename: str) -> str:
    """Best-effort expected species from the filename, assuming the same
    '<species_id>...' convention used for reference files (see
    _build_matcher_from_references: species_id = path.stem). Only used
    to color-code the chart (correct/incorrect top-1), not for scoring --
    edit the separator list if your test files are named differently."""
    stem = Path(filename).stem
    for sep in ("_", "-", " "):
        if sep in stem:
            return stem.split(sep)[0]
    return stem


def run_inference(top_k=8):
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
        result, status = detector.process_cnn_sample(LocalFile(path), top_k=top_k)
        if status != 200:
            print(f"  -> error ({status}): {result.get('error')}")
            results.append({
                "file": path.name,
                "true_label": guess_true_label(path.name),
                "error": result.get("error"),
                "predictions": [],
            })
            continue

        predictions = result.get("predictions", [])
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
    threshold = threshold if threshold is not None else detector.CNN_MIN_CONFIDENCE

    files = [r["file"] for r in results]
    top_conf = []
    top_species = []
    correct = []

    for r in results:
        preds = r["predictions"]
        if preds:
            top_conf.append(preds[0]["confidence"])
            top_species.append(preds[0]["species"])
            correct.append(preds[0]["species"] == r["true_label"])
        else:
            top_conf.append(0.0)
            top_species.append("(no detection)")
            correct.append(False)

    colors = ["#2ca02c" if c else "#d62728" for c in correct]

    fig, ax = plt.subplots(figsize=(max(8, len(files) * 0.5), 6))
    bars = ax.bar(range(len(files)), top_conf, color=colors)
    ax.axhline(threshold, color="gray", linestyle="--", linewidth=1,
               label=f"confidence threshold ({threshold})")

    ax.set_xticks(range(len(files)))
    ax.set_xticklabels(files, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("Top prediction confidence")
    ax.set_title("Auralis CNN — top-1 confidence per test file")
    ax.set_ylim(0, 1.0)
    ax.legend()

    for bar, species in zip(bars, top_species):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                species, rotation=90, fontsize=7, ha="center", va="bottom")

    fig.tight_layout()
    out_path = Path(__file__).parent / "test_cnn_results_chart.png"
    fig.savefig(out_path, dpi=150)
    print(f"Chart saved to {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot", action="store_true",
                         help="Skip inference; re-plot from the cached results.")
    parser.add_argument("--top-k", type=int, default=8,
                         help="How many predictions to keep per file (default: 8).")
    parser.add_argument("--threshold", type=float, default=None,
                         help="Override the confidence threshold line on the chart.")
    args = parser.parse_args()

    results = load_cached_results() if args.replot else run_inference(top_k=args.top_k)
    plot_results(results, threshold=args.threshold)


if __name__ == "__main__":
    main()
