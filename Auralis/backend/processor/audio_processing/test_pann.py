"""
test_pann.py — Batch-run the zero-shot PANN pathway over every sample file
in samples/test and chart the top-1 confidence per file, mirroring
test.py's shape exactly (same SAMPLES_DIR convention, same cache-then-plot
flow) but calling process_pann_sample instead of process_cnn_sample.

Since this pathway is zero-shot on generic AudioSet tags, guess_true_label
matching top-1 against the filename's species prefix will mostly show
"incorrect" (red) bars -- that's expected and not a bug: a "cat.wav" file
scoring high on "Cat" is a genuine hit, but reference filenames using our
own species taxonomy (e.g. "howler_001.wav") will rarely have an exact
AudioSet label match. Read the printed top predictions per file, not just
the color-coding, when judging this pathway's output.

Usage
-----
    python test_pann.py                  # run inference, cache to test_pann_results.json, then plot
    python test_pann.py --replot         # skip inference, just re-plot from the existing cache
    python test_pann.py --top-k 10       # keep more than the top prediction per file
    python test_pann.py --threshold 0.2  # override the threshold line drawn on the chart

Run this from the `backend/` directory, same as test.py -- see that
file's docstring for the import-path reasoning this script reuses as-is.

Requires matplotlib: pip install matplotlib
Requires panns_inference: pip install panns_inference
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def _find_backend_root(start: Path) -> Path:
    """Identical to test.py's helper of the same name -- see that file's
    docstring for why this walk-up is needed instead of assuming cwd."""
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

from processor.services import processor as detector

SAMPLES_DIR = Path(__file__).resolve().parent / "samples" / "test"
MANIFEST_PATH = SAMPLES_DIR / "generated_manifest.json"
RESULTS_CACHE = Path(__file__).parent / "test_pann_results.json"
ALLOWED_EXTENSIONS = detector.ALLOWED_EXTENSIONS


class LocalFile:
    """Identical duck-type to test.py's LocalFile -- see that file."""

    def __init__(self, path: Path):
        self.filename = path.name
        self._path = path

    def save(self, dst):
        dst.write(self._path.read_bytes())


def guess_true_label(filename: str):
    """Best-effort single-animal label for files absent from the manifest."""
    stem = Path(filename).stem
    known_species = {
        "cat", "cow", "crow", "dog", "frog", "goat", "gunshot", "horse",
        "monkey", "owl", "roaring", "rooster", "wolf",
    }
    tokens = stem.lower().replace("-", "_").replace(" ", "_").split("_")
    labels = [
        species for species in known_species
        if any(token == species or (token.startswith(species) and token[len(species):].isdigit())
               for token in tokens)
    ]
    return labels[0] if len(labels) == 1 else None


def run_inference(top_k=10):
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
        result, status = detector.process_pann_sample(LocalFile(path), top_k=top_k)
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


def _sample_metadata(results):
    """Read desired species/sample kinds from the generator manifest."""
    manifest_samples = {}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest_samples = manifest.get("samples", {})

    metadata = {}
    for result in results:
        filename = result["file"]
        entry = manifest_samples.get(filename)
        if entry:
            metadata[filename] = {
                "kind": entry.get("kind"),
                "species": entry.get("species", []),
            }
            continue

        label = result.get("true_label") or guess_true_label(filename)
        if label:
            metadata[filename] = {"kind": "single_species_variant", "species": [label]}
        else:
            metadata[filename] = {"kind": "unclassified", "species": []}
    return metadata


def plot_results(results, threshold=None):
    from processor.audio_processing import pann_config as pann_cfg
    threshold = threshold if threshold is not None else pann_cfg.MIN_CONFIDENCE
    metadata = _sample_metadata(results)
    singles, non_overlapping, overlapping = [], [], []
    for result in results:
        kind = metadata[result["file"]]["kind"]
        if kind == "single_species_variant":
            singles.append(result)
        elif kind == "non_overlapping_concatenation":
            non_overlapping.append(result)
        elif kind == "overlapping_mix":
            overlapping.append(result)

    panel_width = max(len(singles), 2 * len(non_overlapping), 2 * len(overlapping), 1)
    fig = plt.figure(figsize=(max(18, panel_width * 0.48), 13), facecolor="#f7f8fc")
    grid = fig.add_gridspec(2, 2, height_ratios=(1.35, 1), hspace=0.75, wspace=0.16)
    single_ax = fig.add_subplot(grid[0, :])
    nonoverlap_ax = fig.add_subplot(grid[1, 0])
    overlap_ax = fig.add_subplot(grid[1, 1])

    def style_axis(ax, title):
        ax.set_facecolor("white")
        ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=12)
        ax.set_ylabel("Confidence")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    # Top panel: color each single-animal clip by top-1 correctness.
    style_axis(single_ax, "Single-animal samples")
    single_scores, single_colors, single_labels = [], [], []
    for result in singles:
        desired = {str(item).casefold() for item in metadata[result["file"]]["species"]}
        predictions = result.get("predictions", [])
        top = predictions[0] if predictions else None
        score = float(top["confidence"]) if top else 0.0
        correct = bool(top) and str(top["species"]).casefold() in desired
        single_scores.append(score)
        single_colors.append("#31a36a" if correct else "#d76565" if top else "#aeb7c4")
        single_labels.append(Path(result["file"]).stem)

    x_single = list(range(len(singles)))
    bars = single_ax.bar(x_single, single_scores, color=single_colors, width=0.76,
                         edgecolor="white", linewidth=0.7)
    single_ylim = max(1.12, max(single_scores, default=0.0) * 1.18)
    single_ax.set_ylim(0, single_ylim)
    if 0 < threshold < single_ylim:
        single_ax.axhline(threshold, color="#777f8c", linestyle="--", linewidth=1)
    single_ax.set_xticks(x_single)
    single_ax.set_xticklabels(single_labels, rotation=90, ha="center", fontsize=7)
    for bar, score in zip(bars, single_scores):
        if score > 0:
            single_ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.018,
                           f"{score:.2f}", ha="center", va="bottom", fontsize=6.5)
    single_ax.legend(handles=[
        Patch(facecolor="#31a36a", label="correct"),
        Patch(facecolor="#d76565", label="incorrect"),
        Patch(facecolor="#aeb7c4", label="no detection"),
    ], loc="upper right", frameon=False, ncol=3, fontsize=8)

    def plot_mixtures(ax, rows, title):
        style_axis(ax, title)
        scores, colors, labels, annotations = [], [], [], []
        for result in rows:
            desired = metadata[result["file"]]["species"]
            desired_set = {str(item).casefold() for item in desired}
            matched = [
                prediction for prediction in result.get("predictions", [])
                if str(prediction["species"]).casefold() in desired_set
            ]
            received = list(dict.fromkeys(prediction["species"] for prediction in matched))
            score = (
                sum(float(prediction["confidence"]) for prediction in matched) / len(matched)
                if matched else 0.0
            )
            scores.append(score)
            colors.append("#4b83c3" if matched else "#b9c1ce")
            labels.append(
                f"{Path(result['file']).stem}\nWanted: {', '.join(desired) or 'unknown'}"
            )
            found = ", ".join(received) if received else "No desired species detected"
            annotations.append(f"{found}\nmean {score:.2f}  ·  {len(received)}/{len(desired)} found")

        x_values = list(range(len(rows)))
        bars = ax.bar(x_values, scores, color=colors, width=0.62,
                      edgecolor="white", linewidth=0.8)
        axis_ylim = max(1.12, max(scores, default=0.0) * 1.18)
        ax.set_ylim(0, axis_ylim)
        if 0 < threshold < axis_ylim:
            ax.axhline(threshold, color="#777f8c", linestyle="--", linewidth=1)
        ax.set_xticks(x_values)
        ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
        for bar, annotation in zip(bars, annotations):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025,
                    annotation, ha="center", va="bottom", fontsize=8)

    plot_mixtures(nonoverlap_ax, non_overlapping, "Non-overlapping samples")
    plot_mixtures(overlap_ax, overlapping, "Overlapping samples")

    fig.suptitle("Auralis PANN results", fontsize=17, fontweight="bold", y=0.99)
    fig.subplots_adjust(top=0.91, bottom=0.16, left=0.055, right=0.99)
    out_path = Path(__file__).parent / "test_pann_results_chart.png"
    fig.savefig(out_path, dpi=150)
    print(f"Chart saved to {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot", action="store_true",
                         help="Skip inference; re-plot from the cached results.")
    parser.add_argument("--top-k", type=int, default=10,
                         help="How many predictions to keep per file (default: 10).")
    parser.add_argument("--threshold", type=float, default=None,
                         help="Override the confidence threshold line on the chart.")
    args = parser.parse_args()

    results = load_cached_results() if args.replot else run_inference(top_k=args.top_k)
    plot_results(results, threshold=args.threshold)


if __name__ == "__main__":
    main()
