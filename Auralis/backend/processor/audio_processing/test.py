"""Run CNN inference for every audio file in ``samples/test``.

The model is loaded once, then each supported audio file in the test folder
(including files in its subfolders) is evaluated with the same CNN inference
code as ``predict.py``.

Usage:
    python test.py
    python test.py --test-dir path/to/test --top-k 3
    python -m processor.audio_processing.test
"""

import argparse
from pathlib import Path

try:  # Run as part of the processor.audio_processing package.
    from . import ml_config as cfg
    from .predict import load_model, predict
except ImportError:  # Run directly from this directory.
    import ml_config as cfg
    from predict import load_model, predict


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
DEFAULT_TEST_DIR = Path(__file__).resolve().parent / "samples" / "test"


def find_audio_files(test_dir: Path):
    """Return supported audio files in deterministic, recursive order."""
    return sorted(
        path for path in test_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def format_decision(decision):
    return decision if decision is not None else "no confident detection"


def run_batch(test_dir: Path, model_path: Path, min_confidence: float, top_k: int):
    """Print a CNN prediction report for every supported file in *test_dir*.

    Returns 0 when every file is processed successfully, 1 when no input files
    are found, and 2 if one or more files cannot be processed.
    """
    if not test_dir.is_dir():
        print(f"Test folder not found: {test_dir}")
        return 1

    files = find_audio_files(test_dir)
    if not files:
        extensions = ", ".join(sorted(AUDIO_EXTENSIONS))
        print(f"No audio files found in {test_dir} (supported: {extensions}).")
        return 1

    try:
        model, idx_to_class = load_model(model_path)
    except FileNotFoundError as exc:
        print(f"CNN model or label map not found: {exc}")
        return 1

    print(f"CNN model: {model_path}")
    print(f"Test folder: {test_dir}")
    print(f"Samples: {len(files)}\n")

    failures = 0
    for index, audio_path in enumerate(files, start=1):
        relative_path = audio_path.relative_to(test_dir)
        print(f"[{index}/{len(files)}] {relative_path}")
        try:
            decision, ranked = predict(
                audio_path, model, idx_to_class, min_confidence=min_confidence
            )
        except Exception as exc:
            failures += 1
            print(f"  error: {exc}\n")
            continue

        for label, probability in ranked[:top_k]:
            marker = " <-- top" if label == ranked[0][0] else ""
            print(f"  {label:12s} {probability:.4f}{marker}")
        print(f"  decision: {format_decision(decision)}\n")

    succeeded = len(files) - failures
    print(f"Completed: {succeeded}/{len(files)} samples processed successfully.")
    return 0 if failures == 0 else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=DEFAULT_TEST_DIR,
        help="Folder containing audio samples (default: samples/test).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=cfg.MODEL_PATH,
        help="CNN checkpoint to use.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=cfg.MIN_CONFIDENCE,
        help="Minimum top-class probability needed for a detection.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of class probabilities to show per sample.",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if not 0.0 <= args.min_confidence <= 1.0:
        parser.error("--min-confidence must be between 0 and 1")

    raise SystemExit(
        run_batch(args.test_dir, args.model_path, args.min_confidence, args.top_k)
    )


if __name__ == "__main__":
    main()
