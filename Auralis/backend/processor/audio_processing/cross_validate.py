"""
cross_validate.py

Runs leave-one-recording-out cross-validation over the phase-1 CNN, then
(optionally) trains the final production model on all the data.

WHY THIS EXISTS
----------------
With ~5 recordings per class, a single train/val split holds out just ONE
recording per class. Whichever recording that happens to be can swing the
reported accuracy a lot -- an easy recording held out looks great, a hard one
looks terrible, and you can't tell which happened from one run. This script
removes that luck: it runs k folds (k = number of recordings per class, or
fewer if you cap it with --max-folds), rotating which recording is held out
each time via dataset_producer.py's --fold, trains a fresh model per fold,
and reports the MEAN and STANDARD DEVIATION of accuracy per class across all
folds. A class with a high mean and low std is reliably learned; high mean
but high std means it did well on some recordings and poorly on others --
worth investigating (mislabeled file? one recording much noisier?) before
trusting the number.

This is a diagnostic step. It does NOT produce the model you deploy -- each
fold deliberately withholds one recording, so no single fold sees all your
data. After reviewing the CV report, train the real deployment model the
normal way (dataset_producer.py with no --fold, then train_cnn.py), which
uses every recording. --train-final does this automatically at the end.

Usage:
    python cross_validate.py                                  # CV, then train final model
    python cross_validate.py --no-train-final                 # CV report only
    python cross_validate.py --epochs 20 --max-folds 3         # quicker, fewer folds
"""

import argparse
import json
import logging
import shutil
import statistics
import sys
from pathlib import Path

import numpy as np

import ml_config as cfg
import dataset_producer as dp
import train_cnn as tc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
ALLOWED_EXTENSIONS = dp.ALLOWED_EXTENSIONS


def _recording_counts(references_dir: Path):
    by_class = {}
    for p in sorted(references_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS:
            by_class.setdefault(dp._class_label_for(p), []).append(p)
    return {label: len(paths) for label, paths in by_class.items()}


def run_cv(references_dir: Path, work_dir: Path, k: int, variants_per_class: int, epochs: int,
          batch_size: int, lr: float, seed: int, keep_datasets: bool = False):
    counts = _recording_counts(references_dir)
    logger.info("Recordings per class: %s", counts)
    eligible = {c: n for c, n in counts.items() if n >= 2}
    if len(eligible) < len(counts):
        skipped = sorted(set(counts) - set(eligible))
        logger.warning("Class(es) with only 1 recording can't be cross-validated (no recording left to "
                       "hold out): %s. They'll still train fine, but their accuracy stays 'optimistic' "
                       "until you add a 2nd recording -- see dataset_producer.py's per-class warning.",
                       skipped)
    if not eligible:
        raise RuntimeError("No class has >= 2 recordings; cross-validation needs at least 2 per class.")

    k = min(k, min(eligible.values()))
    logger.info("Running %d fold(s) (limited by the class with the fewest recordings: %d)", k,
               min(eligible.values()))

    per_class_accs = {c: [] for c in counts}  # class -> list of per-fold accuracy (0..1)
    overall_accs = []
    fold_reports = []

    for fold in range(k):
        fold_dir = work_dir / f"fold_{fold}"
        model_path = work_dir / f"cv_model_fold{fold}.pt"
        logger.info("=" * 60)
        logger.info("FOLD %d/%d", fold + 1, k)
        logger.info("=" * 60)
        dp.build_dataset(references_dir, fold_dir, variants_per_class, cfg.VAL_FRACTION, seed, "auto",
                         fold=fold)
        # val_mode tells us, PER CLASS, whether this fold actually held out a real
        # recording ("source") or reused the same recording for train+val ("variant",
        # only possible with 1 recording). Only "source" folds are genuine
        # cross-validation; a "variant" class must not be averaged in as if it were,
        # or its optimistic number would masquerade as a validated one.
        fold_val_mode = {c: v["val_mode"] for c, v in
                         json.load(open(fold_dir / "dataset_info.json"))["classes"].items()}
        best_acc, per_class = tc.train(fold_dir, model_path, epochs, batch_size, lr)
        overall_accs.append(best_acc)
        fold_reports.append({"fold": fold, "overall_val_acc": best_acc, "per_class": per_class,
                             "val_mode": fold_val_mode})
        for c, stat in per_class.items():
            if c == cfg.BACKGROUND_LABEL:
                continue  # not a species; not in `counts` (has no reference recordings to fold over)
            if stat["total"] > 0 and fold_val_mode.get(c) == "source":
                per_class_accs[c].append(stat["correct"] / stat["total"])
        if not keep_datasets:
            shutil.rmtree(fold_dir, ignore_errors=True)
        model_path.unlink(missing_ok=True)
        model_path.with_suffix(".labels.json").unlink(missing_ok=True)

    logger.info("=" * 60)
    logger.info("CROSS-VALIDATION SUMMARY (%d folds)", k)
    logger.info("=" * 60)
    logger.info("Overall val accuracy: mean=%.1f%%  std=%.1f%%  (per fold: %s)",
               100 * statistics.mean(overall_accs),
               100 * (statistics.stdev(overall_accs) if len(overall_accs) > 1 else 0.0),
               [f"{a:.1%}" for a in overall_accs])
    summary = {"k": k, "overall": {"mean": statistics.mean(overall_accs),
                                   "std": statistics.stdev(overall_accs) if len(overall_accs) > 1 else 0.0,
                                   "per_fold": overall_accs},
              "per_class": {}}
    for c in sorted(counts):
        accs = per_class_accs[c]
        if not accs:
            reason = "only 1 recording" if counts[c] < 2 else "no 'source' fold ran for it"
            logger.info("  %-12s -- not cross-validated (%s); its val accuracy in fold logs above is "
                       "OPTIMISTIC (same recording used for train and val)", c, reason)
            summary["per_class"][c] = {"mean": None, "std": None, "n_folds": 0}
            continue
        mean = statistics.mean(accs)
        std = statistics.stdev(accs) if len(accs) > 1 else 0.0
        flag = "  <-- high variance, check this class" if std > 0.20 else ""
        logger.info("  %-12s mean=%5.1f%%  std=%5.1f%%  (n=%d recordings tested)%s",
                   c, 100 * mean, 100 * std, len(accs), flag)
        summary["per_class"][c] = {"mean": mean, "std": std, "n_folds": len(accs)}

    with open(work_dir / "cv_report.json", "w") as f:
        json.dump({"summary": summary, "folds": fold_reports}, f, indent=2)
    logger.info("Full report: %s", work_dir / "cv_report.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--references-dir", type=Path, default=cfg.REFERENCES_DIR)
    parser.add_argument("--work-dir", type=Path, default=cfg.HERE / "cv_work",
                        help="scratch space for per-fold datasets/models (deleted per fold unless "
                             "--keep-datasets); cv_report.json is written here")
    parser.add_argument("--max-folds", type=int, default=10,
                        help="cap on folds; actual k is also capped by the class with fewest recordings")
    parser.add_argument("--variants-per-class", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS,
                        help="use fewer than a full training run's epochs to keep CV fast, if you like")
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=cfg.LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-datasets", action="store_true", help="don't delete per-fold dataset wavs")
    parser.add_argument("--no-train-final", dest="train_final", action="store_false",
                        help="skip training the final all-data deployment model after CV")
    parser.add_argument("--final-model-path", type=Path, default=cfg.MODEL_PATH)
    parser.add_argument("--final-dataset-dir", type=Path, default=cfg.DATASET_DIR)
    parser.add_argument("--final-epochs", type=int, default=None,
                        help="epochs for the final model (default: same as --epochs)")
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    summary = run_cv(args.references_dir, args.work_dir, args.max_folds, args.variants_per_class,
                     args.epochs, args.batch_size, args.lr, args.seed, args.keep_datasets)

    low = [c for c, s in summary["per_class"].items() if s["mean"] is not None and s["mean"] < 0.5]
    if low:
        logger.warning("Class(es) below 50%% mean CV accuracy: %s. More/cleaner recordings for these "
                       "would help more than any training-time change.", low)

    if args.train_final:
        logger.info("=" * 60)
        logger.info("Training FINAL model on ALL data (no recording held out)")
        logger.info("=" * 60)
        dp.build_dataset(args.references_dir, args.final_dataset_dir, args.variants_per_class,
                         cfg.VAL_FRACTION, args.seed, "auto")  # fold=None: normal random split
        final_epochs = args.final_epochs if args.final_epochs is not None else args.epochs
        best_acc, per_class = tc.train(args.final_dataset_dir, args.final_model_path, final_epochs,
                                       args.batch_size, args.lr)
        logger.info("Final model saved to %s (val_acc=%.3f on its own held-out split -- the CV report "
                    "above is the more trustworthy accuracy estimate for classes with >= 2 recordings)",
                    args.final_model_path, best_acc)
    else:
        logger.info("Skipped final training (--no-train-final). Run dataset_producer.py + train_cnn.py "
                    "normally (no --fold) when ready to produce the deployment model.")


if __name__ == "__main__":
    main()