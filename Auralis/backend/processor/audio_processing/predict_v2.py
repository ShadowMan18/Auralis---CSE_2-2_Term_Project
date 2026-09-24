"""
predict_v2.py

End-to-end phase-2 inference: upload a forest recording (+ optionally lat,
lon, month) -> a ranked list of candidate species/rollup-groups with
confidence.

Pipeline:
    1. Slice the whole recording into WINDOW_SECONDS windows strided by
       WINDOW_STRIDE_SECONDS (same windowing as training).
    2. Run the CNN on every window; pool per-class probabilities across
       windows into one per-recording score (WINDOW_AGGREGATION: "max" by
       default -- a species calling once in a long recording shouldn't be
       diluted by all the windows around it that don't contain it).
    3. If lat/lon/month are given and a geo prior is configured, apply it
       (geo_prior.py) -- this both rules out geographically-impossible
       candidates and disambiguates acoustically-similar-but-geographically-
       disjoint confusions (e.g. lion vs. jaguar) automatically.
    4. Report "no confident detection" rather than forcing an answer, if
       the winning class is background or below MIN_CONFIDENCE -- matches
       phase-1's predict.py behavior, now over the real class space.

Usage:
    python predict_v2.py recording.wav
    python predict_v2.py recording.wav --lat 3.1 --lon 101.6 --month 7 \\
        --geo-prior-json my_region_season_table.json
    python predict_v2.py recording.wav --top-k 10
"""

import argparse
import json
from pathlib import Path

import numpy as np
import librosa
import torch
import torch.nn.functional as F

try:  # package import (Flask) and direct-script import both work
    from . import ml_config_v2 as cfg
    from . import geo_prior as geo
    from .train_cnn_v2 import build_model, compute_logmel
except ImportError:
    import ml_config_v2 as cfg
    import geo_prior as geo
    from train_cnn_v2 import build_model, compute_logmel


def load_full_audio(path: str, sr: int = cfg.SAMPLE_RATE):
    y, _sr = librosa.load(str(path), sr=sr, mono=True)
    return y


def window_starts_seconds(duration_s: float, window_s: float, stride_s: float):
    if duration_s <= window_s:
        return [0.0]
    starts, t = [], 0.0
    while t + window_s <= duration_s:
        starts.append(t)
        t += stride_s
    return starts or [0.0]


def slice_windows(y: np.ndarray, sr: int, window_s: float, stride_s: float):
    duration = len(y) / sr
    target_len = int(round(window_s * sr))
    windows = []
    for start_s in window_starts_seconds(duration, window_s, stride_s):
        start = int(round(start_s * sr))
        chunk = y[start:start + target_len]
        if len(chunk) < target_len:
            chunk = np.pad(chunk, (0, target_len - len(chunk)))
        windows.append(chunk)
    return windows


def load_model(model_path: Path):
    labels_path = model_path.with_suffix(".labels.json")
    with open(labels_path) as f:
        class_to_idx = json.load(f)
    idx_to_class = {i: label for label, i in class_to_idx.items()}

    checkpoint = torch.load(model_path, map_location="cpu")
    arch = checkpoint.get("arch", cfg.MODEL_ARCH)
    loss_kind = checkpoint.get("loss_kind", cfg.LOSS)
    model = build_model(arch, num_classes=len(class_to_idx), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, idx_to_class, loss_kind


def load_label_display_map(path: Path):
    """slug -> {"final_label": ..., "resolution": ...}, from
    inatsounds_route.py's class_label_map.json. Falls back to using the
    slug itself if the map file isn't present."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def cnn_predict_recording(y: np.ndarray, model, idx_to_class: dict, loss_kind: str) -> dict:
    windows = slice_windows(y, cfg.SAMPLE_RATE, cfg.WINDOW_SECONDS, cfg.WINDOW_STRIDE_SECONDS)
    logmels = [compute_logmel(w) for w in windows]
    x = torch.from_numpy(np.stack(logmels)).unsqueeze(1).repeat(1, 3, 1, 1)  # (N,3,mel,frames)

    with torch.no_grad():
        logits = model(x)
        if loss_kind == "assume_negative_bce":
            probs = torch.sigmoid(logits)
        else:
            probs = F.softmax(logits, dim=1)
    probs = probs.numpy()  # (num_windows, num_classes)

    if cfg.WINDOW_AGGREGATION == "max":
        pooled = probs.max(axis=0)
    else:
        pooled = probs.mean(axis=0)

    # renormalize so downstream geo-prior combination sees a proper
    # distribution regardless of aggregation / loss choice
    total = pooled.sum()
    if total > 0:
        pooled = pooled / total
    return {idx_to_class[i]: float(pooled[i]) for i in range(len(pooled))}


def predict(audio_path: Path, model_path: Path = cfg.MODEL_PATH,
            lat: float | None = None, lon: float | None = None, month: int | None = None,
            geo_prior_obj=None, min_confidence: float = cfg.MIN_CONFIDENCE):
    model, idx_to_class, loss_kind = load_model(model_path)
    label_map = load_label_display_map(cfg.CLASS_LABEL_MAP_PATH)

    y = load_full_audio(audio_path)
    cnn_probs = cnn_predict_recording(y, model, idx_to_class, loss_kind)

    prior = None
    if lat is not None and lon is not None and month is not None and geo_prior_obj is not None:
        prior = geo_prior_obj.predict(lat, lon, month, list(cnn_probs.keys()))
    final_probs = geo.apply_geo_prior(cnn_probs, prior)

    ranked = sorted(final_probs.items(), key=lambda kv: -kv[1])
    top_label, top_prob = ranked[0]

    decision = None
    if top_label != cfg.BACKGROUND_LABEL and top_prob >= min_confidence:
        decision = top_label

    def display(slug):
        info = label_map.get(slug)
        if info is None:
            return slug
        tag = "" if info["resolution"] == "species" else f" [{info['resolution']}-level group]"
        return f"{info['final_label']}{tag}"

    ranked_display = [(display(label), prob) for label, prob in ranked]
    return decision, ranked_display


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--model-path", type=Path, default=cfg.MODEL_PATH)
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument("--month", type=int, default=None, help="1-12")
    parser.add_argument("--geo-prior-json", type=Path, default=None,
                         help="a JsonGeoPrior table (see geo_prior.py docstring)")
    parser.add_argument("--min-confidence", type=float, default=cfg.MIN_CONFIDENCE)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    geo_prior_obj = None
    if args.geo_prior_json is not None:
        geo_prior_obj = geo.JsonGeoPrior(args.geo_prior_json)

    decision, ranked = predict(
        args.audio_path, args.model_path,
        lat=args.lat, lon=args.lon, month=args.month, geo_prior_obj=geo_prior_obj,
        min_confidence=args.min_confidence,
    )

    print(f"file: {args.audio_path.name}")
    for label, prob in ranked[:args.top_k]:
        marker = " <-- top" if label == ranked[0][0] else ""
        print(f"  {label:40s} {prob:.4f}{marker}")
    print(f"decision: {decision if decision else 'no confident detection'}")


if __name__ == "__main__":
    main()
