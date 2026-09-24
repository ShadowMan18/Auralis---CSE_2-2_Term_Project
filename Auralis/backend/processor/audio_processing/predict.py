"""
predict.py

Run the trained phase-1 CNN on a single audio file and print class
probabilities. Reports "no confident detection" (rather than forcing a
guess) if the winning class's probability is below cfg.MIN_CONFIDENCE, or
if the background class wins -- a closed-set softmax classifier otherwise
always confidently picks SOMETHING, even for silence or an unrelated
sound.

Usage:
    python predict.py path/to/audio.wav
    python predict.py path/to/audio.wav --model-path species_cnn_phase1.pt
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import ml_config as cfg
from train_cnn import SmallAudioCNN, compute_logmel


def load_model(model_path: Path):
    labels_path = model_path.with_suffix(".labels.json")
    with open(labels_path) as f:
        class_to_idx = json.load(f)
    idx_to_class = {i: label for label, i in class_to_idx.items()}

    model = SmallAudioCNN(num_classes=len(class_to_idx))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, idx_to_class


def predict(path, model, idx_to_class, min_confidence=cfg.MIN_CONFIDENCE):
    logmel = compute_logmel(path)
    x = torch.from_numpy(logmel).unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, n_frames)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1).squeeze(0)

    ranked = sorted(
        ((idx_to_class[i], probs[i].item()) for i in range(len(probs))),
        key=lambda kv: -kv[1],
    )
    top_label, top_prob = ranked[0]

    if top_label == cfg.BACKGROUND_LABEL or top_prob < min_confidence:
        decision = None
    else:
        decision = top_label

    return decision, ranked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--model-path", type=Path, default=cfg.MODEL_PATH)
    parser.add_argument("--min-confidence", type=float, default=cfg.MIN_CONFIDENCE)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    model, idx_to_class = load_model(args.model_path)
    decision, ranked = predict(args.audio_path, model, idx_to_class, args.min_confidence)

    print(f"file: {args.audio_path.name}")
    for label, prob in ranked[:args.top_k]:
        marker = " <-- top" if label == ranked[0][0] else ""
        print(f"  {label:12s} {prob:.4f}{marker}")
    print(f"decision: {decision if decision else 'no confident detection'}")


if __name__ == "__main__":
    main()
