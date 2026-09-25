"""
predict_pann.py

Zero-shot audio tagging with a pretrained PANN (CNN14, AudioSet, 527
classes) -- NO fine-tuning, NO species labels of our own. This is a
second, exploratory pipeline next to predict.py's phase-1 CNN: it will
never report "macaw" or "howler monkey", only whatever AudioSet's own
generic classes happen to cover (e.g. "Bird", "Bird vocalization, bird
call, bird song", "Squawk", "Parrot", "Insect", "Rain", "Wind noise
(microphone)" ...). See pann_config.py's module docstring for why, and
for the accepted "library black-box front-end" exception this pathway
uses.

Resampling from our canonical 22050 Hz to PANN's required 32000 Hz uses
our own hand-rolled Resample (resample.py) -- only the model's INTERNAL
mel-spectrogram computation is the accepted black-box, not this step.

Requires: pip install panns_inference torch librosa

Usage:
    python predict_pann.py path/to/audio.wav
    python predict_pann.py path/to/audio.wav --top-k 15 --min-confidence 0.1
"""

import argparse
from pathlib import Path

import librosa
import numpy as np

try:  # imported as part of the processor.audio_processing package (Flask service)
    from . import pann_config as cfg
    from .resample import Resample
except ImportError:  # run directly as a script from this folder
    import pann_config as cfg
    from resample import Resample

_tagger = None


def _get_tagger():
    """Lazily import + construct panns_inference.AudioTagging -- imported
    lazily (same reasoning as predict.py/processor.py lazily importing
    torch) so the rest of the project keeps working on a machine without
    panns_inference installed."""
    global _tagger
    if _tagger is None:
        checkpoint_path = cfg.ensure_assets()
        from panns_inference import AudioTagging
        _tagger = AudioTagging(checkpoint_path=str(checkpoint_path), device="cpu")
    return _tagger


def load_and_resample(path: Path) -> np.ndarray:
    """Load at the file's native rate, downmix to mono, then resample to
    PANN's required 32000 Hz with our own polyphase resampler -- NOT
    librosa's resampling, even though librosa is used for the container
    decode + mono downmix (same division of labor bandpass.py/stft.py
    already draw between "decode a container" and "our own DSP math")."""
    y, sr = librosa.load(str(path), sr=None, mono=True)
    if sr != cfg.SAMPLE_RATE:
        resampler = Resample()
        y = resampler.resample_hand_rolled(y.astype(np.float32), sr, cfg.SAMPLE_RATE)
    return y.astype(np.float32)


def predict(path: Path, top_k: int = cfg.TOP_K, min_confidence: float = cfg.MIN_CONFIDENCE):
    """Returns (ranked, detected, embedding).

    ranked : all 527 (label, prob) pairs, sorted descending, truncated to top_k.
    detected : the subset of ranked (unbounded, not just top_k) at or above
        min_confidence -- this is the "should we report it" list.
    embedding : the 2048-dim clip embedding PANN also returns. Unused for
        zero-shot tagging, but kept in the return value because it's the
        thing you'd need later if this pathway ever grows into a
        fine-tuned head on our own species labels (see pann_config.py).
    """
    tagger = _get_tagger()
    waveform = load_and_resample(path)

    # panns_inference expects a batch axis: (n_clips, n_samples).
    clipwise_output, embedding = tagger.inference(waveform[None, :])
    probs = clipwise_output[0]  # (527,)
    emb = embedding[0]          # (2048,)

    labels = tagger.labels
    ranked = sorted(zip(labels, probs.tolist()), key=lambda kv: -kv[1])
    detected = [(label, prob) for label, prob in ranked if prob >= min_confidence]

    return ranked[:top_k], detected, emb


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--top-k", type=int, default=cfg.TOP_K)
    parser.add_argument("--min-confidence", type=float, default=cfg.MIN_CONFIDENCE)
    args = parser.parse_args()

    ranked, detected, _emb = predict(args.audio_path, args.top_k, args.min_confidence)

    print(f"file: {args.audio_path.name}")
    for label, prob in ranked:
        marker = " <-- above threshold" if prob >= args.min_confidence else ""
        print(f"  {label:40s} {prob:.4f}{marker}")
    detected_labels = [label for label, _ in detected] if detected else "none"
    print(f"detected (>= {args.min_confidence}): {detected_labels}")


if __name__ == "__main__":
    main()
