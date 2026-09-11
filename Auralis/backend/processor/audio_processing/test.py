"""
processor/services/processor.py

Business logic for POST /api/upload-sample. The route handler stays thin
(pull the file off the request, call process_sample, jsonify the result);
all validation, DSP pipeline wiring, and formatting lives here so it can
be unit-tested without Flask/JWT in the loop.

Builds the AudioMatcher reference library directly from the audio files
in AURALIS_REFERENCES_DIR (default: samples/references/) -- no separate
build-and-save-a-.pkl step. This is built ONCE per process (cached in
_matcher below), not per request; rebuilding from raw audio on every
upload would be far too slow.

NOTE on imports below: your bandpass.py currently imports via
`from processor.audio_processing.resample import Resample`
while routes.py imports via `from processor.services import processor`.
Those two imply different PYTHONPATH roots. I've used the same style as
bandpass.py here -- adjust to match whichever one your actual run
config (wsgi entrypoint / Docker WORKDIR) uses, since both can't be
correct simultaneously.
"""

import logging
import os
from pathlib import Path

from processor.audio_processing.audio_processing import (
    AudioPreprocessor,
    ConsumerSpec,
)
from processor.audio_processing.bandpass import apply_bandpass
from processor.audio_processing.stft import stft, log_magnitude_spectrogram
from processor.audio_processing.audio_matcher import AudioMatcher

logger = logging.getLogger(__name__)

# --- Configuration -------------------------------------------------------

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

REFERENCES_DIR = os.environ.get(
    "AURALIS_REFERENCES_DIR",
    str(Path(__file__).resolve().parent.parent / "audio_processing" / "samples" / "references"),
)

# Must exactly match whatever's used for the query spectrograms below --
# a mismatch here doesn't error, it just silently produces garbage scores
# (the whole hash scheme depends on identical freq-bin/time-bin mapping).
REFERENCE_SPEC = ConsumerSpec(
    target_sample_rate=22050,
    target_channels=1,
    target_duration=None,  # whole clip, no windowing -- one reference call = one unit
)
QUERY_SPEC = ConsumerSpec(
    target_sample_rate=22050,
    target_channels=1,
    target_duration=3.0,
    hop_duration=1.5,
    amplitude_range=(-1.0, 1.0),
)
BANDPASS_PARAMS = dict(low_cutoff_hz=50.0, high_cutoff_hz=10000.0, transition_bandwidth_hz=200.0)
STFT_PARAMS = dict(window_length_sec=0.046, hop_length_sec=0.012, window_type="hann")

# Below this, a species is not reported -- still a placeholder. Given the
# noise-sensitivity we measured earlier (clean self-match ~0.8, noisy
# mixture ~0.04-0.08), 0.15 may be rejecting real detections on noisy
# input -- validate against real score distributions once you have some,
# rather than trusting this number as-is.
CONFIDENCE_THRESHOLD = 0.15


# --- Reference library: build once from the references folder, reuse -----

_matcher = None


def _compute_log_spectrogram(waveform, sample_rate):
    filtered = apply_bandpass(waveform, sample_rate, **BANDPASS_PARAMS)
    spectrogram_complex, _freqs, _times = stft(filtered, sample_rate, **STFT_PARAMS)
    return log_magnitude_spectrogram(spectrogram_complex)


def _build_matcher_from_references():
    references_dir = Path(REFERENCES_DIR)
    if not references_dir.exists():
        raise FileNotFoundError(f"References folder not found: {references_dir}")

    reference_files = sorted(
        p for p in references_dir.iterdir()
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
    )
    if not reference_files:
        raise FileNotFoundError(f"No reference audio files found in {references_dir}")

    preprocessor = AudioPreprocessor()
    matcher = AudioMatcher()

    for path in reference_files:
        species_id = path.stem  # "cat.mp3" -> "cat"
        with open(path, "rb") as file_obj:
            windows = preprocessor.process(file_obj, REFERENCE_SPEC)
        waveform = windows[0]  # target_duration=None -> exactly one, whole-clip window
        spectrogram = _compute_log_spectrogram(waveform, REFERENCE_SPEC.target_sample_rate)
        matcher.add_reference(species_id, spectrogram)
        logger.info("Added reference '%s' from %s", species_id, path.name)

        if matcher.reference_hash_counts.get(species_id, 0) == 0:
            logger.warning(
                "'%s' produced ZERO fingerprint hashes -- likely too quiet/short "
                "for the current extract_keypoints settings.", species_id
            )

    return matcher


def _get_matcher():
    """Lazily build the AudioMatcher from the references folder, once per
    process. Rebuilding on every request would be far too slow -- this
    caches the result in-memory after the first call."""
    global _matcher
    if _matcher is None:
        logger.info("Building reference library from %s", REFERENCES_DIR)
        _matcher = _build_matcher_from_references()
        logger.info(
            "Reference library built: %d species", len(_matcher.reference_hash_counts)
        )
    return _matcher


def _detect_in_windows(matcher, windows, sample_rate):
    """Run matcher.detect() per window; keep the best (max) score per
    species across the whole recording -- Stage 8 window aggregation."""
    best_scores = {}
    for window in windows:
        spectrogram = _compute_log_spectrogram(window, sample_rate)
        for species, info in matcher.detect(spectrogram):
            if species not in best_scores or info["score"] > best_scores[species]:
                best_scores[species] = info["score"]
    return best_scores


# --- Public entry point ----------------------------------------------------

def process_sample(sample):
    """
    Run an uploaded audio file through the DSP detection pipeline.

    Parameters
    ----------
    sample : werkzeug.datastructures.FileStorage or None
        The 'sample' file from request.files.get('sample').

    Returns
    -------
    (dict, int) -- matches the (result, status) contract the route
    handler expects: `return jsonify(result), status`.
    """
    if sample is None or not sample.filename:
        return {
            "error": "No file uploaded. Expected a 'sample' field in multipart/form-data."
        }, 400

    extension = Path(sample.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return {
            "error": f"Unsupported file type '{extension}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        }, 415

    try:
        matcher = _get_matcher()
    except FileNotFoundError as exc:
        logger.error("Reference library unavailable: %s", exc)
        return {"error": "Detection service is not ready (no reference audio found)."}, 503

    try:
        # sample is a werkzeug FileStorage -- it delegates read/seek/tell to
        # its underlying stream, so it satisfies the file_obj interface
        # AudioPreprocessor.process() expects directly. No temp file needed.
        preprocessor = AudioPreprocessor()
        windows = preprocessor.process(sample, QUERY_SPEC)

        best_scores = _detect_in_windows(matcher, windows, QUERY_SPEC.target_sample_rate)
        logger.info("Raw scores (pre-threshold) for '%s': %s", sample.filename, best_scores)

    except Exception:
        logger.exception("Failed to process uploaded sample '%s'", sample.filename)
        return {"error": "Failed to process the uploaded audio file."}, 422

    detected = [
        (species, score) for species, score in best_scores.items()
        if score >= CONFIDENCE_THRESHOLD
    ]
    detected.sort(key=lambda pair: pair[1], reverse=True)

    logger.info(
        "Processed '%s': %d species above threshold (of %d candidates)",
        sample.filename, len(detected), len(best_scores),
    )

    return {
        "species": [species for species, _ in detected],
        "confidence": [round(score, 4) for _, score in detected],
    }, 200