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

CHANGE FROM THE ORIGINAL VERSION: every parameter that has to match
build_reference_library.py (spec, bandpass, STFT, log-frequency mapping,
matcher fingerprint settings, keypoint-extraction settings) now comes
from pipeline_config.py instead of being redefined here -- the two
files drifting apart silently produces garbage scores, which is exactly
what was starting to happen (build_reference_library.py wasn't even
using the same spectrogram representation as this file). Query-time
windows are also built through the same per-species reference
augmentation-aware matcher (pitch_invariant + time_bucket hashing,
adaptive per-row noise floor).

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

import librosa

from processor.audio_processing.audio_processing import AudioPreprocessor
from processor.audio_processing.bandpass import apply_bandpass
from processor.audio_processing.stft import stft, to_log_frequency_spectrogram
from processor.audio_processing.audio_matcher import AudioMatcher
from processor.audio_processing import pipeline_config as cfg

logger = logging.getLogger(__name__)


def _augmented_waveforms(waveform, sample_rate):
    """Same augmentation grid as build_reference_library.py -- see that
    file's docstring for why references get several pitch/tempo variants
    instead of one. Duplicated here (rather than imported) only because
    this module can't cleanly import a sibling top-level script; if you
    promote build_reference_library.py's version to a shared module,
    replace this copy with an import from there instead."""
    for n_steps in cfg.PITCH_AUGMENT_STEPS:
        for rate in cfg.TEMPO_AUGMENT_RATES:
            if n_steps == 0.0 and rate == 1.0:
                yield "orig", waveform.astype("float32")
                continue
            try:
                y = waveform
                label = "orig"
                if n_steps != 0.0:
                    try:
                        y = librosa.effects.pitch_shift(y=y, sr=sample_rate, n_steps=n_steps)
                    except TypeError:
                        y = librosa.effects.pitch_shift(y, sample_rate, n_steps)
                    label = f"pitch{n_steps:+.0f}"
                if rate != 1.0:
                    try:
                        y = librosa.effects.time_stretch(y=y, rate=rate)
                    except TypeError:
                        y = librosa.effects.time_stretch(y, rate)
                    label = f"{label}_rate{rate:.2f}"
                yield label, y.astype("float32")
            except Exception as exc:
                logger.warning(
                    "Skipping augmentation variant (pitch=%+.1f, rate=%.2f): %s",
                    n_steps, rate, exc,
                )

# --- Configuration -------------------------------------------------------

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

REFERENCES_DIR = os.environ.get(
    "AURALIS_REFERENCES_DIR",
    str(Path(__file__).resolve().parent.parent / "audio_processing" / "samples" / "references"),
)

# Kept as module-level names (imported from pipeline_config) so existing
# callers/tests that referenced processor.REFERENCE_SPEC etc. still work.
REFERENCE_SPEC = cfg.REFERENCE_SPEC
QUERY_SPEC = cfg.QUERY_SPEC
BANDPASS_PARAMS = cfg.BANDPASS_PARAMS
STFT_PARAMS = cfg.STFT_PARAMS
LOG_FREQ_PARAMS = cfg.LOG_FREQ_PARAMS
EXTRACT_PARAMS = cfg.EXTRACT_PARAMS
MATCHER_PARAMS = cfg.MATCHER_PARAMS
CONFIDENCE_THRESHOLD = cfg.CONFIDENCE_THRESHOLD


# --- Reference library: build once from the references folder, reuse -----

_matcher = None


def _compute_log_spectrogram(waveform, sample_rate):
    filtered = apply_bandpass(waveform, sample_rate, **BANDPASS_PARAMS)
    spectrogram_complex, freqs, _times = stft(filtered, sample_rate, **STFT_PARAMS)
    log_spectrogram, _log_freqs = to_log_frequency_spectrogram(
        spectrogram_complex, freqs, **LOG_FREQ_PARAMS
    )
    return log_spectrogram


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
    matcher = AudioMatcher(**MATCHER_PARAMS)
    sample_rate = REFERENCE_SPEC.target_sample_rate

    for path in reference_files:
        species_id = path.stem  # "cat.mp3" -> "cat"
        with open(path, "rb") as file_obj:
            windows = preprocessor.process(file_obj, REFERENCE_SPEC)
        base_waveform = windows[0]  # target_duration=None -> exactly one, whole-clip window

        n_variants = 0
        for label, waveform in _augmented_waveforms(base_waveform, sample_rate):
            spectrogram = _compute_log_spectrogram(waveform, sample_rate)
            variant_ref_id = f"{species_id}{cfg.VARIANT_SEPARATOR}{label}"
            matcher.add_reference(variant_ref_id, spectrogram, **EXTRACT_PARAMS)
            n_variants += 1
            if matcher.reference_hash_counts.get(variant_ref_id, 0) == 0:
                logger.warning(
                    "'%s' produced ZERO fingerprint hashes -- likely too quiet/short "
                    "for the current extract_keypoints settings.", variant_ref_id
                )
        logger.info("Added reference '%s' from %s (%d variants)", species_id, path.name, n_variants)

    return matcher


def _get_matcher():
    """Lazily build the AudioMatcher from the references folder, once per
    process. Rebuilding on every request would be far too slow -- this
    caches the result in-memory after the first call.

    NOTE: this rebuilds the augmented library from raw audio on process
    start, same augmentation grid as build_reference_library.py. If you'd
    rather not pay that cost per-process, point
    AURALIS_REFERENCE_LIBRARY_PATH at build_reference_library.py's saved
    .pkl and load it via AudioMatcher.load() instead -- just make sure
    that .pkl was built with the same pipeline_config.py this process is
    running, since a stale .pkl built under old settings will silently
    produce garbage scores (see pipeline_config.py's module docstring).
    """
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
        detections = matcher.detect(
            spectrogram, variant_separator=cfg.VARIANT_SEPARATOR, **EXTRACT_PARAMS
        )
        for species, info in detections:
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
