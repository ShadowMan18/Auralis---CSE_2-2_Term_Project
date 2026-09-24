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

CHANGE FROM THE PREVIOUS VERSION: pitch/tempo/noise-tolerant matching.
Every parameter that has to match build_reference_library.py (spec,
bandpass, STFT, log-frequency mapping, matcher fingerprint settings,
keypoint-extraction settings, the augmentation grid) now comes from
processor.audio_processing.pipeline_config instead of being redefined
here -- put pipeline_config.py in processor/audio_processing/ alongside
the other audio_processing modules. This file's own whole-clip QUERY_SPEC
(target_duration=None, hop_duration=None) and CONFIDENCE_THRESHOLD=0.0
are preserved as-is from your current deployed version.

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
import tempfile
from pathlib import Path

import librosa

from processor.audio_processing.audio_processing import AudioPreprocessor
from processor.audio_processing.bandpass import apply_bandpass
from processor.audio_processing.stft import stft, to_log_frequency_spectrogram
from processor.audio_processing.audio_matcher import AudioMatcher
from processor.audio_processing import pipeline_config as cfg

logger = logging.getLogger(__name__)

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

# The old fingerprint matcher below remains available for experimentation,
# but the API route uses this CNN path. Keeping the checkpoint outside git is
# intentional: model artifacts are large and must be built from the licensed
# iNatSounds data locally.
CNN_MODEL_PATH = Path(os.environ.get(
    "AURALIS_CNN_MODEL_PATH",
    str(Path(__file__).resolve().parent.parent / "audio_processing" / "species_cnn_v2.pt"),
))
CNN_GEO_PRIOR_PATH = os.environ.get("AURALIS_GEO_PRIOR_JSON")


def _augmented_waveforms(waveform, sample_rate, species_id="?"):
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
                    "'%s': skipping augmentation variant (pitch=%+.1f, rate=%.2f) -- "
                    "clip likely too short for this stretch factor: %s",
                    species_id, n_steps, rate, exc,
                )


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
        for label, waveform in _augmented_waveforms(base_waveform, sample_rate, species_id):
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

    IMPORTANT: this is a module-level global built once per process. If
    your server was already running when you deployed these changes, it
    is STILL using the matcher it built at startup -- restart the
    process (don't just reload/save) to pick up the new pipeline.
    """
    global _matcher
    if _matcher is None:
        logger.info("Building reference library from %s", REFERENCES_DIR)
        _matcher = _build_matcher_from_references()
        n_species = len({rid.split(cfg.VARIANT_SEPARATOR)[0] for rid in _matcher.reference_hash_counts})
        logger.info(
            "Reference library built: %d species (%d variant entries)",
            n_species, len(_matcher.reference_hash_counts),
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


def _optional_geo_prior():
    """Load the project's explicit region/season table only when configured."""
    if not CNN_GEO_PRIOR_PATH:
        return None
    from processor.audio_processing import geo_prior
    path = Path(CNN_GEO_PRIOR_PATH)
    if not path.is_file():
        raise FileNotFoundError(f"Configured geo-prior JSON was not found: {path}")
    return geo_prior.JsonGeoPrior(path)


def _parse_geo_inputs(lat, lon, month):
    """Return validated floats/integers, allowing all three to be omitted."""
    supplied = [value is not None and str(value).strip() != "" for value in (lat, lon, month)]
    if any(supplied) and not all(supplied):
        raise ValueError("Provide latitude, longitude, and month together, or omit all three.")
    if not any(supplied):
        return None, None, None
    lat, lon, month = float(lat), float(lon), int(month)
    if not -90 <= lat <= 90 or not -180 <= lon <= 180 or not 1 <= month <= 12:
        raise ValueError("latitude must be -90..90, longitude -180..180, and month 1..12.")
    return lat, lon, month


def process_cnn_sample(sample, lat=None, lon=None, month=None, top_k=8):
    """Run the trained species CNN and return ranked species predictions.

    Form fields: ``sample`` plus optional ``lat``/``lon``/``month``. The
    latter three activate a configured JSON geo-prior, never change the CNN
    training labels, and are therefore safe to omit until the lookup table
    has been prepared.
    """
    if sample is None or not sample.filename:
        return {"error": "No file uploaded. Expected a 'sample' field in multipart/form-data."}, 400
    extension = Path(sample.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return {"error": f"Unsupported file type '{extension}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}"}, 415
    if not CNN_MODEL_PATH.is_file() or not CNN_MODEL_PATH.with_suffix(".labels.json").is_file():
        return {
            "error": "CNN model is not ready. Complete the iNatSounds download, manifest, and training steps first."
        }, 503

    try:
        lat, lon, month = _parse_geo_inputs(lat, lon, month)
        top_k = max(1, min(int(top_k), 50))
        geo = _optional_geo_prior()
        if any(value is not None for value in (lat, lon, month)) and geo is None:
            logger.warning("Location/date supplied but AURALIS_GEO_PRIOR_JSON is not configured; using CNN-only scores.")

        # librosa/predict_v2 need a filesystem path. NamedTemporaryFile is
        # closed before Windows reopens it, then always removed afterwards.
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temp:
                temp_path = Path(temp.name)
                sample.save(temp)
            from processor.audio_processing import predict_v2
            decision, ranked = predict_v2.predict(
                temp_path, CNN_MODEL_PATH, lat=lat, lon=lon, month=month, geo_prior_obj=geo
            )
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
    except ValueError as exc:
        return {"error": str(exc)}, 400
    except FileNotFoundError as exc:
        logger.error("CNN prediction configuration error: %s", exc)
        return {"error": "CNN prediction service is not configured correctly."}, 503
    except Exception:
        logger.exception("CNN prediction failed for '%s'", sample.filename)
        return {"error": "Failed to process the uploaded audio file."}, 422

    predictions = [
        {"species": label, "confidence": round(float(confidence), 6)}
        for label, confidence in ranked[:top_k]
    ]
    return {
        "decision": predictions[0]["species"] if decision and predictions else None,
        "predictions": predictions,
        # Retain the prior response shape for the existing frontend client.
        "species": [item["species"] for item in predictions],
        "confidence": [item["confidence"] for item in predictions],
        "geo_prior_applied": geo is not None and lat is not None,
    }, 200
