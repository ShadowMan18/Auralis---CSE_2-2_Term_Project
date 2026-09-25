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

CNN PATH (process_cnn_sample): runs your own phase-1 CNN (train_cnn.py ->
species_cnn_phase1.pt, trained by dataset_producer.py on the references
folder). The upload is cut into overlapping 3 s windows -- the length the
CNN was trained on -- every window goes through the SAME feature code used in
training (train_cnn.waveform_to_logmel), and each species keeps its best
window probability, exactly like the DSP path's per-window aggregation. That
is what lets one upload contain several animals at different times.

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
import threading
from io import BytesIO
from pathlib import Path

import librosa
import numpy as np

from processor.audio_processing.audio_processing import AudioPreprocessor
from processor.audio_processing.bandpass import apply_bandpass
from processor.audio_processing.stft import stft, to_log_frequency_spectrogram
from processor.audio_processing.audio_matcher import AudioMatcher
from processor.audio_processing import pipeline_config as cfg
from processor.audio_processing import ml_config as ml_cfg
from processor.audio_processing import pann_config as pann_cfg
from processor.audio_processing.pann_label_mapping import (
    MAPPED_ANIMAL_CATEGORIES,
    map_pann_scores,
)
from processor.audio_processing.resample import Resample

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

# --- CNN path configuration -------------------------------------------------
# The fingerprint matcher below remains available for experimentation; the API
# route uses the CNN path (process_cnn_sample). The checkpoint is built locally
# from your own references (dataset_producer.py, then train_cnn.py) and kept
# out of git: model files are build artifacts, not source. Set
# AURALIS_CNN_MODEL_PATH to use a different checkpoint; its .labels.json must
# sit next to it.
CNN_MODEL_PATH = Path(os.environ.get("AURALIS_CNN_MODEL_PATH", str(ml_cfg.MODEL_PATH)))
# A species is reported when its best window probability reaches this value.
CNN_MIN_CONFIDENCE = ml_cfg.MIN_CONFIDENCE
# 3 s windows (ml_cfg.CLIP_SECONDS) advance by this much: 50% overlap so a call
# that straddles a window boundary is still seen whole in the next window.
CNN_WINDOW_HOP_SECONDS = 1.5
# Only the first CNN_MAX_SECONDS of an upload are analysed (bounds memory/CPU).
CNN_MAX_SECONDS = 120
CNN_BATCH_SIZE = 64

# The client combines the three independently-calibrated scores, so every
# path must report the same candidate vocabulary. PANN may also emit generic
# AudioSet tags (for example "Rain") which are useful for diagnostics but are
# not animal candidates and must not enter the ensemble.
ENSEMBLE_CONFIDENCE_THRESHOLD = 0.20


class _ReplayableUpload:
    """Small FileStorage-compatible copy of one request upload.

    Each inference path consumes the stream differently: DSP decodes from a
    file object while CNN/PANN save it before decoding. A fresh replayable
    instance for each path prevents the first model from leaving the other
    models at EOF, without asking the browser to upload the recording three
    times.
    """

    def __init__(self, filename, payload):
        self.filename = filename
        self._payload = payload
        self.stream = BytesIO(payload)

    def seek(self, *args, **kwargs):
        return self.stream.seek(*args, **kwargs)

    def read(self, *args, **kwargs):
        return self.stream.read(*args, **kwargs)

    def tell(self):
        return self.stream.tell()

    def save(self, destination):
        """Match the FileStorage.save subset used by the CNN/PANN paths."""
        destination.write(self._payload)


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
            spectrogram,
            min_raw_count=cfg.MIN_RAW_MATCH_COUNT,
            variant_separator=cfg.VARIANT_SEPARATOR,
            **EXTRACT_PARAMS,
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

    # The reference library contains multiple recordings per animal
    # (cat_001, cat_002, ...), each with several augmentation variants.
    # Collapse those exemplars to the animal class for the API. Keep the
    # strongest exemplar score: each recording is a separate call/template,
    # so averaging would penalize a valid call that resembles only one of
    # the available references. This remains a fingerprint similarity score,
    # not a calibrated probability.
    species_scores = {}
    for reference_id, score in best_scores.items():
        species = reference_id.split("_", 1)[0]
        if species not in species_scores or score > species_scores[species]:
            species_scores[species] = score

    detected = [
        (species, score) for species, score in species_scores.items()
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


# --- CNN path ----------------------------------------------------------------

_cnn = None
_cnn_lock = threading.Lock()


def _get_cnn():
    """Load the CNN once per process and reuse it (same idea as _get_matcher).

    torch is imported lazily here so the DSP path keeps working on a machine
    without torch. Like the matcher, this is cached for the life of the
    process: after training a new checkpoint, RESTART the server to pick it up.
    """
    global _cnn
    if _cnn is None:
        with _cnn_lock:
            if _cnn is None:
                from processor.audio_processing.predict import load_model
                logger.info("Loading CNN from %s", CNN_MODEL_PATH)
                model, idx_to_class = load_model(CNN_MODEL_PATH)
                logger.info("CNN loaded: %d classes %s", len(idx_to_class), sorted(idx_to_class.values()))
                _cnn = (model, idx_to_class)
    return _cnn


def _cnn_window_starts(n_samples):
    """Start offsets (in samples) of the overlapping CNN windows."""
    window = int(ml_cfg.CLIP_SECONDS * ml_cfg.SAMPLE_RATE)
    hop = max(1, int(CNN_WINDOW_HOP_SECONDS * ml_cfg.SAMPLE_RATE))
    if n_samples <= window:
        return [0]  # short clip: one window, zero-padded exactly as in training
    starts = list(range(0, n_samples - window + 1, hop))
    if starts[-1] + window < n_samples:
        starts.append(n_samples - window)  # keep the tail instead of dropping it
    return starts


def _cnn_window_probs(model, waveform):
    """Softmax probabilities, shape (n_windows, n_classes).

    Each window is exactly CLIP_SECONDS of the upload EXCEPT possibly the
    last one, when the upload is shorter than one window; that window gets
    padded with realistic background noise (train_cnn._pad_with_noise_bed) to
    reach the model's fixed input size. valid_frames tells the model exactly
    how many of those columns are real audio, so it can pool over only the
    real part -- the padded columns get zero weight in the prediction,
    whatever they contain. Full-length windows (the normal case) have
    valid_frames == N_FRAMES, i.e. nothing is masked."""
    import torch
    import torch.nn.functional as F
    from processor.audio_processing.train_cnn import waveform_to_logmel_with_length

    window = int(ml_cfg.CLIP_SECONDS * ml_cfg.SAMPLE_RATE)
    logmels, valid_frames = [], []
    for start in _cnn_window_starts(len(waveform)):
        logmel, n_valid = waveform_to_logmel_with_length(waveform[start:start + window])
        logmels.append(logmel)
        valid_frames.append(n_valid)
    x = torch.from_numpy(np.stack(logmels)).unsqueeze(1)  # (n_windows, 1, n_mels, n_frames)
    valid_frames = torch.tensor(valid_frames, dtype=torch.long)
    batches = []
    with torch.inference_mode():
        for i in range(0, len(x), CNN_BATCH_SIZE):
            logits = model(x[i:i + CNN_BATCH_SIZE], valid_frames=valid_frames[i:i + CNN_BATCH_SIZE])
            batches.append(F.softmax(logits, dim=1))
    return torch.cat(batches).numpy()


def _aggregate_cnn_scores(probs, idx_to_class):
    """Best probability per species across windows -> {species: score}.

    Same rule as predict.py: a window whose winning class is the background
    class is "no detection" and contributes nothing, so silence/noise windows
    can't leak small species scores into the result.
    """
    labels = [idx_to_class[i] for i in range(probs.shape[1])]
    species_cols = [i for i, name in enumerate(labels) if name != ml_cfg.BACKGROUND_LABEL]
    background_cols = [i for i, name in enumerate(labels) if name == ml_cfg.BACKGROUND_LABEL]
    if not species_cols:
        return {}
    keep = ~np.isin(probs.argmax(axis=1), background_cols)
    if not keep.any():
        return {}
    best = probs[keep][:, species_cols].max(axis=0)
    return {labels[col]: float(score) for col, score in zip(species_cols, best)}


def process_cnn_sample(sample, lat=None, lon=None, month=None, top_k=8):
    """Run the trained phase-1 CNN over an uploaded audio file.

    Parameters
    ----------
    sample : werkzeug FileStorage or None -- the 'sample' multipart field.
    lat, lon, month : accepted and ignored. The geo-temporal prior belonged to
        the removed iNatSounds pipeline; the parameters stay so the existing
        route and frontend keep working unchanged.
    top_k : maximum number of species to return (1..50).

    Returns (dict, int) -- the (result, status) contract the route expects.
    ``species``/``confidence`` list only species whose best window reached
    CNN_MIN_CONFIDENCE (same meaning as the DSP path's threshold); ``decision``
    is the strongest of them, or None for "no confident detection".
    """
    if sample is None or not sample.filename:
        return {"error": "No file uploaded. Expected a 'sample' field in multipart/form-data."}, 400
    extension = Path(sample.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return {"error": f"Unsupported file type '{extension}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}"}, 415
    try:
        top_k = max(1, min(int(top_k), 50))
    except (TypeError, ValueError):
        return {"error": "top_k must be an integer."}, 400

    if not CNN_MODEL_PATH.is_file() or not CNN_MODEL_PATH.with_suffix(".labels.json").is_file():
        logger.error("CNN checkpoint or labels file missing at %s", CNN_MODEL_PATH)
        return {"error": "CNN model is not ready. Train it first (dataset_producer.py, then train_cnn.py)."}, 503
    try:
        model, idx_to_class = _get_cnn()
    except Exception:
        logger.exception("Could not load the CNN model from %s", CNN_MODEL_PATH)
        return {"error": "CNN prediction service is not configured correctly."}, 503

    try:
        # librosa needs a filesystem path for every format (mp3/m4a included).
        # NamedTemporaryFile is closed before librosa reopens it (required on
        # Windows) and the file is always removed afterwards.
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temp:
                temp_path = Path(temp.name)
                sample.save(temp)
            waveform, _sr = librosa.load(
                str(temp_path), sr=ml_cfg.SAMPLE_RATE, mono=True, duration=CNN_MAX_SECONDS
            )
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

        if waveform.size == 0:
            return {"error": "The uploaded audio file contains no audio."}, 422
        if waveform.size >= CNN_MAX_SECONDS * ml_cfg.SAMPLE_RATE:
            logger.warning("'%s' is longer than %ds; only the first %ds were analysed.",
                           sample.filename, CNN_MAX_SECONDS, CNN_MAX_SECONDS)

        probs = _cnn_window_probs(model, waveform)
        best_scores = _aggregate_cnn_scores(probs, idx_to_class)
        logger.info("CNN scores (pre-threshold, %d windows) for '%s': %s",
                    len(probs), sample.filename,
                    {k: round(v, 4) for k, v in best_scores.items()})
    except Exception:
        logger.exception("CNN prediction failed for '%s'", sample.filename)
        return {"error": "Failed to process the uploaded audio file."}, 422

    detected = sorted(
        ((species, score) for species, score in best_scores.items() if score >= CNN_MIN_CONFIDENCE),
        key=lambda pair: pair[1], reverse=True,
    )[:top_k]
    predictions = [{"species": species, "confidence": round(score, 6)} for species, score in detected]
    return {
        "decision": predictions[0]["species"] if predictions else None,
        "predictions": predictions,
        # Same flat lists as the DSP path / previous CNN response, for the existing frontend.
        "species": [item["species"] for item in predictions],
        "confidence": [item["confidence"] for item in predictions],
        "geo_prior_applied": False,
    }, 200


# --- PANN zero-shot path (exploratory; no fine-tuning, no species labels) --
# Sits alongside the DSP fingerprint path (process_sample) and the phase-1
# CNN path (process_cnn_sample). See pann_config.py's module docstring for
# why this pathway is a documented exception to "team implements the DSP":
# PANN's internal mel-spectrogram front-end is a library black-box, unlike
# stft.py; only the *resampling* step (native rate -> PANN's required
# 32000 Hz) still goes through our own hand-rolled Resample, same as
# everywhere else in the project.

_pann = None
_pann_lock = threading.Lock()


def _get_pann():
    """Load panns_inference.AudioTagging once per process and reuse it
    (same pattern as _get_matcher / _get_cnn). Imported lazily so the rest
    of the service keeps working on a machine without panns_inference
    installed. Like the CNN checkpoint: if you ever pin a different PANN
    checkpoint, RESTART the server to pick it up -- this is cached for the
    life of the process, same as _matcher and _cnn."""
    global _pann
    if _pann is None:
        with _pann_lock:
            if _pann is None:
                checkpoint_path = pann_cfg.ensure_assets()
                import torch
                from panns_inference import AudioTagging
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(
                    "Loading PANN (%s, AudioSet 527 classes) on %s",
                    pann_cfg.MODEL_TYPE,
                    device,
                )
                _pann = AudioTagging(checkpoint_path=str(checkpoint_path), device=device)
                logger.info("PANN loaded: %d AudioSet labels on %s", len(_pann.labels), device)
    return _pann


def _pann_window_starts(n_samples):
    """Start offsets (in samples, at PANN's 32000 Hz) of overlapping
    analysis windows -- same idea as _cnn_window_starts, different
    window/hop length (pann_config.CLIP_SECONDS / WINDOW_HOP_SECONDS)."""
    window = int(pann_cfg.CLIP_SECONDS * pann_cfg.SAMPLE_RATE)
    hop = max(1, int(pann_cfg.WINDOW_HOP_SECONDS * pann_cfg.SAMPLE_RATE))
    if n_samples <= window:
        return [0]
    starts = list(range(0, n_samples - window + 1, hop))
    if starts[-1] + window < n_samples:
        starts.append(n_samples - window)  # keep the tail instead of dropping it
    return starts


def _pann_window_probs(tagger, waveform):
    """Sigmoid probabilities per window, shape (n_windows, 527).

    NOTE: unlike the CNN path's valid_frames masking (train_cnn.py's
    SmallAudioCNN.forward), a short trailing window here is zero-padded
    with NO masking -- PANN's forward pass has no equivalent hook for
    "ignore these input samples". For a short upload this means the
    padded silence can dilute (not fabricate, but soften) that window's
    scores slightly. Acceptable for an exploratory zero-shot pathway;
    flag if this ever needs tighter handling."""
    window = int(pann_cfg.CLIP_SECONDS * pann_cfg.SAMPLE_RATE)
    starts = _pann_window_starts(len(waveform))
    batch = np.stack([
        np.pad(waveform[s:s + window], (0, max(0, window - len(waveform[s:s + window]))))
        for s in starts
    ])
    clipwise_output, _embedding = tagger.inference(batch)
    return clipwise_output  # (n_windows, 527)


def _aggregate_pann_scores(probs, labels):
    """Best probability per AudioSet label across windows -> {label: score}.

    No 'background' class to exclude here (unlike _aggregate_cnn_scores) --
    PANN's output is a plain 527-way multi-label sigmoid, not a closed-set
    softmax with a trained background/silence class, so there's nothing
    equivalent to filter out at this stage. Thresholding in
    process_pann_sample (pann_cfg.MIN_CONFIDENCE) is what keeps
    quiet/irrelevant tags out of the final result."""
    best = probs.max(axis=0)
    return {label: float(score) for label, score in zip(labels, best)}


def process_pann_sample(sample, top_k=None):
    """Run zero-shot PANN (CNN14/AudioSet) tagging over an uploaded audio
    file. Returns the SAME (dict, int) contract as process_cnn_sample, so
    it can be wired to a route the same way -- but 'species' here are
    generic AudioSet tag names (e.g. 'Bird', 'Dog', 'Rain'), NOT our own
    candidate species list, since this pathway is zero-shot with no
    fine-tuning. Don't feed this straight into UI code that assumes our
    species taxonomy without relabeling/filtering it first.
    """
    if sample is None or not sample.filename:
        return {"error": "No file uploaded. Expected a 'sample' field in multipart/form-data."}, 400
    extension = Path(sample.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return {"error": f"Unsupported file type '{extension}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}"}, 415

    try:
        top_k = pann_cfg.TOP_K if top_k is None else max(1, min(int(top_k), 50))
    except (TypeError, ValueError):
        return {"error": "top_k must be an integer."}, 400

    try:
        tagger = _get_pann()
    except Exception:
        logger.exception("Could not load PANN")
        return {"error": "PANN tagging service is not configured correctly."}, 503

    try:
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temp:
                temp_path = Path(temp.name)
                sample.save(temp)
            # Load at native rate + downmix (container decode only, same as
            # process_cnn_sample's librosa.load call), THEN resample with our
            # own polyphase resampler -- same division of labor as
            # predict_pann.load_and_resample.
            waveform, native_sr = librosa.load(
                str(temp_path), sr=None, mono=True, duration=pann_cfg.MAX_SECONDS
            )
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

        if waveform.size == 0:
            return {"error": "The uploaded audio file contains no audio."}, 422

        if native_sr != pann_cfg.SAMPLE_RATE:
            waveform = Resample().resample_hand_rolled(
                waveform.astype(np.float32), native_sr, pann_cfg.SAMPLE_RATE
            )

        probs = _pann_window_probs(tagger, waveform)
        raw_scores = _aggregate_pann_scores(probs, tagger.labels)
        best_scores = map_pann_scores(
            raw_scores, min_confidence=pann_cfg.MIN_CONFIDENCE
        )
        logger.info(
            "Mapped PANN scores (pre-threshold, %d windows) for '%s': %s",
            len(probs), sample.filename,
            sorted(best_scores.items(), key=lambda kv: -kv[1]),
        )
    except Exception:
        logger.exception("PANN tagging failed for '%s'", sample.filename)
        return {"error": "Failed to process the uploaded audio file."}, 422

    detected = sorted(
        ((label, score) for label, score in best_scores.items() if score >= pann_cfg.MIN_CONFIDENCE),
        key=lambda pair: pair[1], reverse=True,
    )[:top_k]
    predictions = [{"species": label, "confidence": round(score, 6)} for label, score in detected]
    return {
        "decision": predictions[0]["species"] if predictions else None,
        "predictions": predictions,
        # Same flat lists as the DSP/CNN paths, for the existing frontend --
        # but see the docstring above: these are generic AudioSet tags.
        "species": [item["species"] for item in predictions],
        "confidence": [item["confidence"] for item in predictions],
        "zero_shot": True,
    }, 200


# --- Ensemble upload path ---------------------------------------------------

def _read_upload_payload(sample):
    """Read an upload once and restore its request stream when possible."""
    source = getattr(sample, "stream", sample)
    try:
        source.seek(0)
    except (AttributeError, OSError):
        pass
    payload = source.read()
    try:
        source.seek(0)
    except (AttributeError, OSError):
        pass
    return payload


def _threshold_predictions(predictions, allowed_species=None):
    """Return the stable API shape and enforce the common ensemble cutoff."""
    best_scores = {}
    for prediction in predictions or []:
        species = str(prediction.get("species", "")).strip()
        try:
            score = float(prediction.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        if not species or score <= ENSEMBLE_CONFIDENCE_THRESHOLD:
            continue
        if allowed_species is not None and species not in allowed_species:
            continue
        best_scores[species] = max(best_scores.get(species, 0.0), score)

    return [
        {"species": species, "confidence": round(score, 6)}
        for species, score in sorted(best_scores.items(), key=lambda item: (-item[1], item[0]))
    ]


def _flat_scores_to_predictions(result):
    """Adapt the original DSP response to the common prediction shape."""
    return [
        {"species": species, "confidence": score}
        for species, score in zip(result.get("species", []), result.get("confidence", []))
    ]


def process_ensemble_sample(sample):
    """Run one uploaded recording through DSP, the trained CNN, and PANN.

    The endpoint deliberately returns *unweighted* per-path predictions. The
    frontend owns the requested final score calculation:

        0.3 * pann + 0.4 * cnn + 0.3 * dsp

    A path failing to initialise (for example, a missing local CNN checkpoint)
    is reported in ``path_errors`` while the other available paths still return
    their findings. That makes deployment problems visible without throwing
    away a valid result from the remaining models.
    """
    if sample is None or not sample.filename:
        return {"error": "No file uploaded. Expected a 'sample' field in multipart/form-data."}, 400

    extension = Path(sample.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return {
            "error": f"Unsupported file type '{extension}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        }, 415

    try:
        payload = _read_upload_payload(sample)
    except Exception:
        logger.exception("Could not read uploaded sample '%s'", sample.filename)
        return {"error": "Failed to read the uploaded audio file."}, 422
    if not payload:
        return {"error": "The uploaded audio file is empty."}, 422

    # Keep all path calls in one place so their input and common post-filtering
    # cannot drift. `top_k=50` is safely above the current class counts and
    # avoids silently dropping an animal that is above the requested threshold.
    calls = {
        "dsp": lambda upload: process_sample(upload),
        "cnn": lambda upload: process_cnn_sample(upload, top_k=50),
        "pann": lambda upload: process_pann_sample(upload, top_k=50),
    }
    paths = {name: [] for name in calls}
    path_errors = {}
    successful_paths = 0

    for name, run_path in calls.items():
        try:
            result, status = run_path(_ReplayableUpload(sample.filename, payload))
        except Exception:
            logger.exception("Unexpected %s failure for '%s'", name.upper(), sample.filename)
            result, status = {"error": "Unexpected processing failure."}, 500

        if status != 200:
            path_errors[name] = result.get("error", f"{name.upper()} processing failed.")
            continue

        successful_paths += 1
        raw_predictions = (
            _flat_scores_to_predictions(result)
            if name == "dsp"
            else result.get("predictions", [])
        )
        # PANN is multi-label AudioSet tagging. Only labels explicitly mapped
        # to Auralis animal categories can be combined with DSP/CNN classes.
        allowed_species = MAPPED_ANIMAL_CATEGORIES if name == "pann" else None
        paths[name] = _threshold_predictions(raw_predictions, allowed_species)

    if successful_paths == 0:
        return {
            "error": "None of the detection paths is currently available.",
            "paths": paths,
            "path_errors": path_errors,
            "threshold": ENSEMBLE_CONFIDENCE_THRESHOLD,
        }, 503

    return {
        "paths": paths,
        "path_errors": path_errors,
        "threshold": ENSEMBLE_CONFIDENCE_THRESHOLD,
    }, 200
