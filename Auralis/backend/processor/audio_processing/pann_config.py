"""
pann_config.py

Configuration for the PANN (Pretrained Audio Neural Networks) zero-shot
tagging pathway -- a second, exploratory pipeline alongside the DSP
fingerprint matcher (process_sample) and the phase-1 CNN
(process_cnn_sample), using Kong et al.'s CNN14 checkpoint trained on
AudioSet (527 general-purpose sound classes), used AS-IS with NO
fine-tuning on our own species labels.

Documented exception to "team implements the DSP, no library black-boxes":
PANN's internal mel-spectrogram front-end (torchlibrosa layers baked into
the checkpoint) is a black-box -- there's no way to swap in our own
stft.py without discarding the pretrained weights, and this pathway's
whole point is to use the pretrained weights as-is. Accepted because this
is an exploratory second pipeline, not the graded DSP deliverable (the
DSP path -- bandpass.py -> stft.py -> to_log_frequency_spectrogram -- is
untouched by this). The one piece of DSP that DOES still go through our
own code even here is resampling: PANN requires 32000 Hz input, and we
convert to it with our own hand-rolled Resample (resample.py), same as
everywhere else in the project.
"""

import os
import hashlib
import shutil
import urllib.request
from pathlib import Path

# PANN's required input rate. Fixed by the checkpoint's architecture --
# not a tunable preprocessing choice. Everything below it (window/hop/
# mel bins/fmin/fmax) mirrors the official "Cnn14" config from
# qiuqiangkong/audioset_tagging_cnn and panns_inference; only listed here
# for reference/debugging, since panns_inference sets these internally
# once you pick MODEL_TYPE.
SAMPLE_RATE = 32000
MODEL_TYPE = "Cnn14"
WINDOW_SIZE = 1024
HOP_SIZE = 320
MEL_BINS = 64
FMIN = 50
FMAX = 14000

# Cache the AudioSet labels and official checkpoint under ~/panns_data.
# panns_inference itself uses Unix `wget`, unavailable in a default Windows
# install, so ensure_assets() downloads missing files with Python. Set
# PANN_CHECKPOINT_PATH to pin an existing local checkpoint instead.
DATA_DIR = Path.home() / "panns_data"
LABELS_CSV_PATH = DATA_DIR / "class_labels_indices.csv"
_CUSTOM_CHECKPOINT = "PANN_CHECKPOINT_PATH" in os.environ
CHECKPOINT_PATH = Path(os.environ.get(
    "PANN_CHECKPOINT_PATH", str(DATA_DIR / "Cnn14_mAP=0.431.pth")
))

LABELS_URL = "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
CHECKPOINT_URLS = (
    # Verified mirror of the checkpoint. The older Zenodo URL can return a
    # small HTML page instead of the model file, so it is only a fallback.
    "https://huggingface.co/thelou1s/panns-inference/resolve/main/Cnn14_mAP%3D0.431.pth?download=true",
    "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1",
)
CHECKPOINT_BYTES = 327_428_481
CHECKPOINT_SHA256 = "0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31"


def _download_if_invalid(path: Path, url: str, is_valid):
    """Download an asset atomically, avoiding panns_inference's `wget` calls."""
    if path.is_file() and is_valid(path):
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".part")
    urls = (url,) if isinstance(url, str) else url
    last_error = None
    for candidate_url in urls:
        try:
            with urllib.request.urlopen(candidate_url, timeout=60) as response:
                with temporary_path.open("wb") as output:
                    shutil.copyfileobj(response, output)
            if not is_valid(temporary_path):
                raise RuntimeError(f"Downloaded PANN asset is invalid: {path.name}")
            temporary_path.replace(path)
            return
        except Exception as exc:
            last_error = exc
        finally:
            temporary_path.unlink(missing_ok=True)

    raise RuntimeError(f"Could not download a valid PANN asset: {path.name}") from last_error


def _valid_labels(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as source:
            return source.readline().strip().startswith("index,mid,display_name")
    except (OSError, UnicodeError):
        return False


def _valid_checkpoint(path: Path) -> bool:
    try:
        if path.stat().st_size != CHECKPOINT_BYTES:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest() == CHECKPOINT_SHA256
    except OSError:
        return False


def ensure_assets() -> Path:
    """Ensure AudioSet labels and the CNN14 checkpoint exist in the cache.

    Returns the checkpoint path. Downloads happen on first PANN use only;
    subsequent calls reuse the validated files.
    """
    _download_if_invalid(LABELS_CSV_PATH, LABELS_URL, _valid_labels)
    if _CUSTOM_CHECKPOINT:
        if not _valid_checkpoint(CHECKPOINT_PATH):
            raise FileNotFoundError(
                f"Configured PANN checkpoint is missing or incomplete: {CHECKPOINT_PATH}"
            )
    else:
        _download_if_invalid(CHECKPOINT_PATH, CHECKPOINT_URLS, _valid_checkpoint)
    return CHECKPOINT_PATH

# Sliding-window settings, mirroring ml_config.CLIP_SECONDS / the CNN
# path's CNN_WINDOW_HOP_SECONDS -- so one upload can still register
# several different sounds heard at different times, same behavior as
# the CNN path, even though this path does no training of its own.
CLIP_SECONDS = 6.0
WINDOW_HOP_SECONDS = 3.0
MAX_SECONDS = 120  # only the first N seconds of an upload are analysed

# A tag is reported when its best window's probability reaches this.
# PANN's 527-way SIGMOID output is multi-label and generally not as
# sharply peaked as the phase-1 CNN's closed-set softmax -- this is a
# starting point, not a transferred value from ml_config.MIN_CONFIDENCE.
# Tune it against your own test clips (see test_pann.py).
MIN_CONFIDENCE = 0.15

# How many AudioSet tags to keep per report, before thresholding.
TOP_K = 10
