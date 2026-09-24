"""
ml_config_v2.py

Shared constants for the phase-2 pipeline:
    inatsounds_route.py -> build_window_manifest.py -> train_cnn_v2.py -> predict_v2.py

This is a SEPARATE file from ml_config.py (the phase-1 farm-sound config).
Phase-1's dataset_producer.py / train_cnn.py / predict.py still import
ml_config.py and are untouched -- they were a pipeline-validation exercise
on placeholder classes, not the real classifier. Everything below is for
the real iNatSounds-trained, 2000+ class, wild-species classifier.

As with ml_config.py: every value under "Audio / feature extraction" MUST
be identical between training (train_cnn_v2.py) and inference
(predict_v2.py), or the model silently sees a different input
distribution than it was trained on -- no error, just quietly wrong
predictions. Both import this module rather than redefining anything.
"""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- Audio / feature extraction (train and inference must match) -----------
# Values below reflect the recipe improvements adopted from the iNatSounds
# paper (see workflow summary, "next steps" #5), applied on top of the
# phase-1 defaults in ml_config.py:
#   - 128 mel bins over [50Hz, 11025Hz] (was 64 / 10000Hz — 11025Hz is the
#     true Nyquist limit at this sample rate, so we're no longer discarding
#     the top ~500Hz of usable spectrum for no reason)
#   - windows are 3s strided by 1.5s across the FULL recording (was: one
#     fixed 3s crop/pad per file) -- see build_window_manifest.py

SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128
FMIN = 50.0
FMAX = 11025.0

WINDOW_SECONDS = 3.0
WINDOW_STRIDE_SECONDS = 1.5
# Fixed number of time frames a WINDOW_SECONDS window produces at this hop
# -- computed once so the CNN always sees a fixed-size input regardless of
# minor per-file rounding in framing.
N_FRAMES = 1 + int(WINDOW_SECONDS * SAMPLE_RATE) // HOP_LENGTH

# --- SpecAugment (train-time only; see train_cnn_v2.py) --------------------

SPECAUGMENT = True
SPECAUG_N_FREQ_MASKS = 2
SPECAUG_FREQ_MASK_MAX = 16     # mel bins
SPECAUG_N_TIME_MASKS = 2
SPECAUG_TIME_MASK_MAX = 24     # frames

# --- Mixup (train-time only) -------------------------------------------

MIXUP = True
MIXUP_ALPHA = 0.4   # Beta(alpha, alpha); paper reports this as the single
                     # biggest ablated improvement of everything in step 5

# --- Loss -----------------------------------------------------------------
# "ce": plain softmax cross-entropy, single label per window (multiclass).
# "assume_negative_bce": sigmoid + binary cross-entropy per class, treating
#   every non-annotated class as a negative for that window. The iNatSounds
#   paper found this transferred better to real downstream (non-weakly-
#   labeled) audio than multiclass softmax, despite multiclass scoring
#   higher on their own weakly-labeled eval split -- i.e. multiclass looks
#   better on the metric that matches its own assumption, but this project's
#   actual use case (real field recordings) is closer to their downstream
#   transfer setting. Default follows their recommendation; "ce" is kept
#   available for direct comparison since it's a one-flag switch.
LOSS = "assume_negative_bce"

# --- Model -----------------------------------------------------------------
# SmallAudioCNN (phase-1, 3 conv blocks / 64 channels) was sized for ~9
# classes on 1 augmented clip each. A 2000+-way classifier over real,
# diverse recordings needs real capacity -- per the iNatSounds paper's own
# benchmarks (mobilenet / resnet18 / resnet50 / vit), not a from-scratch
# toy net. train_cnn_v2.py builds these via torchvision and replicates the
# single-channel log-mel to 3 channels so ImageNet-pretrained weights are
# usable, same approach the official iNatSounds dataset.py takes
# (img = np.stack([img]*3)).
MODEL_ARCH = "resnet50"          # one of: resnet18, resnet50, mobilenet_v3_large
PRETRAINED = True

# --- Dataset layout ----------------------------------------------------

# Root the routed, per-final-class iNatSounds audio ends up in (see
# inatsounds_route.py). Layout: <ROOT>/<split>/<class_slug>/*.{wav,flac,...}
ROUTED_AUDIO_DIR = HERE / "inat_routed"
BACKGROUND_LABEL = "_background"

# species_class_assignments.json produced by build_taxonomy_classes.py.
CLASS_ASSIGNMENTS_PATH = HERE / "species_class_assignments.json"
# class_label_map.json produced by inatsounds_route.py: slug -> display label
# + resolution ("species"/"genus"/"family"/"order"), so predict_v2.py can
# show "Panthera spp. (genus-level)" instead of a bare filesystem slug.
CLASS_LABEL_MAP_PATH = ROUTED_AUDIO_DIR / "class_label_map.json"

# Window manifest produced by build_window_manifest.py.
MANIFEST_DIR = HERE / "manifests"

# --- Training -----------------------------------------------------------

MODEL_PATH = HERE / "species_cnn_v2.pt"
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
VAL_FRACTION = 0.15   # split at the RECORDING level, not the window level
                      # (see build_window_manifest.py) -- otherwise
                      # adjacent, near-identical windows from the same
                      # recording leak between train/val and inflate val
                      # accuracy without meaning anything.
NUM_WORKERS = 4
# iNatSounds is extremely long-tailed.  A value of 0.5 means a class with
# 100x more windows is sampled only 10x more often, rather than 100x. Set to
# 0.0 to preserve the raw archive distribution. This is a pragmatic middle
# ground: fully uniform sampling can overfit the smallest rolled-up classes.
CLASS_BALANCE_POWER = 0.5

# Optional lookup table for user-provided latitude, longitude and month.
# Leave unset to return CNN-only predictions. A region name must be resolved
# to coordinates by the client or a separate gazetteer before it reaches the
# model; a name alone is not an unambiguous geographic prior.
GEO_PRIOR_JSON = os.environ.get("AURALIS_GEO_PRIOR_JSON")

# --- Inference ------------------------------------------------------------

# Below this probability (even for the winning class), report "no
# confident detection" rather than forcing an answer.
MIN_CONFIDENCE = 0.15

# How per-window probabilities are combined into one per-recording score.
# "max": a species that calls once in a long recording shouldn't be
#   diluted by all the silent/other windows around it -- standard choice
#   for weakly-labeled audio tagging. "mean" is offered for comparison.
WINDOW_AGGREGATION = "max"
