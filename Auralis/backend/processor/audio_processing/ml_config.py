"""
ml_config.py

Shared constants for dataset_producer.py, train_cnn.py, and predict.py --
same "single source of truth" reasoning as the DSP pipeline's
pipeline_config.py: feature extraction settings here MUST be identical
between training and inference, or the model sees a different input
distribution than it was trained on with no error, just silently bad
predictions.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent

# Hardcoded to match your actual layout: these files live directly in
# processor/audio_processing/, alongside samples/references/. If you move
# these files elsewhere, either update this path or pass
# --references-dir explicitly on the command line.
REFERENCES_DIR = HERE / "samples" / "references"

# --- Audio / feature extraction (train and inference must match) -----------

SAMPLE_RATE = 22050
CLIP_SECONDS = 3.0          # every sample is padded/cropped to exactly this
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
FMIN = 50.0
FMAX = 10000.0
# Fixed number of time frames a CLIP_SECONDS clip produces at this hop --
# computed once so the CNN always sees a fixed-size input.
N_FRAMES = 1 + int(CLIP_SECONDS * SAMPLE_RATE) // HOP_LENGTH

# --- Dataset layout ----------------------------------------------------

DATASET_DIR = HERE / "dataset"
BACKGROUND_LABEL = "_background"

# --- Training -----------------------------------------------------------

MODEL_PATH = HERE / "species_cnn_phase1.pt"
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 1e-3
VAL_FRACTION = 0.15

# --- Inference ------------------------------------------------------------

# Below this softmax probability (even for the winning class), report "no
# confident detection" rather than forcing an answer -- a closed-set
# classifier will otherwise always pick something, even for silence, wind,
# or a totally unrelated sound.
MIN_CONFIDENCE = 0.2