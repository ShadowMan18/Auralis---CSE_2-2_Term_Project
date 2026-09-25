"""Shared removal of leading and trailing silence from audio waveforms.

Only the quiet regions at the beginning and end are removed. Pauses within a
recording are intentionally preserved, because they can separate distinct
animal calls and are handled by later pipeline stages when needed.
"""

import numpy as np
import librosa


# A sample 35 dB below the recording's peak is treated as silence. This is
# deliberately shared by preprocessing and trim_references.py so references
# and uploaded recordings receive the same treatment.
DEFAULT_TRIM_TOP_DB = 35.0
SILENCE_PEAK_EPSILON = 1e-4


def trim_leading_trailing_silence(
    samples: np.ndarray, top_db: float = DEFAULT_TRIM_TOP_DB
):
    """Return ``(trimmed_samples, [start, end])`` for an audio waveform.

    ``samples`` may be mono ``(frames,)`` or channel-first ``(channels,
    frames)``. All channels use the same boundaries to keep them aligned.
    Fully silent input is intentionally returned unchanged: downstream
    segmentation can still create its expected padded window instead of
    receiving an empty waveform.
    """
    samples = np.asarray(samples)
    if samples.ndim not in (1, 2):
        raise ValueError("samples must have shape (frames,) or (channels, frames)")

    n_frames = samples.shape[-1]
    full_index = np.array([0, n_frames], dtype=int)
    if n_frames == 0 or float(np.max(np.abs(samples))) < SILENCE_PEAK_EPSILON:
        return samples.astype(np.float32, copy=False), full_index

    trimmed, index = librosa.effects.trim(samples, top_db=top_db)
    # Do not turn an unusual thresholding edge case into an empty downstream
    # input. The reference-trimming command reports near-empty clips for review.
    if trimmed.shape[-1] == 0:
        return samples.astype(np.float32, copy=False), full_index
    return trimmed.astype(np.float32, copy=False), index.astype(int, copy=False)
