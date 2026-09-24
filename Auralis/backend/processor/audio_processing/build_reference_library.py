"""
build_reference_library.py

Run this ONCE (and again any time you add/change reference clips) to
build the AudioMatcher reference library and save it to disk. The Flask
service (processor/services/processor.py) loads the saved .pkl at
runtime -- it does NOT rebuild from raw audio on every request, and it
does NOT accept a folder of audio files as REFERENCE_LIBRARY_PATH.

Usage:
    python build_reference_library.py

Expects, relative to this script:
    samples/references/cat.mp3
    samples/references/dog.mp3
    ...

Produces:
    reference_library.pkl   (next to this script, in audio_processing/)

Make sure AURALIS_REFERENCE_LIBRARY_PATH (if you've set it) points at
THIS output .pkl file, not at the samples/references/ folder.

CHANGE FROM THE ORIGINAL VERSION: each reference clip is no longer added
to the matcher as a single exemplar. It's added several times, each time
pitch-shifted and/or time-stretched by a small amount (PITCH_AUGMENT_STEPS
x TEMPO_AUGMENT_RATES in pipeline_config.py), all merged under the same
species_id. Landmark hashing can only ever recognize hash overlap with
what's actually IN the index -- a single clean reference clip has almost
no hash overlap with a real-world "foreign" recording of the same
species, even before you add pitch_invariant hashing into the mix. Giving
the index several plausible variants directly raises the odds of a real
hit. This also uses the same log-frequency spectrogram + pitch_invariant
hash settings as processor.py (via pipeline_config.py), which the old
version did not.
"""

import logging
import sys
from pathlib import Path

import librosa

try:
    from .audio_processing import AudioPreprocessor
    from .bandpass import apply_bandpass
    from .stft import stft, to_log_frequency_spectrogram
    from .audio_matcher import AudioMatcher
    from . import pipeline_config as cfg
except ImportError:
    from audio_processing import AudioPreprocessor
    from bandpass import apply_bandpass
    from stft import stft, to_log_frequency_spectrogram
    from audio_matcher import AudioMatcher
    import pipeline_config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
REFERENCES_DIR = HERE / "samples" / "references"
OUTPUT_PATH = HERE / "reference_library.pkl"

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


def _augmented_waveforms(waveform, sample_rate, species_id="?"):
    """
    Yield (label, waveform) for every (pitch_step, tempo_rate) combination
    in the augmentation grid, including the unmodified original.

    librosa's phase vocoder (used by pitch_shift/time_stretch) can raise
    on very short clips (some reference calls here are under a second),
    so each variant is generated best-effort: a failure just skips that
    one variant with a warning rather than aborting the whole build --
    you still get the original plus whichever variants succeeded.
    """
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


def _log_spectrogram_for(waveform, sample_rate):
    filtered = apply_bandpass(waveform, sample_rate, **cfg.BANDPASS_PARAMS)
    spectrogram_complex, freqs, _times = stft(filtered, sample_rate, **cfg.STFT_PARAMS)
    log_spectrogram, _log_freqs = to_log_frequency_spectrogram(
        spectrogram_complex, freqs, **cfg.LOG_FREQ_PARAMS
    )
    return log_spectrogram


def build():
    if not REFERENCES_DIR.exists():
        logger.error("References folder not found: %s", REFERENCES_DIR)
        sys.exit(1)

    reference_files = sorted(
        p for p in REFERENCES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
    )
    if not reference_files:
        logger.error("No audio files found in %s", REFERENCES_DIR)
        sys.exit(1)

    preprocessor = AudioPreprocessor()
    matcher = AudioMatcher(**cfg.MATCHER_PARAMS)
    sample_rate = cfg.REFERENCE_SPEC.target_sample_rate

    for path in reference_files:
        species_id = path.stem  # "cat.mp3" -> "cat"
        with open(path, "rb") as file_obj:
            windows = preprocessor.process(file_obj, cfg.REFERENCE_SPEC)
        base_waveform = windows[0]  # target_duration=None -> exactly one, whole-clip window

        n_variants = 0
        for label, waveform in _augmented_waveforms(base_waveform, sample_rate, species_id):
            log_spectrogram = _log_spectrogram_for(waveform, sample_rate)
            variant_ref_id = f"{species_id}{cfg.VARIANT_SEPARATOR}{label}"
            matcher.add_reference(variant_ref_id, log_spectrogram, **cfg.EXTRACT_PARAMS)
            n_variants += 1

        logger.info(
            "Added reference '%s' from %s (%d augmented variants)",
            species_id, path.name, n_variants,
        )

    matcher.save(str(OUTPUT_PATH))
    logger.info("Saved reference library to %s", OUTPUT_PATH)
    species_seen = sorted({rid.split(cfg.VARIANT_SEPARATOR, 1)[0]
                            for rid in matcher.reference_hash_counts})
    logger.info("Species in library: %s", species_seen)
    for variant_ref_id, count in matcher.reference_hash_counts.items():
        if count == 0:
            logger.warning(
                "'%s' produced ZERO fingerprint hashes -- likely too quiet/short "
                "for the current extract_keypoints settings.", variant_ref_id
            )


if __name__ == "__main__":
    build()
