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
"""

import logging
import sys
from pathlib import Path

try:
    from .audio_processing import AudioPreprocessor, ConsumerSpec
    from .bandpass import apply_bandpass
    from .stft import stft, log_magnitude_spectrogram
    from .audio_matcher import AudioMatcher
except ImportError:
    from audio_processing import AudioPreprocessor, ConsumerSpec
    from bandpass import apply_bandpass
    from stft import stft, log_magnitude_spectrogram
    from audio_matcher import AudioMatcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
REFERENCES_DIR = HERE / "samples" / "references"
OUTPUT_PATH = HERE / "reference_library.pkl"

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

# Must be IDENTICAL to the values used in processor/services/processor.py --
# a mismatch here won't error, it'll just silently produce garbage scores
# at request time (freq-bin/time-bin indices stop lining up).
REFERENCE_SPEC = ConsumerSpec(
    target_sample_rate=22050,
    target_channels=1,
    target_duration=None,  # whole clip, no windowing -- one reference call = one unit
)
BANDPASS_PARAMS = dict(low_cutoff_hz=50.0, high_cutoff_hz=10000.0, transition_bandwidth_hz=200.0)
STFT_PARAMS = dict(window_length_sec=0.046, hop_length_sec=0.012, window_type="hann")


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
    matcher = AudioMatcher()

    for path in reference_files:
        species_id = path.stem  # "cat.mp3" -> "cat"
        with open(path, "rb") as file_obj:
            windows = preprocessor.process(file_obj, REFERENCE_SPEC)
        waveform = windows[0]  # target_duration=None -> exactly one, whole-clip window

        filtered = apply_bandpass(waveform, REFERENCE_SPEC.target_sample_rate, **BANDPASS_PARAMS)
        spectrogram_complex, _freqs, _times = stft(filtered, REFERENCE_SPEC.target_sample_rate, **STFT_PARAMS)
        log_spectrogram = log_magnitude_spectrogram(spectrogram_complex)

        matcher.add_reference(species_id, log_spectrogram)
        logger.info("Added reference '%s' from %s", species_id, path.name)

    matcher.save(str(OUTPUT_PATH))
    logger.info("Saved reference library to %s", OUTPUT_PATH)
    logger.info("Species in library: %s", sorted(matcher.reference_hash_counts.keys()))
    for species, count in matcher.reference_hash_counts.items():
        if count == 0:
            logger.warning(
                "'%s' produced ZERO fingerprint hashes -- likely too quiet/short "
                "for the current extract_keypoints settings.", species
            )


if __name__ == "__main__":
    build()
