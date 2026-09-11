"""
End-to-end verification of the DSP detection pipeline.

Pipeline under test:
    raw audio file
      -> AudioPreprocessor (decode / resample / downmix / normalize)   [preprocess_audio.py]
      -> stft() + log_magnitude_spectrogram()                         [stft.py]
      -> AudioMatcher.add_reference() / AudioMatcher.detect()         [audio_matcher.py]

Directory layout assumed (adjust SAMPLES_DIR below if different):

    audio_processing/
        preprocess_audio.py
        resample.py
        stft.py
        audio_matcher.py
        samples/
            sample1.mp3
            sample2.wav
            ...
            references/
                cat.mp3
                dog.wav
                ...

Both .mp3 and .wav are supported for references and samples --
AudioDecoder already handles either via soundfile/librosa, this file just
needed to *discover* both extensions instead of only *.mp3.

Run it:
    pytest -v -s test_species_detection.py

    (the -s is important -- without it you won't see the printed
    detection results, only pass/fail)

If pytest isn't installed, just run it as a plain script:
    python test_species_detection.py
"""

import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup -- adjust these two lines if your project layout differs.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent          # folder containing the .py modules
SAMPLES_DIR = PROJECT_ROOT / "samples"
REFERENCES_DIR = SAMPLES_DIR / "references"

sys.path.insert(0, str(PROJECT_ROOT))

from Auralis.backend.processor.audio_processing.audio_processing import AudioPreprocessor, ConsumerSpec   # noqa: E402
from Auralis.backend.processor.audio_processing.stft import stft, log_magnitude_spectrogram                # noqa: E402
from Auralis.backend.processor.audio_processing.audio_matcher import AudioMatcher                           # noqa: E402


# ---------------------------------------------------------------------------
# Shared DSP configuration.
#
# CRITICAL: these values must be IDENTICAL for every reference and every
# query. AudioMatcher hashes are (freq_bin_index, freq_bin_index, time_delta)
# -- if the sample rate or STFT window/hop/FFT size differs between how a
# reference was built and how a query is built, the bin indices stop
# corresponding to the same physical frequencies/times and matching breaks
# silently (you'll just get empty or garbage results, no error).
# ---------------------------------------------------------------------------
TARGET_SAMPLE_RATE = 22050
STFT_WINDOW_SEC = 0.046      # ~1024 samples at 22050 Hz
STFT_HOP_SEC = 0.012         # ~256 samples at 22050 Hz
STFT_WINDOW_TYPE = "hann"
STFT_FFT_SIZE = None         # defaults to window length

FINGERPRINT_KWARGS = dict(fan_out=5, min_time_delta=1, max_time_delta=100)
EXTRACT_KWARGS = dict(
    neighborhood_size=(15, 15),
    amplitude_floor_db=-40.0,
    num_zones_freq=4,
    num_zones_time=8,
    peaks_per_zone=5,
)

# Detection thresholds -- tune once you've looked at real score distributions.
MIN_RAW_COUNT = 1
SCORE_THRESHOLD = 0.0
TOP_K = 5

# Extensions that count as audio for both references and samples.
AUDIO_EXTENSIONS = (".mp3", ".wav")

# ---------------------------------------------------------------------------
# Ground truth. Fill this in once you know what's actually mixed into each
# sample*.{mp3,wav} -- keys are the file stem ("sample1"), values are either:
#   - a single species name (str)            -> top match must equal it
#   - a set/list of species names             -> top match must be one of them,
#                                                OR any of them must appear
#                                                somewhere in the results
# Leave a sample out of this dict to skip strict assertion and just print
# what was detected (useful while you're still labeling samples).
#
# generate_test_samples.py prints/writes a ready-to-paste version of this
# dict (and samples/ground_truth.json) after it builds synthetic samples.
# ---------------------------------------------------------------------------
EXPECTED = {
    "sample1": {"cat", "cow", "crow"},
    "sample2": {"dog", "goat", "horse", "monkey"},
    "sample3": {"cow", "crow", "goat", "rooster"},
}


def spectrogram_for(path: Path, preprocessor: AudioPreprocessor, spec: ConsumerSpec) -> np.ndarray:
    """Run one audio file through preprocessing + STFT -> log-magnitude spectrogram."""
    with open(path, "rb") as file_obj:
        windows = preprocessor.process(file_obj, spec)
    # spec.target_duration is None -> segment_or_pad returns exactly one
    # un-padded, whole-clip window.
    samples = windows[0]

    spectrogram, _freqs, _times = stft(
        samples,
        spec.target_sample_rate,
        window_length_sec=STFT_WINDOW_SEC,
        hop_length_sec=STFT_HOP_SEC,
        window_type=STFT_WINDOW_TYPE,
        fft_size=STFT_FFT_SIZE,
    )
    return log_magnitude_spectrogram(spectrogram)

def _discover_audio(directory: Path, exclude_dirs=False):
    """All files under `directory` (non-recursive) whose extension is in
    AUDIO_EXTENSIONS, case-insensitive, sorted by name."""
    if not directory.is_dir():
        return []
    files = [
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]
    return sorted(files)


def discover_reference_files():
    return _discover_audio(REFERENCES_DIR)


def discover_sample_files():
    # Only files directly under SAMPLES_DIR -- references/ is a subfolder
    # and its contents don't count as samples.
    return [p for p in _discover_audio(SAMPLES_DIR) if p.parent == SAMPLES_DIR]


def build_matcher(preprocessor: AudioPreprocessor, spec: ConsumerSpec, reference_files):
    matcher = AudioMatcher(**FINGERPRINT_KWARGS)
    for ref_path in reference_files:
        species = ref_path.stem  # "cat.mp3" -> "cat", "cat.wav" -> "cat"
        log_spec = spectrogram_for(ref_path, preprocessor, spec)
        matcher.add_reference(species, log_spec, **EXTRACT_KWARGS)
    return matcher


def format_results(results):
    lines = []
    for species, info in results:
        lines.append(
            f"  {species:<15s} score={info['score']:.4f}  "
            f"raw_count={info['raw_count']:4d}  offset={info['offset']}"
        )
    return "\n".join(lines) if lines else "  (no matches above threshold)"


# ---------------------------------------------------------------------------
# pytest section
# ---------------------------------------------------------------------------
try:
    import pytest

    @pytest.fixture(scope="session")
    def preprocessor():
        return AudioPreprocessor()

    @pytest.fixture(scope="session")
    def consumer_spec():
        return ConsumerSpec(
            target_sample_rate=TARGET_SAMPLE_RATE,
            target_channels=1,
            target_duration=None,   # whole clip, no windowing -- fingerprinting
            hop_duration=None,      # needs the full spectrogram, not fixed chunks
            amplitude_range=(-1.0, 1.0),
        )

    @pytest.fixture(scope="session")
    def reference_files():
        files = discover_reference_files()
        if not files:
            pytest.skip(f"No reference audio found in {REFERENCES_DIR}")
        return files

    @pytest.fixture(scope="session")
    def matcher(preprocessor, consumer_spec, reference_files):
        return build_matcher(preprocessor, consumer_spec, reference_files)

    def _sample_ids(path):
        # pytest probes this with its own NOTSET sentinel during collection
        # when the parametrize list is empty -- must not blow up on that.
        return path.stem if isinstance(path, Path) else "no-samples-found"

    _sample_files = discover_sample_files()

    @pytest.mark.parametrize(
        "sample_path",
        _sample_files if _sample_files else [None],
        ids=_sample_ids,
    )
    def test_species_detected(sample_path, preprocessor, consumer_spec, matcher):
        if sample_path is None:
            pytest.skip(
                f"No sample*.mp3/.wav files found directly under {SAMPLES_DIR} "
                f"(files inside references/ don't count -- check filenames/extensions)"
            )

        log_spec = spectrogram_for(sample_path, preprocessor, consumer_spec)
        results = matcher.detect(
            log_spec,
            min_raw_count=MIN_RAW_COUNT,
            threshold=SCORE_THRESHOLD,
            top_k=TOP_K,
            **EXTRACT_KWARGS,
        )

        print(f"\n{sample_path.name} detections:\n{format_results(results)}")

        assert results, (
            f"No species detected in {sample_path.name} -- check STFT params, "
            f"EXTRACT_KWARGS, or that the reference library actually loaded."
        )

        top_species, top_info = results[0]
        expected = EXPECTED.get(sample_path.stem)

        if expected is None:
            pytest.skip(
                f"No ground truth recorded for '{sample_path.stem}' yet -- "
                f"top match was '{top_species}' (score={top_info['score']:.4f}). "
                f"Add it to EXPECTED once you've confirmed it by ear."
            )
        elif isinstance(expected, (set, list, tuple)):
            detected_species = {s for s, _ in results}
            assert detected_species & set(expected), (
                f"expected one of {expected} in results, got: "
                f"{[s for s, _ in results]}"
            )
        else:
            assert top_species == expected, (
                f"expected top match '{expected}', got '{top_species}' "
                f"(score={top_info['score']:.4f})\nfull results:\n{format_results(results)}"
            )

    def test_reference_library_not_empty(matcher, reference_files):
        assert matcher.reference_hash_counts, "reference index is empty after loading references"
        for ref_path in reference_files:
            species = ref_path.stem
            assert matcher.reference_hash_counts.get(species, 0) > 0, (
                f"'{species}' produced zero fingerprint hashes -- likely too short/quiet, "
                f"or extract_keypoints/EXTRACT_KWARGS need tuning for this clip"
            )

    _HAVE_PYTEST = True

except ImportError:
    _HAVE_PYTEST = False


# ---------------------------------------------------------------------------
# Plain-script fallback (no pytest required)
# ---------------------------------------------------------------------------
def run_plain():
    preprocessor = AudioPreprocessor()
    spec = ConsumerSpec(
        target_sample_rate=TARGET_SAMPLE_RATE,
        target_channels=1,
        target_duration=None,
        hop_duration=None,
        amplitude_range=(-1.0, 1.0),
    )

    reference_files = discover_reference_files()
    if not reference_files:
        print(f"No reference audio found in {REFERENCES_DIR}")
        return 1

    print(f"Building reference library from {len(reference_files)} file(s)...")
    matcher = build_matcher(preprocessor, spec, reference_files)
    for species, count in matcher.reference_hash_counts.items():
        print(f"  {species:<15s} {count} hashes")

    sample_files = discover_sample_files()
    if not sample_files:
        print(f"No sample*.mp3/.wav files found in {SAMPLES_DIR}")
        return 1

    exit_code = 0
    for sample_path in sample_files:
        log_spec = spectrogram_for(sample_path, preprocessor, spec)
        results = matcher.detect(
            log_spec,
            min_raw_count=MIN_RAW_COUNT,
            threshold=SCORE_THRESHOLD,
            top_k=TOP_K,
            **EXTRACT_KWARGS,
        )
        print(f"\n{sample_path.name} detections:")
        print(format_results(results))

        if not results:
            exit_code = 1
            continue

        top_species, top_info = results[0]
        expected = EXPECTED.get(sample_path.stem)
        if expected is None:
            print(f"  (no ground truth recorded for '{sample_path.stem}' yet)")
        else:
            ok = (top_species in expected) if isinstance(expected, (set, list, tuple)) else (top_species == expected)
            status = "OK" if ok else "MISMATCH"
            print(f"  expected={expected}  top={top_species}  -> {status}")
            if not ok:
                exit_code = 1

    return exit_code


if __name__ == "__main__":
    if _HAVE_PYTEST:
        import pytest as _pytest
        raise SystemExit(_pytest.main([__file__, "-v", "-s"]))
    else:
        print("pytest not found -- running as a plain script instead.\n")
        raise SystemExit(run_plain())