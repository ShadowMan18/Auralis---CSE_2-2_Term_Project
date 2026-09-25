"""
pipeline_config.py

Single source of truth for every parameter that MUST be identical
between the reference-library build step and the query-time detection
step. Previously these were duplicated (and had already drifted slightly)
across build_reference_library.py and processor/services/processor.py
(test.py in this zip) -- a mismatch there doesn't raise an error, it just
silently produces garbage scores, because the whole hash scheme depends
on both sides mapping (frequency, time) to bin indices the same way.

Import from here in both places instead of redefining these constants.
"""

try:
    from .audio_processing import ConsumerSpec
except ImportError:
    from audio_processing import ConsumerSpec

# --- Preprocessing (must match on both sides) -------------------------------

REFERENCE_SPEC = ConsumerSpec(
    target_sample_rate=22050,
    target_channels=1,
    target_duration=None,  # whole clip, no windowing -- one reference call = one unit
)

# Smaller hop than a windowed original (1.5s) would help avoid a call
# being split across window boundaries -- but the actual deployed
# processor.py uses whole-clip processing (target_duration=None,
# hop_duration=None: the whole query is one unsegmented window), which
# sidesteps the boundary-split problem entirely by not windowing at all.
# Kept matching that here. Revisit if/when you're feeding this long,
# continuous field recordings, where whole-clip STFT stops being cheap
# and windowing becomes necessary again.
QUERY_SPEC = ConsumerSpec(
    target_sample_rate=22050,
    target_channels=1,
    target_duration=None,
    hop_duration=None,
    amplitude_range=(-1.0, 1.0),
)

BANDPASS_PARAMS = dict(low_cutoff_hz=50.0, high_cutoff_hz=10000.0, transition_bandwidth_hz=200.0)
STFT_PARAMS = dict(window_length_sec=0.046, hop_length_sec=0.012, window_type="hann")

# Log-frequency spectrogram (see stft.to_log_frequency_spectrogram): this
# is what gives the pitch_invariant hash mode below something to work
# with. fmin/fmax intentionally match BANDPASS_PARAMS' cutoffs -- no
# point spending log bins on a band the bandpass filter already removed.
LOG_FREQ_PARAMS = dict(
    num_bins=128,
    fmin=BANDPASS_PARAMS["low_cutoff_hz"],
    fmax=BANDPASS_PARAMS["high_cutoff_hz"],
)

# --- Keypoint extraction (must match on both sides) -------------------------

EXTRACT_PARAMS = dict(
    neighborhood_size=(15, 15),
    amplitude_floor_db=-40.0,
    num_zones_freq=4,
    num_zones_time=8,
    peaks_per_zone=5,
    adaptive_floor=True,        # per-row noise floor -- see AudioMatcher docstring
    noise_floor_margin_db=15.0,
)

# --- Fingerprint / hashing (must match on both sides -- these are the
# AudioMatcher fingerprint_kwargs, enforced consistent by its own __init__
# contract) -----------------------------------------------------------------

MATCHER_PARAMS = dict(
    fan_out=5,
    min_time_delta=1,
    max_time_delta=100,
    # Keep absolute log-frequency bins in the hash. Relative-only hashes
    # collide across unrelated animal calls that share similar intervals.
    # The reference augmentation grid below supplies limited pitch tolerance.
    pitch_invariant=False,
    time_bucket=2,          # coarser dt -- some tolerance to mild time-stretch
)

# --- Reference-library augmentation (build side only) -----------------------

# Small grid of (semitone shift, tempo rate) variants added to the
# library for EACH reference clip, all merged under the same species_id.
# Because matching is "does this exact hash reappear anywhere in the
# index", giving the index several plausible variants of a call directly
# raises the odds of a real hit for a query that happens to be shifted in
# a similar way -- independent of, and complementary to, the
# pitch_invariant/time_bucket hashing tolerance above. Keep this grid
# small: it multiplies build time and index size per reference.
PITCH_AUGMENT_STEPS = (-2.0, 0.0, 2.0)   # semitones
TEMPO_AUGMENT_RATES = (1.0, 1.15)        # rate multiplier (1.0 = unchanged)

# ref_id for each augmented variant is built as f"{species_id}{VARIANT_SEPARATOR}{label}"
# so AudioMatcher.detect(..., variant_separator=VARIANT_SEPARATOR) can collapse
# them back down to one score per species -- see collapse_variant_scores'
# docstring for why this has to happen AFTER scoring, not by merging
# variants under one ref_id at build time.
VARIANT_SEPARATOR = "::"

# Keep the original low match-count floor so pitch-shifted and time-stretched
# variants are not discarded before their reference augmentations can vote.
MIN_RAW_MATCH_COUNT = 5

# --- Detection threshold ------------------------------------------------

# The deployed processor.py currently has this at 0.0 (report everything,
# no filtering) after moving to whole-clip processing removed some of the
# windowing-induced score degradation. With pitch_invariant hashing +
# multi-variant references now in the mix, the score distribution has
# shifted again (see the actual before/after numbers we measured) --
# re-validate before raising this, rather than trusting either number.
CONFIDENCE_THRESHOLD = 0.0
