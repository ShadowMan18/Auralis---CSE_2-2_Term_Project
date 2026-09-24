"""
Stage 5 — STFT / Spectrogram (generic version)

Parameterized by window type, window length, hop length, and FFT size -
the actual trade-off knobs discussed earlier (spectral leakage vs.
time/frequency resolution), rather than one fixed choice baked in.
This is the core "signals" deliverable: slice the signal into
overlapping windows, apply a window function to each (to reduce
spectral leakage from treating a finite chunk as if it repeated
infinitely), then FFT each windowed frame and stack the results into a
2D time-frequency array.
"""

import numpy as np


def stft(signal, sample_rate, window_length_sec, hop_length_sec,
         window_type="hann", fft_size=None):
    """
    Compute the Short-Time Fourier Transform of a 1D signal.

    window_length_sec : length of each analysis frame, in seconds.
        Longer = better frequency resolution, worse time resolution
        (you're averaging over a bigger time chunk).
    hop_length_sec : how far the window advances between frames, in
        seconds. Smaller = more overlap = smoother look across time,
        more frames to compute. hop < window_length means overlap;
        hop == window_length means no overlap.
    window_type : "hann", "hamming", "blackman", or "rectangular".
        Anything other than rectangular tapers the frame's edges to
        near zero before the FFT - a rectangular window's abrupt edges
        cause spectral leakage (energy smearing into neighboring
        frequency bins) because the FFT implicitly treats the frame as
        one period of an infinitely repeating signal, and an abrupt
        cut mid-waveform creates a discontinuity at the seam.
    fft_size : number of FFT points per frame. Defaults to
        window_length (samples). Larger than the window length
        zero-pads each frame first, which interpolates extra points
        into the frequency axis (smoother-looking spectrum) WITHOUT
        actually improving true frequency resolution - that's still
        set by window_length_sec alone.

    Returns
    -------
    spectrogram : complex ndarray, shape (fft_size // 2 + 1, num_frames)
    freqs : ndarray, the frequency (Hz) each row corresponds to
    times : ndarray, the time (seconds) each column's frame starts at
    """
    window_length = int(round(window_length_sec * sample_rate))
    hop_length = int(round(hop_length_sec * sample_rate))
    if fft_size is None:
        fft_size = window_length

    window = _make_window(window_type, window_length)

    if len(signal) < window_length:
        signal = np.pad(signal, (0, window_length - len(signal)))

    num_frames = 1 + (len(signal) - window_length) // hop_length

    num_freq_bins = fft_size // 2 + 1
    spectrogram = np.empty((num_freq_bins, num_frames), dtype=complex)

    for i in range(num_frames):
        start = i * hop_length
        frame = signal[start: start + window_length]
        windowed_frame = frame * window
        spectrogram[:, i] = np.fft.rfft(windowed_frame, n=fft_size)

    freqs = np.fft.rfftfreq(fft_size, d=1 / sample_rate)
    times = np.arange(num_frames) * hop_length / sample_rate
    return spectrogram, freqs, times


def log_magnitude_spectrogram(spectrogram, floor_db=-80.0):
    """
    Convert a complex STFT output to a log-magnitude spectrogram in dB -
    the representation you'd actually feed to feature extraction or a
    CNN, since raw linear magnitude is dominated by a few loud bins and
    hides quieter structure. floor_db clips silence/near-zero bins so
    log(0) doesn't produce -inf.
    """
    magnitude = np.abs(spectrogram)
    magnitude = np.maximum(magnitude, 1e-10)  # avoid log(0)
    db = 20 * np.log10(magnitude)
    return np.maximum(db, floor_db)


def build_log_freq_filterbank(freqs, num_bins=128, fmin=50.0, fmax=None):
    """
    Triangular filterbank mapping linear FFT bins onto log2-spaced center
    frequencies (mel-like, but plain log2 rather than the mel formula --
    simpler, and all that's actually needed here).

    Why this matters for matching: on a LINEAR frequency axis, a pitch
    shift multiplies every harmonic's frequency by a constant factor, so
    it lands on a different, unpredictable bin depending on the harmonic's
    absolute frequency. On a LOG frequency axis, that same multiplicative
    shift becomes a constant ADDITIVE bin shift -- every harmonic moves by
    the same number of bins. That's what makes a relative-frequency hash
    (AudioMatcher's pitch_invariant mode, which hashes freq2 - freq1
    instead of the absolute bins) actually pitch-shift invariant: the
    difference between two bins on this axis is unchanged by a constant
    shift, even though the absolute bins both moved.
    """
    if fmax is None:
        fmax = freqs[-1]
    log_fmin = np.log2(max(fmin, 1e-6))
    log_fmax = np.log2(fmax)
    centers_log = np.linspace(log_fmin, log_fmax, num_bins + 2)
    centers_hz = 2.0 ** centers_log

    filterbank = np.zeros((num_bins, len(freqs)), dtype=np.float64)
    for i in range(num_bins):
        f_lo, f_center, f_hi = centers_hz[i], centers_hz[i + 1], centers_hz[i + 2]
        left = (freqs - f_lo) / max(f_center - f_lo, 1e-9)
        right = (f_hi - freqs) / max(f_hi - f_center, 1e-9)
        tri = np.clip(np.minimum(left, right), 0.0, None)
        filterbank[i] = tri

    return filterbank, centers_hz[1:-1]


def to_log_frequency_spectrogram(spectrogram, freqs, num_bins=128, fmin=50.0,
                                  fmax=None, floor_db=-80.0):
    """
    Resample a complex (or linear-magnitude) STFT onto log2-spaced
    frequency bins, then convert to log-magnitude (dB).

    Use this INSTEAD OF log_magnitude_spectrogram() wherever the result
    feeds AudioMatcher.extract_keypoints in pitch_invariant mode -- see
    build_log_freq_filterbank's docstring for why. Both the reference
    library build and the query-time path must use the same num_bins /
    fmin / fmax (matching the existing "must be IDENTICAL on both sides"
    convention already used for REFERENCE_SPEC / STFT_PARAMS elsewhere in
    this project) or bin indices won't line up between them.

    Returns
    -------
    log_spectrogram : ndarray, shape (num_bins, n_time), dB
    log_freqs : ndarray, shape (num_bins,) -- the center frequency (Hz)
        of each output row, for reference/debugging.
    """
    magnitude = np.abs(spectrogram)
    filterbank, log_freqs = build_log_freq_filterbank(freqs, num_bins, fmin, fmax)
    log_magnitude = filterbank @ magnitude  # (num_bins, n_time)
    log_magnitude = np.maximum(log_magnitude, 1e-10)
    db = 20 * np.log10(log_magnitude)
    return np.maximum(db, floor_db), log_freqs


def _make_window(window_type, length):
    if window_type == "hann":
        return np.hanning(length)
    elif window_type == "hamming":
        return np.hamming(length)
    elif window_type == "blackman":
        return np.blackman(length)
    elif window_type == "rectangular":
        return np.ones(length)
    else:
        raise ValueError(
            f"unknown window_type '{window_type}' - "
            "use 'hann', 'hamming', 'blackman', or 'rectangular'"
        )


if __name__ == "__main__":
    # Sanity check: a 1000 Hz tone should show up as a clear peak at
    # bin frequency ~1000 Hz in every frame, regardless of window choice.
    sample_rate = 22050
    duration = 2.0
    t = np.arange(int(sample_rate * duration)) / sample_rate
    tone = np.sin(2 * np.pi * 1000 * t).astype(np.float32)

    for window_type in ["rectangular", "hann", "hamming", "blackman"]:
        spectrogram, freqs, times = stft(
            tone, sample_rate,
            window_length_sec=0.05, hop_length_sec=0.025,
            window_type=window_type,
        )
        magnitude = np.abs(spectrogram)
        # Check the middle frame (avoids edge-of-signal artifacts)
        mid_frame = magnitude[:, magnitude.shape[1] // 2]
        peak_freq = freqs[np.argmax(mid_frame)]
        print(f"{window_type:12s}: spectrogram shape {spectrogram.shape}, "
              f"peak at {peak_freq:.1f} Hz (expected 1000 Hz)")