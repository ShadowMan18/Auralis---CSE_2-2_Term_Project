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