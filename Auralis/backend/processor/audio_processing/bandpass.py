"""
Stage 4 — Bandpass Filtering (generic version)

Same design philosophy as resample_hand_rolled.py: parameterized by
WHAT you want (cutoff frequencies, transition sharpness), not hardcoded
for one specific signal. A bandpass filter is built here as the
DIFFERENCE of two low-pass filters (one passing everything below the
high cutoff, one passing everything below the low cutoff) - subtracting
them leaves only the band in between. Applied via the same FFT-based
convolution used for resampling.
"""

import numpy as np
from Auralis.backend.processor.audio_processing.resample import Resample


def design_bandpass_fir(low_cutoff_hz, high_cutoff_hz, sample_rate,
                         transition_bandwidth_hz=200.0, window="hamming"):
    """
    Design a bandpass FIR filter passing [low_cutoff_hz, high_cutoff_hz].

    transition_bandwidth_hz : how sharp the cutoff edges are, in Hz -
        smaller means a sharper (more abrupt) filter edge, at the cost
        of needing more taps. This uses the standard Hamming-window
        approximation: transition_width (Hz) ~= 3.3 * sample_rate / num_taps,
        solved for num_taps. Exposing this instead of a raw tap count
        keeps the parameter meaningful in the units you actually care
        about (Hz), rather than an arbitrary filter-length number.
    """
    num_taps = int(np.ceil(3.3 * sample_rate / transition_bandwidth_hz))
    if num_taps % 2 == 0:
        num_taps += 1
    resampler=Resample()
    lowpass_at_high_edge = resampler.design_lowpass_fir(high_cutoff_hz, sample_rate, num_taps, window)
    lowpass_at_low_edge = resampler.design_lowpass_fir(low_cutoff_hz, sample_rate, num_taps, window)

    # Passes everything below high_cutoff, MINUS everything below
    # low_cutoff, leaves only the band in between.
    bandpass = lowpass_at_high_edge - lowpass_at_low_edge
    return bandpass


def apply_bandpass(samples, sample_rate, low_cutoff_hz, high_cutoff_hz,
                    transition_bandwidth_hz=200.0, window="hamming"):
    """
    Filter `samples` (1D, mono - run per-channel yourself if needed)
    to pass only [low_cutoff_hz, high_cutoff_hz], via convolution.
    """
    fir = design_bandpass_fir(
        low_cutoff_hz, high_cutoff_hz, sample_rate, transition_bandwidth_hz, window
    )
    resampler=Resample()
    filtered = resampler.convolve_via_fft(samples, fir)

    # Trim the convolution's edge padding back to align with the input
    # (same alignment fix used in resample_hand_rolled.py).
    edge = (len(fir) - 1) // 2
    filtered = filtered[edge: edge + len(samples)]
    return filtered.astype(np.float32)


if __name__ == "__main__":
    # Sanity check: three tones - below, inside, and above a test band -
    # only the in-band one should survive with close to full amplitude.
    sample_rate = 22050
    duration = 2.0
    t = np.arange(int(sample_rate * duration)) / sample_rate

    below_band = np.sin(2 * np.pi * 200 * t)    # should be attenuated
    in_band = np.sin(2 * np.pi * 2000 * t)      # should pass through
    above_band = np.sin(2 * np.pi * 9000 * t)   # should be attenuated

    mixed = (below_band + in_band + above_band).astype(np.float32)

    filtered = apply_bandpass(
        mixed, sample_rate, low_cutoff_hz=1000, high_cutoff_hz=3000,
        transition_bandwidth_hz=200.0,
    )

    # Check via FFT which frequencies actually survived.
    spectrum = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(filtered), d=1 / sample_rate)

    def energy_near(target_hz, tolerance_hz=50):
        mask = np.abs(freqs - target_hz) < tolerance_hz
        return spectrum[mask].max()

    print(f"energy at 200 Hz  (below band, should be small): {energy_near(200):.2f}")
    print(f"energy at 2000 Hz (in band, should be large):     {energy_near(2000):.2f}")
    print(f"energy at 9000 Hz (above band, should be small):  {energy_near(9000):.2f}")