"""
Stage 1 — Ingestion & Normalization (class-based version)

Same behavior as the functional version, reorganized into classes:

- ConsumerSpec: a plain data holder for Step 1's four facts
  (target_sample_rate, target_channels, target_duration, hop_duration,
  amplitude_range) for one consumer (your own DSP pipeline, BirdNET, Perch...).
- AudioDecoder / Resampler / ChannelMixer / Segmenter / AmplitudeScaler:
  one class per pipeline stage, each stateless (just grouping the
  transform + its helpers together).
- AudioPreprocessor: the coordinator. Owns one instance of each stage
  and runs a file through all of them for a given ConsumerSpec. Stages
  are injectable in __init__, mainly so tests can swap in a fake
  Resampler etc. without touching the others.

Uses soundfile/librosa purely for file I/O and resampling (decoding
compressed containers and resampling are not the DSP content this
project is being graded on). To hand-roll resampling for course credit,
swap out Resampler only — everything else stays the same.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

from Auralis.backend.processor.audio_processing.resample import Resample


@dataclass(frozen=True)
class ConsumerSpec:
    """Step 1's four facts for one consumer. Look these up in the
    consumer's own docs before constructing one — don't guess them."""

    target_sample_rate: int
    target_channels: int = 1
    target_duration: Optional[float] = None
    hop_duration: Optional[float] = None
    amplitude_range: Tuple[float, float] = (-1.0, 1.0)


class AudioDecoder:
    """Bytes on disk -> float32 (channels, frames) array + native rate."""

    def decode(self, file_path: str):
        try:
            samples, native_rate = sf.read(file_path, always_2d=True)
            samples = samples.T  # (frames, channels) -> (channels, frames)
        except Exception:
            # Compressed / less common containers (mp3, m4a, 3gp, ...)
            samples, native_rate = librosa.load(file_path, sr=None, mono=False)
            if samples.ndim == 1:
                samples = samples[np.newaxis, :]
        return samples.astype(np.float32), native_rate


class Resampler:
    """Swap this class out if you want the resampling step hand-rolled
    for course credit; everything else is unaffected."""

    def resample(self, samples, native_rate: int, target_rate: int):
        if native_rate == target_rate:
            return samples
        resampler=Resample()
        return resampler.resample_hand_rolled(samples, native_rate, target_rate)


class ChannelMixer:
    def downmix(self, samples, target_channels: int):
        n_channels = samples.shape[0]
        if target_channels == 1 and n_channels > 1:
            samples = samples.mean(axis=0, keepdims=True)
        elif target_channels > 1 and n_channels == 1:
            samples = np.repeat(samples, target_channels, axis=0)
        return samples[0] if target_channels == 1 else samples


class Segmenter:
    def segment_or_pad(
        self,
        samples,
        sample_rate: int,
        target_duration: Optional[float],
        hop_duration: Optional[float],
    ):
        if target_duration is None:
            return [samples]

        target_len = int(round(target_duration * sample_rate))
        hop_len = (
            int(round(hop_duration * sample_rate))
            if hop_duration is not None
            else target_len // 2
        )

        if len(samples) <= target_len:
            pad_width = target_len - len(samples)
            return [np.pad(samples, (0, pad_width))]

        windows = []
        start = 0
        while start + target_len <= len(samples):
            windows.append(samples[start:start + target_len])
            start += hop_len
        if start < len(samples):
            tail = samples[start:]
            pad_width = target_len - len(tail)
            windows.append(np.pad(tail, (0, pad_width)))
        return windows


class AmplitudeScaler:
    def scale(self, window, amplitude_range: Tuple[float, float]):
        lo, hi = amplitude_range
        peak = np.max(np.abs(window)) if len(window) else 0.0
        normalized = window / peak if peak > 0 else window  # -> [-1, 1]
        scaled = lo + (normalized + 1.0) * (hi - lo) / 2.0    # [-1, 1] -> [lo, hi]
        return scaled.astype(np.float32)


class AudioPreprocessor:
    """
    Coordinator. Run the same raw file through this multiple times with
    different ConsumerSpecs — the "two separate forks from one source
    recording" idea, made concrete.
    """

    def __init__(
        self,
        decoder: Optional[AudioDecoder] = None,
        resampler: Optional[Resampler] = None,
        mixer: Optional[ChannelMixer] = None,
        segmenter: Optional[Segmenter] = None,
        scaler: Optional[AmplitudeScaler] = None,
    ):
        self._decoder = decoder or AudioDecoder()
        self._resampler = resampler or Resampler()
        self._mixer = mixer or ChannelMixer()
        self._segmenter = segmenter or Segmenter()
        self._scaler = scaler or AmplitudeScaler()

    def process(self, file_path: str, spec: ConsumerSpec) -> List[np.ndarray]:
        """
        Returns one float32 array per window, each exactly
        round(spec.target_duration * spec.target_sample_rate) samples
        long, scaled into spec.amplitude_range. If spec.target_duration
        is None, a single un-segmented array is returned as a one-item
        list.
        """
        samples, native_rate = self._decoder.decode(file_path)
        samples = self._resampler.resample(samples, native_rate, spec.target_sample_rate)
        samples = self._mixer.downmix(samples, spec.target_channels)
        windows = self._segmenter.segment_or_pad(
            samples, spec.target_sample_rate, spec.target_duration, spec.hop_duration
        )
        return [self._scaler.scale(w, spec.amplitude_range) for w in windows]


if __name__ == "__main__":
    # One raw recording, three different forks — same AudioPreprocessor,
    # different ConsumerSpec per consumer.

    preprocessor = AudioPreprocessor()

    own_pipeline_spec = ConsumerSpec(
        target_sample_rate=22050,
        target_channels=1,
        target_duration=3.0,
        hop_duration=1.5,
        amplitude_range=(-1.0, 1.0),
    )

    birdnet_spec = ConsumerSpec(
        target_sample_rate=48000,
        target_channels=1,
        target_duration=3.0,
        hop_duration=1.5,
        amplitude_range=(-1.0, 1.0),  # confirmed in BirdNET's own docs
    )

    perch_spec = ConsumerSpec(
        target_sample_rate=32000,
        target_channels=1,
        target_duration=5.0,
        hop_duration=2.5,
        amplitude_range=(-1.0, 1.0),  # NOT independently confirmed — verify before trusting
    )

    audio_file = Path(__file__).parent / "rec.ogg"

    own_pipeline_windows = preprocessor.process(
        str(audio_file),
        own_pipeline_spec
    )

    birdnet_windows = preprocessor.process(
        str(audio_file),
        birdnet_spec
    )

    perch_windows = preprocessor.process(
        str(audio_file),
        perch_spec
    )

    print("Processing successful!")
    print("Number of windows:", len(own_pipeline_windows))
    print("First window shape:", own_pipeline_windows[0].shape)
    print("Data type:", own_pipeline_windows[0].dtype)