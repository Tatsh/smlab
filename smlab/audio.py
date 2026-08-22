"""
Audio loading and onset-envelope extraction.

Onset envelopes are computed from explicit parameter sets rather than one default, because the three
jobs they serve pull in opposite directions. Tempo estimation wants a long, smooth window; locating
the beat phase wants a short window that localises transients sharply; separating beats from
off-beats wants frequency resolution in the bass band.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
import logging

import librosa
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

__all__ = (
    'DEFAULT_HOP_LENGTH',
    'DEFAULT_N_FFT',
    'DEFAULT_N_MELS',
    'DEFAULT_SAMPLE_RATE',
    'OnsetParams',
    'audio_duration',
    'envelope_rate',
    'load_audio',
    'onset_envelope',
)

log = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 22050
"""
Sample rate used for timing analysis.

Percussive onsets are well resolved here and it halves the decode cost relative to full-rate audio.

:meta hide-value:
"""
DEFAULT_HOP_LENGTH = 128
"""
Hop length for onset envelopes, giving about 5.8 ms per frame at :data:`DEFAULT_SAMPLE_RATE`.

:meta hide-value:
"""
DEFAULT_N_FFT = 1024
"""Analysis window length in samples.

:meta hide-value:
"""
DEFAULT_N_MELS = 64
"""
Mel bands the onset detector aggregates over.

librosa's default of 128 leaves many filters empty at short window lengths, which silently degrades
the envelope, so a lower count is used throughout.

:meta hide-value:
"""


@dataclass(frozen=True, slots=True)
class OnsetParams:
    """Settings controlling how an onset envelope is computed."""

    fmax: float | None = None
    """Upper frequency bound, or ``None`` for the full band."""
    hop_length: int = DEFAULT_HOP_LENGTH
    """Frames advance by this many samples."""
    n_fft: int = DEFAULT_N_FFT
    """Analysis window length in samples."""
    n_mels: int = DEFAULT_N_MELS
    """Number of mel bands to aggregate over."""
    sample_rate: int = DEFAULT_SAMPLE_RATE
    """Sample rate the audio is decoded at."""

    @property
    def frame_rate(self) -> float:
        """Envelope frames per second."""
        return self.sample_rate / self.hop_length


def load_audio(path: Path, *, sample_rate: int = DEFAULT_SAMPLE_RATE) -> NDArray[np.float32]:
    """
    Load an audio file as mono samples.

    Parameters
    ----------
    path : :py:class:`~pathlib.Path`
        The audio file to read.
    sample_rate : int
        Target sample rate.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        Mono samples as 32-bit floats.
    """
    samples, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return np.asarray(samples, dtype=np.float32)


def audio_duration(path: Path, *, sample_rate: int = DEFAULT_SAMPLE_RATE) -> float:
    """
    Measure how long an audio file runs, by decoding it.

    The duration a container advertises can be some way off what actually decodes, so the samples
    are counted instead. StepMania divides the groove radar rates by this figure.

    Parameters
    ----------
    path : :py:class:`~pathlib.Path`
        The audio file to measure.
    sample_rate : int
        Target sample rate.

    Returns
    -------
    float
        Length in seconds.
    """
    return len(load_audio(path, sample_rate=sample_rate)) / sample_rate


def envelope_rate(
    sample_rate: int = DEFAULT_SAMPLE_RATE, hop_length: int = DEFAULT_HOP_LENGTH
) -> float:
    """
    Return the frame rate of an onset envelope.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate.
    hop_length : int
        Hop length used to compute the envelope.

    Returns
    -------
    float
        Envelope frames per second.
    """
    return sample_rate / hop_length


def onset_envelope(
    samples: NDArray[np.float32], params: OnsetParams | None = None
) -> NDArray[np.float32]:
    """
    Compute a spectral-flux onset strength envelope.

    Parameters
    ----------
    samples : :py:class:`~numpy.ndarray`
        Mono audio samples.
    params : OnsetParams | None
        Settings to use, or ``None`` for the defaults.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        The onset strength envelope, one value per frame.
    """
    settings = params if params is not None else OnsetParams()
    envelope = librosa.onset.onset_strength(
        y=samples,
        sr=settings.sample_rate,
        hop_length=settings.hop_length,
        n_fft=settings.n_fft,
        n_mels=settings.n_mels,
        fmax=settings.fmax if settings.fmax is not None else settings.sample_rate / 2,
        aggregate=np.median,
    )
    return np.asarray(envelope, dtype=np.float32)
