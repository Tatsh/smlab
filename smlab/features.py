"""
Beat-grid features built from separated stems.

Two grids are used and they are deliberately different. Notes live on twelfths of a beat, because
that is the rhythmic vocabulary charts are written in. Audio is sampled at twenty-fourths of a beat,
because averaging a whole note slot destroys the attack shape that distinguishes a kick from a snare
from a strummed chord. The network pools the fine grid down to the note grid itself, so it decides
what to discard rather than having it discarded beforehand.

Each stem contributes its own channels, so "which layer is prominent right now" is something a model
reads rather than infers. The mixture is kept alongside them because separation leaks, and the real
signal is a useful fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import logging

import librosa
import numpy as np

from .audio import DEFAULT_HOP_LENGTH, DEFAULT_SAMPLE_RATE
from .stems import STEM_NAMES

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .timing import TimingData

__all__ = (
    'FINE_SUBDIVISIONS',
    'MIXTURE_MELS',
    'SILENT_DECIBELS',
    'STEM_CHANNELS',
    'STEM_MELS',
    'TOTAL_CHANNELS',
    'fine_features',
    'grid_times',
    'mixture_loudness',
)

log = logging.getLogger(__name__)

FINE_SUBDIVISIONS = 24
"""
Audio samples per beat.

Twice the note grid, so the network sees the shape of an attack rather than its average. At 142
beats per minute one fine slot is 17.6 ms, close to the 5.8 ms hop of the underlying transform.

:meta hide-value:
"""
STEM_MELS = 48
"""Mel bands kept per separated stem.

:meta hide-value:
"""
MIXTURE_MELS = 64
"""Mel bands kept for the unseparated mixture.

:meta hide-value:
"""
STEM_CHANNELS = STEM_MELS + 2
"""Channels each stem contributes: its mel bands plus mean and peak onset.

:meta hide-value:
"""
TOTAL_CHANNELS = len(STEM_NAMES) * STEM_CHANNELS + MIXTURE_MELS + 2
"""Width of one fine slot across every stem and the mixture.

:meta hide-value:
"""
_N_FFT = 1024
_DECIBEL_FLOOR = -80.0
SILENT_DECIBELS = -70.0
"""
Level below which a slot is treated as carrying no music, in decibels.

Mel bands are measured against the loudest point of the song and floored at
:py:data:`_DECIBEL_FLOOR`, so digital silence sits exactly on that floor while music runs far above
it: over the tail of one song the played measures average -29 to -35 dB and the dead air after them
sits at -80. Anywhere in between separates the two, and this leaves ten decibels of room above the
floor for a fade that has not quite reached it.

:meta hide-value:
"""


def grid_times(timing: TimingData, duration: float) -> NDArray[np.float64]:
    """
    Return the time of every fine grid slot inside a song.

    Parameters
    ----------
    timing : TimingData
        Timing used to map beats onto times.
    duration : float
        Length of the audio in seconds.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        Slot times in seconds, one per fine slot.
    """
    total_beats = max(timing.beat_at_time(duration), 0.0)
    count = max(int(total_beats * FINE_SUBDIVISIONS), 1)
    beats = np.arange(count, dtype=np.float64) / FINE_SUBDIVISIONS
    return np.array([timing.time_at_beat(float(beat)) for beat in beats])


def _layer_features(
    samples: NDArray[np.float32], times: NDArray[np.float64], mels: int, sample_rate: int
) -> NDArray[np.float32]:
    """
    Summarise one audio layer on the fine grid.

    Parameters
    ----------
    samples : :py:class:`~numpy.ndarray`
        Mono samples for this layer.
    times : :py:class:`~numpy.ndarray`
        Fine grid slot times in seconds.
    mels : int
        Mel bands to keep.
    sample_rate : int
        Sample rate of ``samples``.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        Features shaped ``(slots, mels + 2)``.
    """
    spectrogram = librosa.feature.melspectrogram(
        y=samples, sr=sample_rate, n_fft=_N_FFT, hop_length=DEFAULT_HOP_LENGTH, n_mels=mels
    )
    decibels = librosa.power_to_db(spectrogram, ref=np.max, top_db=-_DECIBEL_FLOOR)
    onset = librosa.onset.onset_strength(
        S=decibels, sr=sample_rate, hop_length=DEFAULT_HOP_LENGTH, aggregate=np.median
    )
    frame_rate = sample_rate / DEFAULT_HOP_LENGTH
    frames = np.clip((times * frame_rate).astype(np.int64), 0, decibels.shape[1] - 1)
    edges = np.append(frames, decibels.shape[1])
    output = np.zeros((len(times), mels + 2), dtype=np.float32)
    for index in range(len(times)):
        start = edges[index]
        stop = max(edges[index + 1], start + 1)
        output[index, :mels] = decibels[:, start:stop].mean(axis=1)
        # The start edge is clipped inside the envelope and the stop edge is forced past it, so
        # this slice never comes back empty.
        window = onset[start:stop]
        output[index, mels] = window.mean()
        output[index, mels + 1] = window.max()
    return output


def fine_features(
    stems: dict[str, NDArray[np.float32]],
    mixture: NDArray[np.float32],
    timing: TimingData,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> NDArray[np.float16]:
    """
    Build the fine-grid feature array for one song.

    Parameters
    ----------
    stems : dict[str, :py:class:`~numpy.ndarray`]
        Separated mono stems, resampled to ``sample_rate``.
    mixture : :py:class:`~numpy.ndarray`
        The unseparated mono mixture.
    timing : TimingData
        Timing used to map beats onto times.
    sample_rate : int
        Sample rate of every supplied layer.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        Features shaped ``(slots, TOTAL_CHANNELS)`` as 16-bit floats.
    """
    times = grid_times(timing, len(mixture) / sample_rate)
    blocks = [
        _layer_features(stems.get(name, np.zeros_like(mixture)), times, STEM_MELS, sample_rate)
        for name in STEM_NAMES
    ]
    blocks.append(_layer_features(mixture, times, MIXTURE_MELS, sample_rate))
    return np.concatenate(blocks, axis=1).astype(np.float16)


def mixture_loudness(features: NDArray[np.float16]) -> NDArray[np.float32]:
    """
    Return how loud the unseparated mixture is at each note slot, in decibels.

    The note grid is half the resolution the features are built on, so each slot averages the two
    fine slots it covers.

    Parameters
    ----------
    features : :py:class:`~numpy.ndarray`
        Fine-grid features from :py:func:`fine_features`.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        One decibel level per note slot, taken from its loudest mel band.
    """
    start = len(STEM_NAMES) * STEM_CHANNELS
    bands = np.asarray(features[:, start : start + MIXTURE_MELS], dtype=np.float32)
    slots = bands.shape[0] // 2
    if slots == 0:
        return np.zeros(0, dtype=np.float32)
    # The loudest band, not the average of them. A sound occupying one corner of the spectrum
    # leaves every other band on the floor, so an average reads a held note or a solo instrument as
    # though it were silence.
    folded = bands[: 2 * slots].reshape(slots, 2 * MIXTURE_MELS)
    return np.asarray(folded.max(axis=1), dtype=np.float32)
