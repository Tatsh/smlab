"""
Feature extraction and training-example construction.

Features are built in *beat space* rather than wall-clock frame space. Once the
tempo and offset are known, the spectrogram is resampled onto a musical grid of
:data:`SUBDIVISIONS_PER_BEAT` slots per beat, which makes every prediction land
on a legal note position by construction, renders a sixteenth-note run
identical at any tempo, and represents the rhythmic vocabulary DDR actually
uses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import logging

import librosa
import numpy as np

from .audio import DEFAULT_HOP_LENGTH, DEFAULT_SAMPLE_RATE, OnsetParams, load_audio, onset_envelope
from .chart import ACTIVE_CHARS, DIFFICULTIES, HOLD_HEAD, LIFT, MINE, ROLL_HEAD, TAIL, TAP
from .timing import BEATS_PER_MEASURE

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from .chart import Chart
    from .timing import TimingData

__all__ = (
    'CODE_BY_CHAR',
    'FEATURE_DIMENSION',
    'N_MELS',
    'PANEL_CODES',
    'SUBDIVISIONS_PER_BEAT',
    'ChartTargets',
    'beat_features',
    'chart_targets',
    'difficulty_index',
    'load_song_features',
    'measure_position',
    'placement_vector',
)

log = logging.getLogger(__name__)

SUBDIVISIONS_PER_BEAT = 12
"""
Grid slots per beat.

Twelve represents quarter, eighth, twelfth, sixteenth, twenty-fourth, and
forty-eighth notes exactly, which covers essentially all DDR charting.

:meta hide-value:
"""
N_MELS = 64
"""Mel bands retained per slot.

:meta hide-value:
"""
FEATURE_DIMENSION = N_MELS + 2
"""Feature width per slot: the mel bands plus mean and peak onset strength.

:meta hide-value:
"""
CODE_BY_CHAR = {TAP: 1, HOLD_HEAD: 2, TAIL: 3, ROLL_HEAD: 4, MINE: 5, LIFT: 6}
"""Integer code for each note character, where zero means an empty panel.

:meta hide-value:
"""
PANEL_CODES = 7
"""Number of distinct panel codes, including empty.

:meta hide-value:
"""
_FEATURE_PARAMS = OnsetParams(n_fft=1024, n_mels=N_MELS)


class ChartTargets:
    """Placement and selection targets for one chart on the beat grid."""

    def __init__(
        self, slots: NDArray[np.int32], panels: NDArray[np.uint8], difficulty: str, meter: int
    ) -> None:
        self.difficulty = difficulty
        """Canonical difficulty name."""
        self.meter = meter
        """Numeric difficulty rating."""
        self.panels = panels
        """Per-row panel codes, shaped ``(rows, 4)``."""
        self.slots = slots
        """Grid slot index of each row."""

    def __len__(self) -> int:
        """
        Return the number of note rows.

        Returns
        -------
        int
            The number of note rows.
        """
        return int(self.slots.size)


def difficulty_index(name: str) -> int:
    """
    Return the conditioning index for a difficulty name.

    Parameters
    ----------
    name : str
        Canonical difficulty name.

    Returns
    -------
    int
        Position within :data:`~smlab.chart.DIFFICULTIES`, or the ``Edit`` slot
        when unrecognised.
    """
    try:
        return DIFFICULTIES.index(name)
    except ValueError:
        return len(DIFFICULTIES) - 1


def beat_features(
    samples: NDArray[np.float32], timing: TimingData, *, sample_rate: int = DEFAULT_SAMPLE_RATE
) -> NDArray[np.float16]:
    """
    Resample audio onto the beat grid.

    Each slot aggregates the spectrogram frames that fall inside it, so the
    resulting sequence is indexed by musical position rather than by time.

    Parameters
    ----------
    samples : :py:class:`~numpy.ndarray`
        Mono audio samples.
    timing : TimingData
        Timing used to map beats onto times.
    sample_rate : int
        Sample rate of ``samples``.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        Features shaped ``(slots, FEATURE_DIMENSION)`` as 16-bit floats.
    """
    mel = librosa.feature.melspectrogram(
        y=samples, sr=sample_rate, n_fft=1024, hop_length=DEFAULT_HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    envelope = onset_envelope(samples, _FEATURE_PARAMS).astype(np.float32)
    frame_rate = sample_rate / DEFAULT_HOP_LENGTH
    duration = len(samples) / sample_rate
    # Only grid slots whose time falls inside the audio are usable.
    total_beats = timing.beat_at_time(duration)
    slot_count = max(int(total_beats * SUBDIVISIONS_PER_BEAT), 1)
    beats = np.arange(slot_count + 1, dtype=np.float64) / SUBDIVISIONS_PER_BEAT
    times = np.array([timing.time_at_beat(float(beat)) for beat in beats])
    edges = np.clip((times * frame_rate).astype(np.int64), 0, mel_db.shape[1])
    features = np.zeros((slot_count, FEATURE_DIMENSION), dtype=np.float32)
    for index in range(slot_count):
        start, stop = edges[index], max(edges[index + 1], edges[index] + 1)
        if start >= mel_db.shape[1]:
            break
        window = mel_db[:, start:stop]
        features[index, :N_MELS] = window.mean(axis=1)
        onset_window = envelope[min(start, len(envelope) - 1) : max(stop, start + 1)]
        if onset_window.size:
            features[index, N_MELS] = onset_window.mean()
            features[index, N_MELS + 1] = onset_window.max()
    return features.astype(np.float16)


def chart_targets(chart: Chart, slot_count: int) -> ChartTargets:
    """
    Convert a chart's note rows onto the beat grid.

    Rows that do not land on the grid are snapped to the nearest slot, which
    only affects the rare sixty-fourth notes the grid cannot express.

    Parameters
    ----------
    chart : Chart
        The chart to convert.
    slot_count : int
        Number of grid slots available, from :func:`beat_features`.

    Returns
    -------
    ChartTargets
        Slot indices and panel codes for every non-empty row.
    """
    slots: list[int] = []
    panels: list[list[int]] = []
    for beat, columns in chart.rows():
        slot = round(beat * SUBDIVISIONS_PER_BEAT)
        if not 0 <= slot < slot_count:
            continue
        codes = [CODE_BY_CHAR.get(character, 0) for character in columns[:4]]
        codes.extend([0] * (4 - len(codes)))
        slots.append(slot)
        panels.append(codes)
    if not slots:
        return ChartTargets(
            np.zeros(0, dtype=np.int32),
            np.zeros((0, 4), dtype=np.uint8),
            chart.difficulty,
            chart.meter,
        )
    return ChartTargets(
        np.asarray(slots, dtype=np.int32),
        np.asarray(panels, dtype=np.uint8),
        chart.difficulty,
        chart.meter,
    )


def placement_vector(targets: ChartTargets, slot_count: int) -> NDArray[np.float32]:
    """
    Build the binary step-presence target over the whole grid.

    Parameters
    ----------
    targets : ChartTargets
        Converted chart rows.
    slot_count : int
        Number of grid slots.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        One value per slot, set where the player must step.
    """
    vector = np.zeros(slot_count, dtype=np.float32)
    if len(targets):
        stepped = np.isin(targets.panels, [CODE_BY_CHAR[character] for character in ACTIVE_CHARS])
        vector[targets.slots[stepped.any(axis=1)]] = 1.0
    return vector


def measure_position(slot_count: int) -> NDArray[np.int64]:
    """
    Return each slot's position within its measure.

    Charting convention is strongly tied to metric position, so the models are
    told where in the bar each slot sits.

    Parameters
    ----------
    slot_count : int
        Number of grid slots.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        Slot position within the measure, one value per slot.
    """
    per_measure = int(BEATS_PER_MEASURE) * SUBDIVISIONS_PER_BEAT
    return np.arange(slot_count, dtype=np.int64) % per_measure


def load_song_features(audio_path: Path, timing: TimingData) -> NDArray[np.float16]:
    """
    Load an audio file and build its beat-grid features.

    Parameters
    ----------
    audio_path : :py:class:`~pathlib.Path`
        Path to the audio file.
    timing : TimingData
        Timing used to map beats onto times.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        Features shaped ``(slots, FEATURE_DIMENSION)``.
    """
    return beat_features(load_audio(audio_path), timing)
