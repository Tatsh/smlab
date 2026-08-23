"""
Measuring how a song's tempo wanders away from a single number.

A chart is written against one grid, so a song that drifts can only be charted correctly by writing
down where it drifts and by how much. This measures that: the beat phase is tracked against a fixed
reference tempo, and where the music runs faster or slower than the reference the tracked phase
slides. The slope of that slide is the tempo difference, and where the slope changes is where a
tempo segment belongs.

Phase per window is the argument of the envelope's Fourier coefficient at the beat frequency rather
than the peak of a fold. A fold's peak jumps to the off-beat whenever that peak is momentarily the
taller of the two, which reads as a spurious half-period step. The coefficient's phase is biased
late, because an onset envelope is an impulse train with a slow decay, but the measurement here is
entirely in the slope and a constant bias does not survive differentiation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from .audio import load_audio, onset_envelope
from .tempo import PHASE_PARAMS

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

__all__ = ('DEFAULT_HOP', 'DEFAULT_SPAN', 'DEFAULT_WINDOW', 'TempoReading', 'measure_tempo')

DEFAULT_WINDOW = 12.0
"""
Seconds of audio each phase measurement covers.

Long enough to average over roughly twenty beats, short enough that a tempo a fraction of a beat per
minute out has not slid a noticeable part of a beat across it.

:meta hide-value:
"""
DEFAULT_HOP = 2.0
"""Seconds between one phase measurement and the next.

:meta hide-value:
"""
DEFAULT_SPAN = 40.0
"""
Seconds of phase measurements a local tempo is fitted over.

A tempo is a slope, so it needs a baseline to be measured across. Forty seconds resolves a drift of
about a tenth of a beat per minute, which is roughly where a chart starts to feel loose.

:meta hide-value:
"""
_MIN_FRAMES = 8
_MIN_READINGS = 2


class TempoReading(NamedTuple):
    """The tempo measured across one stretch of a song."""

    seconds: float
    """Midpoint of the stretch, in seconds."""
    bpm: float
    """Tempo over the stretch, in beats per minute."""
    slip: float
    """Seconds the grid gains or loses across the stretch against the reference tempo."""


def _window_phase(
    envelope: NDArray[np.float64], frame_rate: float, period: float, start: float, span: float
) -> float:
    """
    Return the beat phase within one window of the envelope.

    Parameters
    ----------
    envelope : :py:class:`~numpy.ndarray`
        Onset strength envelope.
    frame_rate : float
        Envelope frames per second.
    period : float
        Beat period the phase is measured against, in seconds.
    start : float
        Where the window begins, in seconds.
    span : float
        How long the window is, in seconds.

    Returns
    -------
    float
        Phase within one beat, in seconds, or not-a-number when the window is unusable.
    """
    first = int(start * frame_rate)
    last = min(int((start + span) * frame_rate), len(envelope))
    piece = envelope[first:last]
    if len(piece) < _MIN_FRAMES:
        return float('nan')
    weight = np.maximum(piece - piece.mean(), 0.0)
    if (total := float(weight.sum())) <= 0.0:
        return float('nan')
    times = np.arange(len(piece), dtype=np.float64) / frame_rate + start
    vector = complex(np.sum(weight * np.exp(2j * np.pi * times / period)) / total)
    return float((np.angle(vector) / (2.0 * np.pi)) % 1.0 * period)


def _phase_track(
    envelope: NDArray[np.float64], frame_rate: float, bpm: float, window: float, hop: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Track the beat phase across a song against a fixed tempo.

    Parameters
    ----------
    envelope : :py:class:`~numpy.ndarray`
        Onset strength envelope.
    frame_rate : float
        Envelope frames per second.
    bpm : float
        Reference tempo the phase is measured against.
    window : float
        Seconds each measurement covers.
    hop : float
        Seconds between measurements.

    Returns
    -------
    tuple[:py:class:`~numpy.ndarray`, :py:class:`~numpy.ndarray`]
        Midpoint of each window in seconds, and the phase there, unwrapped so that a steady drift
        reads as a straight line rather than sawing around the beat.
    """
    period = 60.0 / bpm
    duration = len(envelope) / frame_rate
    starts = np.arange(0.0, max(duration - window, hop), hop)
    phases = np.array([
        _window_phase(envelope, frame_rate, period, float(s), window) for s in starts
    ])
    usable = np.isfinite(phases)
    if usable.sum() < _MIN_READINGS:
        return np.zeros(0), np.zeros(0)
    starts, phases = starts[usable], phases[usable]
    unwrapped = np.unwrap(phases / period * 2.0 * np.pi) / (2.0 * np.pi) * period
    return starts + window / 2.0, unwrapped


def measure_tempo(
    path: Path,
    bpm: float,
    *,
    window: float = DEFAULT_WINDOW,
    hop: float = DEFAULT_HOP,
    span: float = DEFAULT_SPAN,
) -> list[TempoReading]:
    """
    Measure how the tempo of a song varies around a reference.

    Parameters
    ----------
    path : :py:class:`~pathlib.Path`
        Audio file to read.
    bpm : float
        Reference tempo the phase is tracked against.
    window : float
        Seconds each phase measurement covers.
    hop : float
        Seconds between phase measurements.
    span : float
        Seconds each local tempo is fitted over.

    Returns
    -------
    list[TempoReading]
        One reading per stretch, in time order. Empty when the song is too short or too quiet to
        track a phase through.
    """
    envelope = onset_envelope(load_audio(path, sample_rate=PHASE_PARAMS.sample_rate), PHASE_PARAMS)
    centres, phases = _phase_track(
        envelope.astype(np.float64), PHASE_PARAMS.frame_rate, bpm, window, hop
    )
    width = max(int(span / hop), 4)
    if len(centres) <= width:
        return []
    readings = []
    for lo in range(0, len(centres) - width, max(int(span / 4.0 / hop), 1)):
        hi = lo + width
        slope = float(np.polyfit(centres[lo:hi], phases[lo:hi], 1)[0])
        covered = float(centres[hi - 1] - centres[lo])
        readings.append(
            TempoReading(
                seconds=float(centres[lo] + covered / 2.0),
                bpm=bpm * (1.0 - slope),
                slip=slope * covered,
            )
        )
    return readings
