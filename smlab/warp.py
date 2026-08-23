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

__all__ = (
    'DEFAULT_HOP',
    'DEFAULT_SHORTEST',
    'DEFAULT_SPAN',
    'DEFAULT_TOLERANCE',
    'DEFAULT_WINDOW',
    'TempoReading',
    'Warp',
    'fit_warps',
    'measure_tempo',
)

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
DEFAULT_TOLERANCE = 0.020
"""
How far the grid may wander from the music before another tempo segment is worth writing.

Twenty milliseconds is roughly where a step stops feeling on the beat. Raising it writes fewer
segments and lets the grid slide further between them.

:meta hide-value:
"""
DEFAULT_SHORTEST = 20.0
"""
Shortest stretch a fitted tempo segment may cover, in seconds.

A tempo is read off the slope of the phase, so a short segment reads its tempo from a short
baseline and is mostly noise. This also stops a song being cut into dozens of segments that each
describe the measurement rather than the music.

:meta hide-value:
"""
_MIN_FRAMES = 8
_MIN_READINGS = 2
_LEAST_COVER = 1e-6
_MIN_SEGMENT_POINTS = 4
_UNREACHABLE = (2**30, 0.0)


class TempoReading(NamedTuple):
    """The tempo measured across one stretch of a song."""

    seconds: float
    """Midpoint of the stretch, in seconds."""
    bpm: float
    """Tempo over the stretch, in beats per minute."""
    slip: float
    """Seconds the grid gains or loses across the stretch against the reference tempo."""


class Warp(NamedTuple):
    """A tempo the song holds from one moment until the next warp."""

    seconds: float
    """Where the tempo takes effect, in seconds. The first one is always zero."""
    bpm: float
    """Tempo from then on, in beats per minute."""


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


def _fit_cost(times: NDArray[np.float64], phases: NDArray[np.float64]) -> tuple[float, float]:
    """
    Measure how well one straight line describes a piece of the phase track.

    Parameters
    ----------
    times : :py:class:`~numpy.ndarray`
        Times of each measurement, in seconds.
    phases : :py:class:`~numpy.ndarray`
        Unwrapped phase at each of those times, in seconds.

    Returns
    -------
    tuple[float, float]
        Largest departure from the fitted line, in seconds, and the summed squared departure.
    """
    slope, intercept = np.polyfit(times, phases, 1)
    residual = phases - (slope * times + intercept)
    return float(np.max(np.abs(residual))), float(np.sum(residual**2))


def _segments(
    times: NDArray[np.float64], phases: NDArray[np.float64], tolerance: float, least: int
) -> list[tuple[int, int]]:
    """
    Cut the phase track into the fewest straight pieces that all stay within a tolerance.

    Parameters
    ----------
    times : :py:class:`~numpy.ndarray`
        Times of each measurement, in seconds.
    phases : :py:class:`~numpy.ndarray`
        Unwrapped phase at each of those times, in seconds.
    tolerance : float
        Largest departure from a straight line a piece may contain, in seconds.
    least : int
        Fewest measurements a piece may cover.

    Returns
    -------
    list[tuple[int, int]]
        Half-open index ranges covering the track in order.
    """
    total = len(times)
    # Working left to right and taking each piece as far as it will go is the obvious way to do
    # this and it is wrong: a piece runs past the point where the tempo changed before the tolerance
    # notices, and the next piece then straddles the change and reports a tempo belonging to neither
    # side. Choosing every boundary at once avoids inventing that bridge. Ties on the number of
    # pieces are settled by squared error, which is what pulls a boundary onto the change itself.
    best: list[tuple[int, float]] = [(0, 0.0), *([_UNREACHABLE] * total)]
    previous = [0] * (total + 1)
    for end in range(least, total + 1):
        for start in range(end - least + 1):
            if best[start] == _UNREACHABLE:
                continue
            spread, error = _fit_cost(times[start:end], phases[start:end])
            # A piece short enough that no shorter one could replace it is always allowed, so that
            # some covering always exists however badly the phase behaves.
            if spread > tolerance and end - start >= 2 * least:
                continue
            if (cost := (best[start][0] + 1, best[start][1] + error)) < best[end]:
                best[end], previous[end] = cost, start
    bounds = []
    end = total
    while end > 0:
        bounds.append((previous[end], end))
        end = previous[end]
    return bounds[::-1]


def _tempo(
    times: NDArray[np.float64], phases: NDArray[np.float64], lo: int, hi: int, bpm: float
) -> float:
    """
    Read the tempo off the slope of one piece of the phase track.

    Parameters
    ----------
    times : :py:class:`~numpy.ndarray`
        Times of each measurement, in seconds.
    phases : :py:class:`~numpy.ndarray`
        Unwrapped phase at each of those times, in seconds.
    lo : int
        First measurement of the piece.
    hi : int
        One past the last measurement of the piece.
    bpm : float
        Reference tempo the phase was tracked against.

    Returns
    -------
    float
        Tempo over the piece, in beats per minute.
    """
    return bpm * (1.0 - float(np.polyfit(times[lo:hi], phases[lo:hi], 1)[0]))


def _run_tempo(
    times: NDArray[np.float64],
    phases: NDArray[np.float64],
    run: list[tuple[int, int]],
    bpm: float,
) -> float:
    """
    Settle on one tempo for a group of pieces that were judged to share it.

    The tempo is the middle one by time rather than a line refitted across the whole group. A group
    is held together despite a phase step somewhere inside it, and a line drawn through a step comes
    out tilted, so refitting would read a tempo change off something that never changed tempo.
    Taking the middle tempo lets the short contaminated piece either side of a step be outvoted.

    Parameters
    ----------
    times : :py:class:`~numpy.ndarray`
        Times of each measurement, in seconds.
    phases : :py:class:`~numpy.ndarray`
        Unwrapped phase at each of those times, in seconds.
    run : list[tuple[int, int]]
        Half-open index ranges making up the group, in order.
    bpm : float
        Reference tempo the phase was tracked against.

    Returns
    -------
    float
        Tempo for the whole group, in beats per minute.
    """
    tempi = np.array([_tempo(times, phases, lo, hi, bpm) for lo, hi in run])
    covers = np.array([float(times[hi - 1] - times[lo]) for lo, hi in run])
    order = np.argsort(tempi)
    running = np.cumsum(covers[order])
    return float(tempi[order][int(np.searchsorted(running, running[-1] / 2.0))])


def _weakest_join(
    tempi: list[float], spans: list[float], bpm: float, tolerance: float
) -> tuple[int, int] | None:
    """
    Pick the run of segments least entitled to be told apart, if any of them is.

    Two neighbours are alike when the tempo between them would not move the grid by more than the
    tolerance over the shorter of the two. Failing that, a tempo that departs from its neighbour and
    comes straight back to it is a step in the measured phase rather than anything the music did:
    real tempo changes do not undo themselves, whereas an instrument entering shifts where the beat
    appears to sit and shifts back out again when the fit recovers.

    Parameters
    ----------
    tempi : list[float]
        Tempo of each run, in beats per minute.
    spans : list[float]
        How long each run covers, in seconds.
    bpm : float
        Reference tempo the phase was tracked against.
    tolerance : float
        How far the grid may wander from the music, in seconds.

    Returns
    -------
    tuple[int, int] or None
        First and last run to join, or ``None`` when every boundary earns its place.
    """
    joined, slightest = None, 1.0
    for index in range(len(tempi) - 1):
        needed = tolerance * bpm / max(min(spans[index], spans[index + 1]), _LEAST_COVER)
        if (share := abs(tempi[index + 1] - tempi[index]) / needed) < slightest:
            joined, slightest = (index, index + 1), share
    if joined is not None:
        return joined
    for index in range(1, len(tempi) - 1):
        if (tempi[index] - tempi[index - 1]) * (tempi[index + 1] - tempi[index]) >= 0:
            continue
        needed = tolerance * bpm / max(min(spans[index - 1], spans[index + 1]), _LEAST_COVER)
        if (share := abs(tempi[index + 1] - tempi[index - 1]) / needed) < slightest:
            joined, slightest = (index - 1, index + 1), share
    return joined


def _worthwhile(
    times: NDArray[np.float64],
    phases: NDArray[np.float64],
    bounds: list[tuple[int, int]],
    bpm: float,
    tolerance: float,
) -> list[list[tuple[int, int]]]:
    """
    Group neighbouring pieces whose tempi are too alike for the boundary to be worth writing.

    A tempo segment can only say that the beat spacing changed, so it is the wrong instrument for a
    phase step, and a phase step is what an arrangement change produces: the measured beat moves
    without the beat itself moving. Cutting the track there yields two pieces whose tempi barely
    differ, and writing that as a tempo change puts a tempo on the chart that the song never plays.
    A boundary therefore has to move the grid by more than the tolerance over the shorter of the
    two stretches it separates, otherwise it is describing the measurement rather than the music.

    Parameters
    ----------
    times : :py:class:`~numpy.ndarray`
        Times of each measurement, in seconds.
    phases : :py:class:`~numpy.ndarray`
        Unwrapped phase at each of those times, in seconds.
    bounds : list[tuple[int, int]]
        Half-open index ranges covering the track in order.
    bpm : float
        Reference tempo the phase was tracked against.
    tolerance : float
        How far the grid may wander from the music, in seconds.

    Returns
    -------
    list[list[tuple[int, int]]]
        The pieces gathered into the groups that earn a tempo of their own.
    """
    runs = [[piece] for piece in bounds]
    while len(runs) > 1:
        joined = _weakest_join(
            [_run_tempo(times, phases, run, bpm) for run in runs],
            [float(times[run[-1][1] - 1] - times[run[0][0]]) for run in runs],
            bpm,
            tolerance,
        )
        if joined is None:
            break
        first, last = joined
        runs = [
            *runs[:first],
            [piece for run in runs[first : last + 1] for piece in run],
            *runs[last + 1 :],
        ]
    return runs


def fit_warps(
    path: Path,
    bpm: float,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    window: float = DEFAULT_WINDOW,
    hop: float = DEFAULT_HOP,
    shortest: float = DEFAULT_SHORTEST,
) -> list[Warp]:
    """
    Work out the fewest tempo segments that keep the grid on the music throughout.

    A tempo held over a stretch is a straight line in the tracked phase, so the smallest set of
    tempi is the smallest set of straight pieces the phase can be cut into without any of them
    straying further than the tolerance. Where the pieces meet is where the tempo changed.

    Parameters
    ----------
    path : :py:class:`~pathlib.Path`
        Audio file to read.
    bpm : float
        Reference tempo the phase is tracked against.
    tolerance : float
        How far the grid may wander from the music before another segment is written, in seconds.
    window : float
        Seconds each phase measurement covers.
    hop : float
        Seconds between phase measurements.
    shortest : float
        Shortest stretch a segment may cover, in seconds.

    Returns
    -------
    list[Warp]
        Tempo segments in time order, the first always starting at zero seconds. A single entry
        means one tempo describes the whole song. Empty when the song is too short or too quiet to
        track a phase through.
    """
    envelope = onset_envelope(load_audio(path, sample_rate=PHASE_PARAMS.sample_rate), PHASE_PARAMS)
    centres, phases = _phase_track(
        envelope.astype(np.float64), PHASE_PARAMS.frame_rate, bpm, window, hop
    )
    least = max(int(shortest / hop), _MIN_SEGMENT_POINTS)
    if len(centres) < least:
        return []
    runs = _worthwhile(
        centres, phases, _segments(centres, phases, tolerance, least), bpm, tolerance
    )
    # The first segment sets the tempo the song opens at, and no measurement is centred before half
    # a window in, so it is pinned to the start rather than to where it was measured.
    return [
        Warp(
            seconds=0.0 if not index else float(centres[run[0][0]]),
            bpm=_run_tempo(centres, phases, run, bpm),
        )
        for index, run in enumerate(runs)
    ]
