"""
Beat and time conversion matching StepMania's ``TimingData`` behaviour.

The sign conventions below were read out of the StepMania source tree rather than assumed:

* ``TimingData.cpp:1008`` sets ``start.last_time = -m_fBeat0OffsetInSeconds``, so beat 0 occurs at
  ``-OFFSET`` seconds into the music file.
* ``NotesLoaderSM.cpp:92`` assigns ``#OFFSET`` straight to ``m_fBeat0OffsetInSeconds``, so the tag
  needs no transformation.
* ``NotesLoaderDWI.cpp:629`` computes ``m_fBeat0OffsetInSeconds = -GAP/1000`` with ``GAP`` parsed as
  an integer, so DWI's GAP is the number of whole milliseconds until beat 0.

A positive ``#OFFSET`` places beat 0 before the start of the audio; the small negative offsets
typical of DDR simfiles place it just after.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple
import bisect
import operator

__all__ = (
    'BEATS_PER_MEASURE',
    'BPMSegment',
    'StopSegment',
    'TimingData',
    'gap_ms_to_offset',
    'offset_to_gap_ms',
)

BEATS_PER_MEASURE = 4.0
"""
Beats in one measure. DDR is 4/4 throughout and the ``.sm`` format hardcodes this, as time
signatures exist only as an ``.ssc`` extension.

:meta hide-value:
"""
_BPM_EPSILON = 1e-6


class BPMSegment(NamedTuple):
    """A tempo in effect from a given beat onwards."""

    beat: float
    """Beat at which this tempo takes effect."""
    bpm: float
    """Tempo in beats per minute."""


class StopSegment(NamedTuple):
    """A pause that advances time without advancing the beat."""

    beat: float
    """Beat at which the pause occurs."""
    seconds: float
    """Duration of the pause in seconds."""
    is_delay: bool = False
    """Whether this is an ``.ssc`` delay, which pauses before the row is judged."""


def gap_ms_to_offset(gap_ms: float) -> float:
    """
    Convert a DWI ``#GAP`` to a StepMania ``#OFFSET``.

    Parameters
    ----------
    gap_ms : float
        The DWI gap in milliseconds.

    Returns
    -------
    float
        The equivalent StepMania offset in seconds.
    """
    return -gap_ms / 1000.0


def offset_to_gap_ms(offset: float) -> float:
    """
    Convert a StepMania ``#OFFSET`` to a DWI ``#GAP``.

    Parameters
    ----------
    offset : float
        The StepMania offset in seconds.

    Returns
    -------
    float
        The equivalent DWI gap in milliseconds.
    """
    return -offset * 1000.0


@dataclass
class TimingData:
    """Timing for one song: the beat-0 offset, tempo changes, and stops."""

    bpms: tuple[BPMSegment, ...]
    """Tempo segments, which must contain at least one entry."""
    offset: float = 0.0
    """The ``#OFFSET`` tag value in seconds. Beat 0 occurs at ``-offset``."""
    stops: tuple[StopSegment, ...] = ()
    """Pauses that advance time without advancing the beat."""
    _beats: list[float] = field(default_factory=list, repr=False)
    _built: bool = field(default=False, repr=False)
    _rates: list[float] = field(default_factory=list, repr=False)
    _times: list[float] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        """
        Validate that the chart declares a tempo.

        Raises
        ------
        ValueError
            If no tempo segments were supplied.
        """
        if not self.bpms:
            msg = 'TimingData requires at least one BPM segment.'
            raise ValueError(msg)

    def beat_at_time(self, time: float) -> float:
        """
        Return the beat occurring at a given point in the audio.

        Parameters
        ----------
        time : float
            Seconds into the audio file.

        Returns
        -------
        float
            The corresponding beat, which may be negative before beat 0.
        """
        self._build()
        index = bisect.bisect_right(self._times, time) - 1
        if index < 0:
            # Before beat 0, so extrapolate backwards at the initial tempo.
            return (time - self._times[0]) / (60.0 / self.bpms[0].bpm)
        if (rate := self._rates[index]) <= 0.0:
            # Inside a stop, where the beat is frozen.
            return self._beats[index]
        return self._beats[index] + (time - self._times[index]) / rate

    def bpm_at_beat(self, beat: float) -> float:
        """
        Return the tempo in effect at a given beat.

        Parameters
        ----------
        beat : float
            The beat to query.

        Returns
        -------
        float
            Tempo in beats per minute.
        """
        current = self.bpms[0].bpm
        for segment in sorted(self.bpms):
            if segment.beat > beat:
                break
            current = segment.bpm
        return current

    def bpm_range(self) -> tuple[float, float]:
        """
        Return the lowest and highest tempo in the chart.

        Returns
        -------
        tuple[float, float]
            The minimum and maximum tempo in beats per minute.
        """
        values = [segment.bpm for segment in self.bpms]
        return min(values), max(values)

    @classmethod
    def constant(cls, bpm: float, offset: float) -> TimingData:
        """
        Build timing for a chart that never changes tempo.

        Parameters
        ----------
        bpm : float
            The fixed tempo in beats per minute.
        offset : float
            The ``#OFFSET`` tag value in seconds.

        Returns
        -------
        TimingData
            Timing with a single tempo segment and no stops.
        """
        return cls(bpms=(BPMSegment(0.0, bpm),), offset=offset)

    @property
    def is_constant_bpm(self) -> bool:
        """Whether the chart never changes tempo and never stops."""
        if self.stops:
            return False
        first = self.bpms[0].bpm
        return all(abs(segment.bpm - first) < _BPM_EPSILON for segment in self.bpms)

    @property
    def primary_bpm(self) -> float:
        """The tempo in effect at beat 0."""
        return self.bpms[0].bpm

    def shifted(self, delta_seconds: float) -> TimingData:
        """
        Return a copy whose beat 0 moves later in the audio.

        Because beat 0 sits at ``-offset``, moving it later decreases the offset.

        Parameters
        ----------
        delta_seconds : float
            How much later beat 0 should occur, in seconds.

        Returns
        -------
        TimingData
            A new instance with the adjusted offset.
        """
        return TimingData(bpms=self.bpms, offset=self.offset - delta_seconds, stops=self.stops)

    def time_at_beat(self, beat: float) -> float:
        """
        Return the point in the audio at which a beat occurs.

        Parameters
        ----------
        beat : float
            The beat to locate.

        Returns
        -------
        float
            Seconds into the audio file.
        """
        self._build()
        index = max(bisect.bisect_right(self._beats, beat) - 1, 0)
        # Several breakpoints share a beat across a stop, so taking the last
        # one judges notes on that beat after the pause elapses.
        return self._times[index] + (beat - self._beats[index]) * self._rates[index]

    def _build(self) -> None:
        """Precompute beat and time breakpoints so that lookups are logarithmic."""
        if self._built:
            return
        bpms = sorted(self.bpms)
        if bpms[0].beat > 0.0:
            # StepMania treats the first tempo as governing from beat 0 even
            # when the tag declares a later start beat.
            bpms.insert(0, BPMSegment(0.0, bpms[0].bpm))
        # Each event is (beat, kind, value); the kind orders tempo changes
        # before stops sharing a beat, matching StepMania's segment iteration.
        events = [(segment.beat, 0, segment.bpm) for segment in bpms[1:]]
        events.extend((stop.beat, 1, stop.seconds) for stop in sorted(self.stops))
        events.sort(key=operator.itemgetter(0, 1))
        self._beats = [0.0]
        self._times = [-self.offset]
        self._rates = []
        current_beat = 0.0
        current_time = -self.offset
        current_bpm = bpms[0].bpm
        for beat, kind, value in events:
            if beat > current_beat:
                rate = 60.0 / current_bpm if current_bpm > 0 else 0.0
                current_time += (beat - current_beat) * rate
                current_beat = beat
                self._rates.append(rate)
                self._beats.append(current_beat)
                self._times.append(current_time)
            if kind == 0:
                current_bpm = value
            else:
                # A stop records a zero-width beat interval so that the beat to
                # time mapping stays well defined across the pause.
                current_time += value
                self._rates.append(0.0)
                self._beats.append(current_beat)
                self._times.append(current_time)
        self._rates.append(60.0 / current_bpm if current_bpm > 0 else 0.0)
        self._built = True
