"""Shared typing helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

__all__ = ('ChartRecord', 'ProgressCallback', 'SongRecord', 'TimingEstimate')

ProgressCallback = Callable[[str, int, int], None]
"""
Reports the progress of a download: file name, bytes received, and total expected.

The total is zero when the server does not say how large the file is.

:meta hide-value:
"""


class ChartRecord(TypedDict):
    """One chart summarised for the corpus manifest."""

    difficulty: str
    """Difficulty slot name."""
    meter: int
    """Numeric difficulty rating."""
    rows: int
    """Number of non-empty note rows."""
    style: str
    """How the chart can be performed: ``feet``, ``hands``, or ``keyboard``."""


class SongRecord(TypedDict):
    """One song summarised for the corpus manifest."""

    audio: str
    """Absolute path to the audio file."""
    bpms: list[list[float]]
    """Tempo segments as ``[beat, bpm]`` pairs."""
    charts: list[ChartRecord]
    """Every ``dance-single`` chart the simfile declares."""
    constant_bpm: bool
    """Whether the song never changes tempo and never stops."""
    file_format: str
    """Source format: ``sm``, ``ssc``, or ``dwi``."""
    offset: float
    """The ``#OFFSET`` in seconds. Beat 0 occurs at ``-offset``."""
    offset_declared: bool
    """
    Whether the simfile actually carried an offset tag.

    A zero offset from a file that declared none is an absent label, not a
    human-verified sync point, and must be excluded from timing benchmarks.
    """
    pack: str
    """Name of the pack directory containing the song."""
    primary_bpm: float
    """Tempo in effect at beat 0."""
    sample_length: float
    """The ``#SAMPLELENGTH`` tag in seconds, or zero when absent."""
    sample_start: float
    """The ``#SAMPLESTART`` tag in seconds, or zero when absent."""
    simfile: str
    """Absolute path to the simfile."""
    stops: int
    """Number of stop and delay segments."""
    title: str
    """The ``#TITLE`` tag."""


class TimingEstimate(TypedDict):
    """Timing recovered from audio by the estimator."""

    bpm: float
    """Estimated tempo in beats per minute."""
    confidence: float
    """Score of the winning candidate relative to the runner-up."""
    offset: float
    """Estimated ``#OFFSET`` in seconds."""
