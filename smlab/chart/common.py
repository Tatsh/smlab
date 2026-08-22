"""Data model for parsed simfiles and their note data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

from smlab.timing import BEATS_PER_MEASURE

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from smlab.timing import TimingData

__all__ = (
    'ACTIVE_CHARS',
    'AUDIO_SUFFIXES',
    'DIFFICULTIES',
    'FAKE',
    'HOLD_HEAD',
    'LIFT',
    'MINE',
    'ROLL_HEAD',
    'SINGLE_PANELS',
    'TAIL',
    'TAP',
    'Chart',
    'NoteRow',
    'Simfile',
    'normalize_difficulty',
)

TAP = '1'
"""Note character for a tap.

:meta hide-value:
"""
HOLD_HEAD = '2'
"""Note character starting a hold.

:meta hide-value:
"""
TAIL = '3'
"""Note character ending a hold or roll.

:meta hide-value:
"""
ROLL_HEAD = '4'
"""Note character starting a roll.

:meta hide-value:
"""
MINE = 'M'
"""Note character for a mine, which must be avoided rather than hit.

:meta hide-value:
"""
LIFT = 'L'
"""Note character for a lift, which is triggered by releasing the panel.

:meta hide-value:
"""
FAKE = 'F'
"""Note character for a note that is displayed but never judged.

:meta hide-value:
"""
ACTIVE_CHARS = frozenset({TAP, HOLD_HEAD, ROLL_HEAD, LIFT})
"""Note characters that require the player to step on a panel.

:meta hide-value:
"""
DIFFICULTIES = ('Beginner', 'Easy', 'Medium', 'Hard', 'Challenge', 'Edit')
"""Difficulty slots in StepMania's canonical order.

:meta hide-value:
"""
SINGLE_PANELS = 4
"""Number of panels in a ``dance-single`` chart.

:meta hide-value:
"""
AUDIO_SUFFIXES = ('.ogg', '.mp3', '.wav')
"""Audio file extensions searched when resolving ``#MUSIC``.

:meta hide-value:
"""

_DIFFICULTY_BY_LOWER = {name.lower(): name for name in DIFFICULTIES}


def normalize_difficulty(value: str) -> str:
    """
    Map a difficulty name onto its canonical spelling.

    Packs disagree on capitalisation, so ``challenge`` and ``Challenge`` must
    not become separate conditioning classes.

    Parameters
    ----------
    value : str
        The raw difficulty name.

    Returns
    -------
    str
        The canonical name, or the stripped input when unrecognised.
    """
    return _DIFFICULTY_BY_LOWER.get((stripped := value.strip()).lower(), stripped)


class NoteRow(NamedTuple):
    """One row of note data at a specific beat."""

    beat: float
    """Beat at which the row occurs."""
    columns: str
    """One note character per panel."""


@dataclass
class Chart:
    """One difficulty of one steps type within a simfile."""

    difficulty: str
    """Difficulty slot name, such as ``Challenge``."""
    meter: int
    """Numeric difficulty rating."""
    stepstype: str
    """Steps type, such as ``dance-single``."""
    description: str = ''
    """Free-form author or variant description."""
    raw_notes: str = field(default='', repr=False)
    """Unparsed note data, split into measures by :meth:`rows`."""

    @property
    def is_single(self) -> bool:
        """Whether this chart is a four-panel ``dance-single`` chart."""
        return self.stepstype == 'dance-single'

    def panel_count(self) -> int:
        """
        Return the number of panels this chart uses.

        Returns
        -------
        int
            Four for ``dance-single``, otherwise eight.
        """
        return SINGLE_PANELS if self.is_single else 8

    def rows(self) -> Iterator[NoteRow]:
        """
        Yield the non-empty note rows of this chart.

        A measure spans :data:`~smlab.timing.BEATS_PER_MEASURE` beats and is
        subdivided into however many rows it contains, which is how StepMania
        encodes quantisation: four rows are quarter notes, sixteen rows are
        sixteenth notes, and so on.

        Yields
        ------
        NoteRow
            Each row that contains at least one non-empty panel.
        """
        for measure_index, measure in enumerate(self.raw_notes.split(',')):
            lines = [
                stripped
                for line in measure.splitlines()
                if (stripped := line.strip()) and not stripped.startswith('//')
            ]
            if not lines:
                continue
            base = measure_index * BEATS_PER_MEASURE
            step = BEATS_PER_MEASURE / len(lines)
            for row_index, line in enumerate(lines):
                if set(line) <= {'0'}:
                    continue
                yield NoteRow(base + row_index * step, line)


@dataclass
class Simfile:
    """A parsed simfile header together with its charts."""

    path: Path
    """Location of the file this was parsed from."""
    artist: str = ''
    """The ``#ARTIST`` tag."""
    charts: list[Chart] = field(default_factory=list)
    """Every chart the file declares, across all steps types."""
    display_bpm: str = ''
    """The ``#DISPLAYBPM`` tag, which may be a range."""
    file_format: str = 'sm'
    """Source format: ``sm``, ``ssc``, or ``dwi``."""
    music: str = ''
    """The ``#MUSIC`` tag, resolved by :meth:`music_path`."""
    offset_declared: bool = False
    """
    Whether the file actually carried an offset tag.

    Many official arcade rips omit it entirely, so a zero offset from such a
    file is an absent label rather than a human-verified sync point.
    """
    sample_length: float = 0.0
    """The ``#SAMPLELENGTH`` tag in seconds."""
    sample_start: float = 0.0
    """The ``#SAMPLESTART`` tag in seconds."""
    subtitle: str = ''
    """The ``#SUBTITLE`` tag."""
    timing: TimingData | None = None
    """Parsed timing, or ``None`` when the file declared no usable tempo."""
    title: str = ''
    """The ``#TITLE`` tag."""

    def music_path(self) -> Path | None:
        """
        Resolve the audio file belonging to this simfile.

        The ``#MUSIC`` tag is matched case-insensitively because packs
        frequently disagree with the filesystem, and any audio file in the song
        directory is used as a fallback.

        Returns
        -------
        Path | None
            The audio file, or ``None`` when the directory contains none.
        """
        directory = self.path.parent
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            return None
        if self.music:
            if (candidate := directory / self.music).is_file():
                return candidate
            target = self.music.lower()
            for entry in entries:
                if entry.is_file() and entry.name.lower() == target:
                    return entry
        for suffix in AUDIO_SUFFIXES:
            for entry in entries:
                if entry.is_file() and entry.suffix.lower() == suffix:
                    return entry
        return None

    @property
    def offset(self) -> float:
        """The ``#OFFSET`` in seconds, or zero when timing is absent."""
        return self.timing.offset if self.timing else 0.0

    def singles(self) -> list[Chart]:
        """
        Return only the four-panel charts.

        Returns
        -------
        list[Chart]
            Every ``dance-single`` chart in the file.
        """
        return [chart for chart in self.charts if chart.is_single]
