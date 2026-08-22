"""
Pieces every simfile format shares.

Measures are emitted at the coarsest subdivision that still represents every row they contain,
which is what hand-written simfiles do and what makes note colours read correctly in game.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
import re

from smlab.dataset import SUBDIVISIONS_PER_BEAT
from smlab.timing import BEATS_PER_MEASURE

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = (
    'CHAR_BY_CODE',
    'SLOTS_PER_MEASURE',
    'Format',
    'SongMetadata',
    'by_measure',
    'measure_text',
    'safe_directory_name',
)

Format = Literal['dwi', 'sm', 'ssc']
"""Simfile formats that can be written."""

CHAR_BY_CODE = ('0', '1', '2', '3', '4', 'M', 'L')
"""Note character for each panel code, indexed by the code itself.

:meta hide-value:
"""
SLOTS_PER_MEASURE = int(BEATS_PER_MEASURE) * SUBDIVISIONS_PER_BEAT
"""Grid slots in one measure.

:meta hide-value:
"""
# Row counts a measure may be written at, coarsest first. Each must divide SLOTS_PER_MEASURE so
# that no note is displaced.
_SUBDIVISIONS = (4, 8, 12, 16, 24, 48)
# Characters that are illegal or troublesome in directory names across the filesystems StepMania
# packs get copied between.
_UNSAFE_PATTERN = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_TRAILING_PATTERN = re.compile(r'[. ]+$')
_MAX_NAME_LENGTH = 120


@dataclass(frozen=True, slots=True)
class SongMetadata:
    """Header fields written into a simfile."""

    artist: str = ''
    """The ``#ARTIST`` tag."""
    artist_translit: str = ''
    """Romanised artist, for the ``#ARTISTTRANSLIT`` tag."""
    background: str = ''
    """Background image file name, for the ``#BACKGROUND`` tag."""
    banner: str = ''
    """Banner image file name, for the ``#BANNER`` tag."""
    cdtitle: str = ''
    """Small mix logo file name, for the ``#CDTITLE`` tag."""
    credit: str = ''
    """Chart author, for the ``#CREDIT`` tag."""
    genre: str = ''
    """The ``#GENRE`` tag."""
    music: str = ''
    """Audio file name relative to the simfile, for the ``#MUSIC`` tag."""
    sample_length: float = 15.0
    """Preview length in seconds, for the ``#SAMPLELENGTH`` tag."""
    sample_start: float = 0.0
    """Preview start in seconds, for the ``#SAMPLESTART`` tag."""
    subtitle: str = ''
    """The ``#SUBTITLE`` tag."""
    subtitle_translit: str = ''
    """Romanised subtitle, for the ``#SUBTITLETRANSLIT`` tag."""
    title: str = ''
    """The ``#TITLE`` tag, which also names the song directory."""
    title_translit: str = ''
    """Romanised title, for the ``#TITLETRANSLIT`` tag."""


def safe_directory_name(title: str, fallback: str = 'Untitled') -> str:
    """
    Turn a song title into a usable directory name.

    Parameters
    ----------
    title : str
        The song title.
    fallback : str
        Name to use when nothing usable remains.

    Returns
    -------
    str
        A directory name with illegal characters removed.
    """
    cleaned = _UNSAFE_PATTERN.sub('', title).strip()
    # Windows refuses names ending in a dot or space, and such names are confusing everywhere else.
    cleaned = _TRAILING_PATTERN.sub('', cleaned)[:_MAX_NAME_LENGTH].strip()
    return cleaned or fallback


def measure_text(rows: Mapping[int, Sequence[int]]) -> str:
    """
    Render one measure of note data.

    Parameters
    ----------
    rows : :py:class:`~collections.abc.Mapping`
        Panel codes keyed by slot position within the measure.

    Returns
    -------
    str
        Note lines separated by newlines.
    """
    # The last candidate is one line per slot, whose stride of one divides every position, so the
    # search always settles on something.
    count = next(
        candidate
        for candidate in _SUBDIVISIONS
        if all(slot % (SLOTS_PER_MEASURE // candidate) == 0 for slot in rows)
    )
    stride = SLOTS_PER_MEASURE // count
    lines = []
    for index in range(count):
        codes = rows.get(index * stride)
        lines.append(''.join(CHAR_BY_CODE[code] for code in codes) if codes else '0000')
    return '\n'.join(lines)


def by_measure(
    rows: Sequence[tuple[int, Sequence[int]]],
) -> dict[int, dict[int, Sequence[int]]]:
    """
    Group rows by the measure they fall in.

    Parameters
    ----------
    rows : :py:class:`~collections.abc.Sequence`
        Grid slot and panel codes for each non-empty row.

    Returns
    -------
    dict[int, dict[int, Sequence[int]]]
        Panel codes keyed by slot within the measure, keyed by measure number.
    """
    grouped: dict[int, dict[int, Sequence[int]]] = {}
    for slot, codes in rows:
        grouped.setdefault(slot // SLOTS_PER_MEASURE, {})[slot % SLOTS_PER_MEASURE] = codes
    return grouped
