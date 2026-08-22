"""
Serialisation of generated charts to StepMania ``.sm`` files.

Measures are emitted at the coarsest subdivision that still represents every
row they contain, which is what hand-written simfiles do and what makes note
colours read correctly in game.

A song is written as a directory named after its title, holding the simfile and
a copy of the audio under the same name, which is the layout StepMania expects
inside a pack.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING
import re
import shutil

from .chart import DIFFICULTIES
from .dataset import SUBDIVISIONS_PER_BEAT
from .timing import BEATS_PER_MEASURE

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

    from .timing import TimingData

__all__ = (
    'CHAR_BY_CODE',
    'SLOTS_PER_MEASURE',
    'SongMetadata',
    'measure_text',
    'render_simfile',
    'safe_directory_name',
    'write_song',
)

CHAR_BY_CODE = ('0', '1', '2', '3', '4', 'M', 'L')
"""Note character for each panel code, indexed by the code itself.

:meta hide-value:
"""
SLOTS_PER_MEASURE = int(BEATS_PER_MEASURE) * SUBDIVISIONS_PER_BEAT
"""Grid slots in one measure.

:meta hide-value:
"""
# Row counts a measure may be written at, coarsest first. Each must divide
# SLOTS_PER_MEASURE so that no note is displaced.
_SUBDIVISIONS = (4, 8, 12, 16, 24, 48)
# Characters that are illegal or troublesome in directory names across the
# filesystems StepMania packs get copied between.
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
    # Windows refuses names ending in a dot or space, and such names are
    # confusing everywhere else.
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
    # The last candidate is one line per slot, whose stride of one divides
    # every position, so the search always settles on something.
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


def _chart_text(rows: Iterable[tuple[int, Sequence[int]]]) -> str:
    """
    Render every measure of a chart.

    Parameters
    ----------
    rows : :py:class:`~collections.abc.Iterable`
        Grid slot and panel codes for each non-empty row.

    Returns
    -------
    str
        Measures separated by commas, as ``.sm`` requires.
    """
    by_measure: dict[int, dict[int, Sequence[int]]] = {}
    for slot, codes in rows:
        by_measure.setdefault(slot // SLOTS_PER_MEASURE, {})[slot % SLOTS_PER_MEASURE] = codes
    if not by_measure:
        return measure_text({})
    return '\n,\n'.join(
        measure_text(by_measure.get(index, {})) for index in range(max(by_measure) + 1)
    )


def render_simfile(
    metadata: SongMetadata,
    timing: TimingData,
    charts: Sequence[tuple[str, int, Sequence[tuple[int, Sequence[int]]]]],
) -> str:
    """
    Render a complete ``.sm`` file.

    Every header tag is written even when empty, which is what hand-authored
    simfiles do and what makes the result straightforward to edit afterwards.

    Parameters
    ----------
    metadata : SongMetadata
        Header fields for the song.
    timing : TimingData
        Tempo and offset for the song.
    charts : :py:class:`~collections.abc.Sequence`
        Difficulty name, rating, and rows for each chart to write.

    Returns
    -------
    str
        The complete file contents.
    """
    bpms = ','.join(f'{segment.beat:.3f}={segment.bpm:.3f}' for segment in timing.bpms)
    stops = ','.join(f'{stop.beat:.3f}={stop.seconds:.3f}' for stop in timing.stops)
    lines = [
        f'#TITLE:{metadata.title};',
        f'#SUBTITLE:{metadata.subtitle};',
        f'#ARTIST:{metadata.artist};',
        f'#TITLETRANSLIT:{metadata.title_translit};',
        f'#SUBTITLETRANSLIT:{metadata.subtitle_translit};',
        f'#ARTISTTRANSLIT:{metadata.artist_translit};',
        f'#GENRE:{metadata.genre};',
        f'#CREDIT:{metadata.credit};',
        f'#BANNER:{metadata.banner};',
        f'#BACKGROUND:{metadata.background};',
        f'#CDTITLE:{metadata.cdtitle};',
        f'#MUSIC:{metadata.music};',
        f'#OFFSET:{timing.offset:.6f};',
        f'#SAMPLESTART:{metadata.sample_start:.3f};',
        f'#SAMPLELENGTH:{metadata.sample_length:.3f};',
        '#SELECTABLE:YES;',
        f'#BPMS:{bpms};',
        f'#STOPS:{stops};',
        '',
    ]
    for difficulty, meter, rows in charts:
        name = difficulty if difficulty in DIFFICULTIES else 'Edit'
        lines.extend([
            '#NOTES:',
            '     dance-single:',
            f'     {metadata.credit or "smlab"}:',
            f'     {name}:',
            f'     {meter}:',
            '     0.000,0.000,0.000,0.000,0.000:',
            _chart_text(rows),
            ';',
            '',
        ])
    return '\n'.join(lines)


def write_song(
    metadata: SongMetadata,
    audio: Path,
    timing: TimingData,
    charts: Sequence[tuple[str, int, Sequence[tuple[int, Sequence[int]]]]],
    parent: Path,
) -> Path:
    """
    Write a complete song directory.

    The directory is named after the song title and holds the simfile and a copy
    of the audio under that same name, which is how StepMania expects a song to
    sit inside a pack.

    Parameters
    ----------
    metadata : SongMetadata
        Header fields for the song. Its ``music`` field is replaced with the
        copied audio's name.
    audio : :py:class:`~pathlib.Path`
        Source audio file, which is copied rather than moved.
    timing : TimingData
        Tempo and offset for the song.
    charts : :py:class:`~collections.abc.Sequence`
        Difficulty name, rating, and rows for each chart to write.
    parent : :py:class:`~pathlib.Path`
        Directory the song directory is created inside, usually a pack.

    Returns
    -------
    :py:class:`~pathlib.Path`
        The simfile that was written.
    """
    name = safe_directory_name(metadata.title or audio.stem, audio.stem)
    directory = parent / name
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f'{name}{audio.suffix.lower()}'
    if audio.resolve() != destination.resolve():
        shutil.copy2(audio, destination)
    simfile = directory / f'{name}.sm'
    simfile.write_text(render_simfile(replace(metadata, music=destination.name), timing, charts))
    return simfile
