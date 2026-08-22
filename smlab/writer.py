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
from typing import TYPE_CHECKING, Literal
import hashlib
import re
import shutil

from .chart import DIFFICULTIES, HOLD_HEAD, LIFT, MINE, ROLL_HEAD, TAIL, TAP
from .dataset import CODE_BY_CHAR, SUBDIVISIONS_PER_BEAT
from .timing import BEATS_PER_MEASURE

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

    from .timing import TimingData

__all__ = (
    'CHAR_BY_CODE',
    'SLOTS_PER_MEASURE',
    'STEPFILE_VERSION',
    'Format',
    'SongMetadata',
    'chart_hash',
    'measure_text',
    'radar_values',
    'render_simfile',
    'render_ssc',
    'safe_directory_name',
    'write_song',
)

Format = Literal['ssc', 'sm']
"""Simfile formats that can be written."""

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
_COMMENT_PATTERN = re.compile(r'//[^\n]*')
_UNSAFE_PATTERN = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_TRAILING_PATTERN = re.compile(r'[. ]+$')
_RADAR_COUNTS = (
    'notes',
    'taps_and_holds',
    'jumps',
    'holds',
    'mines',
    'hands',
    'rolls',
    'lifts',
    'fakes',
)
"""Radar categories that are plain counts, in the order the engine writes them."""
_RADAR_TAP_CODES = frozenset(
    CODE_BY_CHAR[character] for character in (TAP, HOLD_HEAD, ROLL_HEAD, LIFT)
)
_VOLTAGE_WINDOW_BEATS = 8.0
_VOLTAGE_WINDOW_SLOTS = int(_VOLTAGE_WINDOW_BEATS * SUBDIVISIONS_PER_BEAT)
_HAND_PANELS = 3
_HOLD_HEAD_CODES = frozenset(CODE_BY_CHAR[character] for character in (HOLD_HEAD, ROLL_HEAD))
_DEFAULT_TICKCOUNT = 4
_PLAYERS = 2
_JUMP_PANELS = 2
STEPFILE_VERSION = '0.83'
"""Version an ``.ssc`` file declares, which must be its first tag.

:meta hide-value:
"""
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


def _hold_ends(
    rows: Sequence[tuple[int, Sequence[int]]],
) -> dict[tuple[int, int], int]:
    """
    Find where each freeze ends, keyed by where it began.

    Note rows carry a head and a tail rather than a duration, so how long a
    foot stays down is only known by looking ahead to the tail. Counting hands
    needs it: a freeze still held is a limb unavailable to the row it spans.

    Parameters
    ----------
    rows : :py:class:`~collections.abc.Sequence`
        Grid slot and panel codes for each non-empty row.

    Returns
    -------
    dict[tuple[int, int], int]
        Slot the freeze ends on, keyed by its starting slot and panel.
    """
    open_at: dict[int, int] = {}
    ends: dict[tuple[int, int], int] = {}
    for slot, codes in sorted(rows):
        for panel, code in enumerate(codes):
            if code in _HOLD_HEAD_CODES:
                open_at[panel] = slot
            elif code == CODE_BY_CHAR[TAIL] and panel in open_at:
                ends[open_at.pop(panel), panel] = slot
    # A freeze left open by a truncated chart ends where the chart does.
    last = max(rows)[0] if rows else 0
    ends.update({(slot, panel): last for panel, slot in open_at.items()})
    return ends


def radar_values(rows: Sequence[tuple[int, Sequence[int]]], seconds: float) -> tuple[float, ...]:
    """
    Measure a chart the way StepMania's groove radar does.

    The five leading figures are rates the engine normalises for display and
    the nine after them are plain counts. Read out of
    ``NoteDataUtil::CalculateRadarValues`` so that a written chart reports the
    same numbers the engine would compute for it.

    Parameters
    ----------
    rows : :py:class:`~collections.abc.Sequence`
        Grid slot and panel codes for each non-empty row.
    seconds : float
        Length of the song.

    Returns
    -------
    tuple[float, ...]
        Stream, voltage, air, freeze, and chaos, then the counts of notes,
        taps and holds, jumps, holds, mines, hands, rolls, lifts, and fakes.
    """
    counts = dict.fromkeys(_RADAR_COUNTS, 0)
    taps = chaos = peak = 0
    recent: list[int] = []
    hold_ends: list[int] = []
    ends = _hold_ends(rows)
    for slot, codes in sorted(rows):
        hold_ends[:] = [end for end in hold_ends if end >= slot]
        held = len(hold_ends)
        # A row finer than an eighth is what the engine counts as chaos.
        chaos += slot % (SUBDIVISIONS_PER_BEAT // 2) != 0
        on_row = 0
        for panel, code in enumerate(codes):
            if code == CODE_BY_CHAR[MINE]:
                counts['mines'] += 1
                continue
            if code not in _RADAR_TAP_CODES:
                continue
            counts['notes'] += 1
            taps += 1
            on_row += 1
            recent.append(slot)
            # A row counts once as a tap and once more as a jump, so a chord of
            # three adds nothing beyond what its second note already did.
            if on_row == 1:
                counts['taps_and_holds'] += 1
            elif on_row == _JUMP_PANELS:
                counts['jumps'] += 1
            if code == CODE_BY_CHAR[HOLD_HEAD]:
                hold_ends.append(ends.get((slot, panel), slot))
                counts['holds'] += 1
            elif code == CODE_BY_CHAR[ROLL_HEAD]:
                hold_ends.append(ends.get((slot, panel), slot))
                counts['rolls'] += 1
            elif code == CODE_BY_CHAR[LIFT]:
                counts['lifts'] += 1
            del panel
        recent[:] = [start for start in recent if start >= slot - _VOLTAGE_WINDOW_SLOTS]
        peak = max(peak, len(recent))
        if on_row + held >= _HAND_PANELS:
            counts['hands'] += 1
    if seconds <= 0:
        return (0.0,) * 5 + tuple(float(counts[name]) for name in _RADAR_COUNTS)
    last_beat = (max(slot for slot, _ in rows) / SUBDIVISIONS_PER_BEAT) if rows else 0.0
    return (
        taps / seconds / 7.0,
        peak / _VOLTAGE_WINDOW_BEATS * (last_beat / seconds) / 10.0,
        counts['jumps'] / seconds,
        counts['holds'] / seconds,
        chaos / seconds * 0.5,
        *(float(counts[name]) for name in _RADAR_COUNTS),
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


def _measure_block(rows: Iterable[tuple[int, Sequence[int]]]) -> str:
    """
    Render every measure of a chart, each labelled with its number.

    Parameters
    ----------
    rows : :py:class:`~collections.abc.Iterable`
        Grid slot and panel codes for each non-empty row.

    Returns
    -------
    str
        Measures separated by commas, each preceded by its number as a comment.
    """
    by_measure: dict[int, dict[int, Sequence[int]]] = {}
    for slot, codes in rows:
        by_measure.setdefault(slot // SLOTS_PER_MEASURE, {})[slot % SLOTS_PER_MEASURE] = codes
    written = []
    for index in range(max(by_measure) + 1 if by_measure else 1):
        separator = '' if index == 0 else ',  '
        written.append(f'{separator}// measure {index}\n{measure_text(by_measure.get(index, {}))}')
    return '\n'.join(written)


def chart_hash(note_data: str) -> str:
    """
    Digest note data the way Project OutFox fills in ``#CHARTHASH``.

    The value hashed is the ``#NOTES`` tag exactly as the MSD parser hands it over, so comments are
    gone but nothing is trimmed and the newline after the tag's colon is part of it. Carriage
    returns are dropped because the file is read as text.

    Parameters
    ----------
    note_data : str
        Everything between the ``#NOTES`` colon and its closing semicolon.

    Returns
    -------
    str
        The digest as lower-case hexadecimal.
    """
    stripped = _COMMENT_PATTERN.sub('', note_data.replace('\r', ''))
    return hashlib.md5(stripped.encode(), usedforsecurity=False).hexdigest()


def _timing_tag(name: str, value: str) -> str:
    """
    Write one timing tag the way the editor lays them out.

    A tag carrying anything puts its value on the tag's own line and closes on
    the next; an empty one closes immediately.

    Parameters
    ----------
    name : str
        Tag name, without its hash.
    value : str
        The value, or an empty string.

    Returns
    -------
    str
        The tag, spanning one line or two.
    """
    return f'#{name}:;' if not value else f'#{name}:{value}\n;'


def render_ssc(
    metadata: SongMetadata,
    timing: TimingData,
    charts: Sequence[tuple[str, int, Sequence[tuple[int, Sequence[int]]]]],
    seconds: float = 0.0,
) -> str:
    """
    Render a complete ``.ssc`` file.

    Every tag is written even when empty, because that is what the editor
    produces and it makes the result straightforward to fill in afterwards.

    Parameters
    ----------
    metadata : SongMetadata
        Header fields for the song.
    timing : TimingData
        Tempo and offset for the song.
    charts : :py:class:`~collections.abc.Sequence`
        Difficulty name, rating, and rows for each chart to write.
    seconds : float
        Length of the song, which the radar figures are rates over. Zero falls
        back to where each chart's last note lands.

    Returns
    -------
    str
        The complete file contents.
    """
    bpms = ','.join(f'{segment.beat:.6f}={segment.bpm:.6f}' for segment in timing.bpms)
    stops = ','.join(f'{stop.beat:.6f}={stop.seconds:.6f}' for stop in timing.stops)
    credit = metadata.credit or 'smlab'
    lines = [
        f'#VERSION:{STEPFILE_VERSION};',
        f'#TITLE:{metadata.title};',
        f'#SUBTITLE:{metadata.subtitle};',
        f'#ARTIST:{metadata.artist};',
        f'#TITLETRANSLIT:{metadata.title_translit};',
        f'#SUBTITLETRANSLIT:{metadata.subtitle_translit};',
        f'#ARTISTTRANSLIT:{metadata.artist_translit};',
        f'#GENRE:{metadata.genre};',
        '#ORIGIN:;',
        '#TAGS:;',
        f'#CREDIT:{credit};',
        f'#BANNER:{metadata.banner};',
        f'#BACKGROUND:{metadata.background};',
        '#PREVIEWVID:;',
        '#JACKET:;',
        '#CDIMAGE:;',
        '#DISCIMAGE:;',
        '#LYRICSPATH:;',
        f'#CDTITLE:{metadata.cdtitle};',
        f'#MUSIC:{metadata.music};',
        f'#OFFSET:{timing.offset:.6f};',
        f'#SAMPLESTART:{metadata.sample_start:.6f};',
        f'#SAMPLELENGTH:{metadata.sample_length:.6f};',
        '#SELECTABLE:YES;',
        _timing_tag('BPMS', bpms),
        _timing_tag('STOPS', stops),
        _timing_tag('DELAYS', ''),
        _timing_tag('WARPS', ''),
        _timing_tag('TIMESIGNATURES', f'0.000000={int(BEATS_PER_MEASURE)}=4'),
        _timing_tag('TICKCOUNTS', f'0.000000={_DEFAULT_TICKCOUNT}'),
        _timing_tag('COMBOS', '0.000000=1'),
        _timing_tag('SPEEDS', '0.000000=1.000000=0.000000=0'),
        _timing_tag('SCROLLS', '0.000000=1.000000'),
        _timing_tag('XSCROLLS', '0.000000=0.000000'),
        _timing_tag('FAKES', ''),
        _timing_tag('LABELS', '0.000000=Song Start'),
        '#BGCHANGES:;',
        '#ATTACKS:\n;',
    ]
    for difficulty, meter, rows in charts:
        name = difficulty if difficulty in DIFFICULTIES else 'Edit'
        last = timing.time_at_beat(max(rows)[0] / SUBDIVISIONS_PER_BEAT) if rows else 0.0
        measured = radar_values(rows, seconds or last)
        radar = ','.join(f'{value:.6f}' for value in measured * _PLAYERS)
        note_data = f'\n{_measure_block(rows)}\n'
        lines.extend([
            '',
            f'//---------------dance-single - {credit}----------------',
            '#NOTEDATA:;',
            f'#CHARTNAME:{credit};',
            f'#CHARTHASH:{chart_hash(note_data)};',
            '#CHARTTYPE:dance-single;',
            '#STEPSTYPE:dance-single;',
            '#BANNER:;',
            f'#DESCRIPTION:{credit};',
            '#CHARTSTYLE:;',
            f'#DIFFICULTY:{name};',
            f'#METER:{meter};',
            '#METERF:0.000000;',
            f'#LASTSECONDHINT:{last:.6f};',
            f'#RADARVALUES:{radar};',
            f'#CREDIT:{credit};',
            f'#NOTES:{note_data};',
        ])
    return '\n'.join([*lines, ''])


def write_song(
    metadata: SongMetadata,
    audio: Path,
    timing: TimingData,
    charts: Sequence[tuple[str, int, Sequence[tuple[int, Sequence[int]]]]],
    parent: Path,
    fmt: Format = 'ssc',
    seconds: float = 0.0,
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
    fmt : Format
        Simfile format to write.
    seconds : float
        Length of the song, used by the ``.ssc`` radar figures.

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
    header = replace(metadata, music=destination.name)
    simfile = directory / f'{name}.{fmt}'
    text = (
        render_ssc(header, timing, charts, seconds)
        if fmt == 'ssc'
        else render_simfile(header, timing, charts)
    )
    simfile.write_text(text)
    return simfile
