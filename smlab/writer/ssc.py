"""
Serialisation of generated charts to StepMania ``.ssc`` files.

Every tag is written even when empty, because that is what the editor produces and it makes the
result straightforward to fill in afterwards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import hashlib
import re

from smlab.chart import DIFFICULTIES, HOLD_HEAD, LIFT, MINE, ROLL_HEAD, TAIL, TAP
from smlab.dataset import CODE_BY_CHAR, SUBDIVISIONS_PER_BEAT
from smlab.timing import BEATS_PER_MEASURE

from .common import by_measure, measure_text

if TYPE_CHECKING:
    from collections.abc import Sequence

    from smlab.timing import TimingData

    from .common import SongMetadata

__all__ = ('STEPFILE_VERSION', 'chart_hash', 'measure_block', 'radar_values', 'render_ssc')

STEPFILE_VERSION = '0.83'
"""Version an ``.ssc`` file declares, which must be its first tag.

:meta hide-value:
"""
_COMMENT_PATTERN = re.compile(r'//[^\n]*')
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


def _hold_ends(rows: Sequence[tuple[int, Sequence[int]]]) -> dict[tuple[int, int], int]:
    """
    Find where each freeze ends, keyed by where it began.

    Note rows carry a head and a tail rather than a duration, so how long a foot stays down is only
    known by looking ahead to the tail. Counting hands needs it: a freeze still held is a limb
    unavailable to the row it spans.

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

    The five leading figures are rates the engine normalises for display and the nine after them
    are plain counts. Read out of ``NoteDataUtil::CalculateRadarValues`` so that a written chart
    reports the same numbers the engine would compute for it.

    Parameters
    ----------
    rows : :py:class:`~collections.abc.Sequence`
        Grid slot and panel codes for each non-empty row.
    seconds : float
        Length of the song.

    Returns
    -------
    tuple[float, ...]
        Stream, voltage, air, freeze, and chaos, then the counts of notes, taps and holds, jumps,
        holds, mines, hands, rolls, lifts, and fakes.
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
            # A row counts once as a tap and once more as a jump, so a chord of three adds nothing
            # beyond what its second note already did.
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


def measure_block(rows: Sequence[tuple[int, Sequence[int]]]) -> str:
    """
    Render every measure of a chart, each labelled with its number.

    Parameters
    ----------
    rows : :py:class:`~collections.abc.Sequence`
        Grid slot and panel codes for each non-empty row.

    Returns
    -------
    str
        Measures separated by commas, each preceded by its number as a comment.
    """
    grouped = by_measure(rows)
    written = []
    for index in range(max(grouped) + 1 if grouped else 1):
        separator = '' if index == 0 else ',  '
        written.append(f'{separator}// measure {index}\n{measure_text(grouped.get(index, {}))}')
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

    A tag carrying anything puts its value on the tag's own line and closes on the next; an empty
    one closes immediately.

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

    Parameters
    ----------
    metadata : SongMetadata
        Header fields for the song.
    timing : TimingData
        Tempo and offset for the song.
    charts : :py:class:`~collections.abc.Sequence`
        Difficulty name, rating, and rows for each chart to write.
    seconds : float
        Length of the song, which the radar figures are rates over. Zero falls back to where each
        chart's last note lands.

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
        note_data = f'\n{measure_block(rows)}\n'
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
