"""
Serialisation of generated charts to ``.dwi`` files.

Rows are not written one per grid position. A bare character advances an eighth of a measure and
bracket groups switch to a finer step, so a measure carries only as many characters as its own
resolution needs. Panels are packed into a single character, one per pair, and a freeze is written
where it starts as ``step!panel`` with no tail of its own, ending at the next step on that panel,
which the reader consumes.

Mines, lifts and rolls have no spelling here. Mines and lifts are dropped, and a roll is written as
an ordinary freeze.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from smlab.chart import HOLD_HEAD, ROLL_HEAD, TAIL, TAP
from smlab.dataset import CODE_BY_CHAR

from .common import SLOTS_PER_MEASURE, by_measure

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from smlab.timing import TimingData

    from .common import SongMetadata

__all__ = ('DIFFICULTY_NAMES', 'render_dwi', 'step_stream')

DIFFICULTY_NAMES = {
    'Beginner': 'BEGINNER',
    'Easy': 'BASIC',
    'Medium': 'ANOTHER',
    'Hard': 'MANIAC',
    'Challenge': 'SMANIAC',
    'Edit': 'EDIT',
}
"""The DWI name for each StepMania difficulty.

:meta hide-value:
"""
# The panel combination each DWI character stands for, as a frozenset of column indices in the
# left, down, up, right order the rest of the package uses.
_PANEL_CHARS = {
    frozenset(): '0',
    frozenset({0}): '4',
    frozenset({1}): '2',
    frozenset({2}): '8',
    frozenset({3}): '6',
    frozenset({0, 1}): '1',
    frozenset({1, 3}): '3',
    frozenset({0, 2}): '7',
    frozenset({2, 3}): '9',
    frozenset({1, 2}): 'A',
    frozenset({0, 3}): 'B',
}
# Strides in grid slots that DWI can state directly, coarsest first, with the brackets that open
# and close each. A bare character is already an eighth, so that one needs none.
_STRIDES = ((6, '', ''), (3, '(', ')'), (2, '[', ']'))
# Every remaining measure falls back to 192nd notes, the only finer resolution whose step divides
# the grid. Four characters cover one slot.
_TICKS_PER_SLOT = 4
_TICK_OPEN = '`'
_TICK_CLOSE = "'"
_PAIR = 2
_MILLISECONDS = 1000
_STEPPED = frozenset(CODE_BY_CHAR[character] for character in (TAP, HOLD_HEAD, ROLL_HEAD, TAIL))
_HELD = frozenset(CODE_BY_CHAR[character] for character in (HOLD_HEAD, ROLL_HEAD))


def _panels(codes: Sequence[int], wanted: frozenset[int]) -> frozenset[int]:
    """
    Select the panels of a row whose code is one of a set.

    Parameters
    ----------
    codes : :py:class:`~collections.abc.Sequence`
        Panel codes for one row.
    wanted : frozenset[int]
        Codes to select.

    Returns
    -------
    frozenset[int]
        Indices of the panels that matched.
    """
    return frozenset(panel for panel, code in enumerate(codes) if code in wanted)


def row_text(codes: Sequence[int]) -> str:
    """
    Write one row of panels as DWI step characters.

    A single character carries at most two panels, so a chord of three or more is written as an
    angle-bracket group of one character each. A freeze is marked by appending ``!`` and the panels
    it starts on.

    Parameters
    ----------
    codes : :py:class:`~collections.abc.Sequence`
        Panel codes for one row.

    Returns
    -------
    str
        The characters standing for that row, or ``0`` when nothing is stepped.
    """
    stepped = _panels(codes, _STEPPED)
    held = _panels(codes, _HELD)
    if not stepped:
        return '0'
    if len(stepped) <= _PAIR:
        return _PANEL_CHARS[stepped] + (f'!{_PANEL_CHARS[held]}' if held else '')
    # A group must not hold a zero before its closing bracket, or the reader takes it for a 192nd
    # note marker instead of a chord. Only stepped panels are written, so none appears.
    inner = ''.join(
        _PANEL_CHARS[frozenset({panel})]
        + (f'!{_PANEL_CHARS[frozenset({panel})]}' if panel in held else '')
        for panel in sorted(stepped)
    )
    return f'<{inner}>'


def _measure_stream(rows: dict[int, Sequence[int]]) -> str:
    """
    Write one measure at the coarsest step that lands every row on a character.

    Parameters
    ----------
    rows : dict[int, :py:class:`~collections.abc.Sequence`]
        Panel codes keyed by slot position within the measure.

    Returns
    -------
    str
        The measure's characters, bracketed when finer than an eighth.
    """
    occupied = {slot for slot, codes in rows.items() if _panels(codes, _STEPPED)}
    for stride, opener, closer in _STRIDES:
        if all(slot % stride == 0 for slot in occupied):
            body = ''.join(
                row_text(rows.get(slot, ())) for slot in range(0, SLOTS_PER_MEASURE, stride)
            )
            return f'{opener}{body}{closer}'
    body = ''.join(
        row_text(rows.get(divmod(tick, _TICKS_PER_SLOT)[0], ()))
        if tick % _TICKS_PER_SLOT == 0
        else '0'
        for tick in range(SLOTS_PER_MEASURE * _TICKS_PER_SLOT)
    )
    return f'{_TICK_OPEN}{body}{_TICK_CLOSE}'


def step_stream(rows: Sequence[tuple[int, Sequence[int]]]) -> str:
    """
    Write a whole chart as a DWI step stream.

    Parameters
    ----------
    rows : :py:class:`~collections.abc.Sequence`
        Grid slot and panel codes for each non-empty row.

    Returns
    -------
    str
        The stream, one measure per line so the result stays readable.
    """
    grouped = by_measure(rows)
    return '\n'.join(
        _measure_stream(grouped.get(index, {}))
        for index in range(max(grouped) + 1 if grouped else 1)
    )


def render_dwi(
    metadata: SongMetadata,
    timing: TimingData,
    charts: Iterable[tuple[str, int, Sequence[tuple[int, Sequence[int]]]]],
) -> str:
    """
    Render a complete ``.dwi`` file.

    Parameters
    ----------
    metadata : SongMetadata
        Header fields for the song.
    timing : TimingData
        Tempo and offset for the song.
    charts : :py:class:`~collections.abc.Iterable`
        Difficulty name, rating, and rows for each chart to write.

    Returns
    -------
    str
        The complete file contents.
    """
    # DWI counts whole milliseconds up to beat 0, the opposite sign to a StepMania offset.
    gap = round(-timing.offset * _MILLISECONDS)
    changes = ','.join(f'{segment.beat:.3f}={segment.bpm:.3f}' for segment in timing.bpms[1:])
    freezes = ','.join(
        f'{stop.beat:.3f}={stop.seconds * _MILLISECONDS:.3f}' for stop in timing.stops
    )
    lines = [
        f'#TITLE:{metadata.title};',
        f'#ARTIST:{metadata.artist};',
        f'#GENRE:{metadata.genre};',
        f'#CDTITLE:{metadata.cdtitle};',
        f'#BPM:{timing.primary_bpm:.3f};',
        f'#GAP:{gap};',
        f'#SAMPLESTART:{metadata.sample_start:.3f};',
        f'#SAMPLELENGTH:{metadata.sample_length:.3f};',
        f'#FILE:{metadata.music};',
        f'#FREEZE:{freezes};',
        f'#CHANGEBPM:{changes};',
        '',
    ]
    for difficulty, meter, rows in charts:
        name = DIFFICULTY_NAMES.get(difficulty, 'EDIT')
        lines.extend([f'#SINGLE:{name}:{meter}:', step_stream(rows), ';', ''])
    return '\n'.join(lines)
