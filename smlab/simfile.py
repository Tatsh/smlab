"""
Loaders for the StepMania simfile family.

Timing conversions follow :mod:`smlab.timing`, whose sign conventions were read
out of the StepMania source rather than assumed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .chart import HOLD_HEAD, TAIL, TAP, Chart, Simfile, normalize_difficulty
from .msd import parse_beat_value_list, parse_float, parse_msd, read_simfile_text
from .timing import BpmSegment, StopSegment, TimingData, gap_ms_to_offset

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from .msd import MsdTag

__all__ = ('SimfileError', 'load_simfile')

_SM_NOTES_FIELDS = 6
_DWI_CHART_FIELDS = 3
_DWI_ROWS_PER_BEAT = 48
_DWI_PANELS = {
    '0': '',
    '1': 'ld',
    '2': 'd',
    '3': 'rd',
    '4': 'l',
    '6': 'r',
    '7': 'lu',
    '8': 'u',
    '9': 'ru',
    'A': 'ud',
    'B': 'lr',
}
_DWI_COLUMN = {'l': 0, 'd': 1, 'u': 2, 'r': 3}
_DWI_DIFFICULTY = {
    'ANOTHER': 'Medium',
    'BASIC': 'Easy',
    'BEGINNER': 'Beginner',
    'MANIAC': 'Hard',
    'SMANIAC': 'Challenge',
}
# DWI expresses quantisation with bracket groups; a bare character is an eighth
# note, so each opener maps to its subdivision measured in beats.
_DWI_OPENERS = {'(': 0.25, '[': 1.0 / 6.0, '{': 1.0 / 16.0}
_DWI_CLOSERS = frozenset(')]}')
_DWI_DEFAULT_STEP = 0.5
_DWI_TICK_STEP = 1.0 / 48.0
_SSC_CHART_TAGS = frozenset({'DESCRIPTION', 'DIFFICULTY', 'METER', 'STEPSTYPE'})
_NOTES_TAGS = frozenset({'NOTES', 'NOTES2'})


class SimfileError(Exception):
    """Raised when a simfile cannot be parsed."""


def _text(tags: Sequence[MsdTag], name: str) -> str:
    """
    Return the first parameter of the first occurrence of a tag.

    Parameters
    ----------
    tags : :py:class:`~collections.abc.Sequence`
        All tags parsed from the file.
    name : str
        The tag name to look up.

    Returns
    -------
    str
        The value, or an empty string when the tag is absent.
    """
    for tag, params in tags:
        if tag == name and params:
            return params[0]
    return ''


def _dwi_events(stream: str) -> list[tuple[float, str, bool]]:
    """
    Decode a DWI note stream into timed panel combinations.

    Angle brackets combine several characters into one simultaneous step, and a
    trailing exclamation mark marks the start of a hold.

    Parameters
    ----------
    stream : str
        The raw DWI note stream for one chart.

    Returns
    -------
    list[tuple[float, str, bool]]
        Each event as its beat, the panels it uses, and whether it starts a hold.
    """
    events: list[tuple[float, str, bool]] = []
    stack: list[float] = []
    position = 0.0
    step = _DWI_DEFAULT_STEP
    in_ticks = False
    index = 0
    while index < len(stream):
        character = stream[index]
        if character in _DWI_OPENERS:
            stack.append(step)
            step = _DWI_OPENERS[character]
            index += 1
        elif character in _DWI_CLOSERS:
            step = stack.pop() if stack else _DWI_DEFAULT_STEP
            index += 1
        elif character == '`':
            # A backtick toggles 192nd notes rather than opening a pair.
            in_ticks = not in_ticks
            step = _DWI_TICK_STEP if in_ticks else _DWI_DEFAULT_STEP
            index += 1
        elif character == '<':
            if (closing := stream.find('>', index + 1)) < 0:
                break
            panels = ''.join(
                _DWI_PANELS.get(inner.upper(), '') for inner in stream[index + 1 : closing]
            )
            index = closing + 1
            is_hold = index < len(stream) and stream[index] == '!'
            index += int(is_hold)
            events.append((position, ''.join(sorted(set(panels))), is_hold))
            position += step
        elif (mapped := _DWI_PANELS.get(character.upper())) is not None:
            index += 1
            is_hold = index < len(stream) and stream[index] == '!'
            index += int(is_hold)
            events.append((position, mapped, is_hold))
            position += step
        else:
            index += 1
    return events


def _dwi_notes_to_sm(stream: str) -> str:
    """
    Convert a DWI note stream into ``.sm`` measure text.

    The result is rasterised onto a 48th-note grid, which represents every
    subdivision DWI can express apart from 192nd notes.

    Parameters
    ----------
    stream : str
        The raw DWI note stream for one chart.

    Returns
    -------
    str
        Measure text using the ``.sm`` note characters.
    """
    if not (events := _dwi_events(stream)):
        return '0000\n0000\n0000\n0000'
    rows_per_measure = _DWI_ROWS_PER_BEAT * 4
    highest = max(round(beat * _DWI_ROWS_PER_BEAT) for beat, _, _ in events) + 1
    measures = max((highest + rows_per_measure - 1) // rows_per_measure, 1)
    total_rows = measures * rows_per_measure
    rows = [['0'] * 4 for _ in range(total_rows)]
    holds_open: dict[int, bool] = {}
    for beat, panels, is_hold in events:
        if (row := round(beat * _DWI_ROWS_PER_BEAT)) >= total_rows:
            continue
        for panel in panels:
            column = _DWI_COLUMN[panel]
            if holds_open.get(column):
                rows[row][column] = TAIL
                holds_open[column] = False
            elif is_hold:
                rows[row][column] = HOLD_HEAD
                holds_open[column] = True
            else:
                rows[row][column] = TAP
    return '\n,\n'.join(
        '\n'.join(''.join(row) for row in rows[start : start + rows_per_measure])
        for start in range(0, total_rows, rows_per_measure)
    )


def _dwi_charts(tags: Sequence[MsdTag]) -> Iterator[Chart]:
    """
    Yield the four-panel charts declared by a ``.dwi`` file.

    Parameters
    ----------
    tags : :py:class:`~collections.abc.Sequence`
        All tags parsed from the file.

    Yields
    ------
    Chart
        Each ``#SINGLE`` chart, converted to ``.sm`` note data.
    """
    for tag, params in tags:
        if tag == 'SINGLE' and len(params) >= _DWI_CHART_FIELDS:
            yield Chart(
                difficulty=_DWI_DIFFICULTY.get(params[0].upper(), 'Edit'),
                meter=int(parse_float(params[1])),
                stepstype='dance-single',
                raw_notes=_dwi_notes_to_sm(params[2]),
            )


def _dwi_timing(tags: Sequence[MsdTag], path: Path) -> TimingData:
    """
    Build timing from a ``.dwi`` file's tags.

    Parameters
    ----------
    tags : :py:class:`~collections.abc.Sequence`
        All tags parsed from the file.
    path : :py:class:`~pathlib.Path`
        Location of the file, used in error messages.

    Returns
    -------
    TimingData
        The parsed timing.

    Raises
    ------
    SimfileError
        If the file declares no usable tempo.
    """
    if (bpm := parse_float(_text(tags, 'BPM'))) <= 0:
        msg = f'{path}: no usable #BPM.'
        raise SimfileError(msg)
    # NotesLoaderDWI.cpp:629 computes OFFSET as -GAP/1000.
    offset = gap_ms_to_offset(parse_float(_text(tags, 'GAP')))
    bpms = [BpmSegment(0.0, bpm)]
    # Change and freeze beats are indexed in quarter notes, and freeze
    # durations are expressed in milliseconds.
    bpms.extend(
        BpmSegment(beat / 4.0, value)
        for beat, value in parse_beat_value_list(_text(tags, 'CHANGEBPM'))
        if value > 0
    )
    stops = tuple(
        StopSegment(beat / 4.0, seconds / 1000.0)
        for beat, seconds in parse_beat_value_list(_text(tags, 'FREEZE'))
    )
    return TimingData(bpms=tuple(bpms), offset=offset, stops=stops)


def _load_dwi(path: Path, text: str) -> Simfile:
    """
    Parse a ``.dwi`` file.

    Parameters
    ----------
    path : :py:class:`~pathlib.Path`
        Location of the file, retained on the result.
    text : str
        The decoded file contents.

    Returns
    -------
    Simfile
        The parsed simfile.
    """
    tags = list(parse_msd(text))
    return Simfile(
        path=path,
        artist=_text(tags, 'ARTIST'),
        charts=list(_dwi_charts(tags)),
        file_format='dwi',
        music=_text(tags, 'FILE'),
        offset_declared=any(tag == 'GAP' for tag, _ in tags),
        timing=_dwi_timing(tags, path),
        title=_text(tags, 'TITLE'),
    )


def _sm_charts(tags: Sequence[MsdTag]) -> Iterator[Chart]:
    """
    Yield the charts declared by a ``.sm`` file.

    The format packs chart metadata into the first five colon-separated fields
    ahead of the note data.

    Parameters
    ----------
    tags : :py:class:`~collections.abc.Sequence`
        All tags parsed from the file.

    Yields
    ------
    Chart
        Each ``#NOTES`` chart.
    """
    for tag, params in tags:
        if tag == 'NOTES' and len(params) >= _SM_NOTES_FIELDS:
            yield Chart(
                difficulty=normalize_difficulty(params[2]),
                meter=int(parse_float(params[3])),
                stepstype=params[0],
                description=params[1],
                raw_notes=params[5],
            )


def _ssc_charts(tags: Sequence[MsdTag]) -> Iterator[Chart]:
    """
    Yield the charts declared by an ``.ssc`` file.

    Chart metadata accumulates from the enclosing ``#NOTEDATA`` block until that
    block's note data arrives.

    Parameters
    ----------
    tags : :py:class:`~collections.abc.Sequence`
        All tags parsed from the file.

    Yields
    ------
    Chart
        Each chart declared by a ``#NOTEDATA`` block.
    """
    pending: dict[str, str] = {}
    for tag, params in tags:
        value = params[0] if params else ''
        if tag == 'NOTEDATA':
            pending = {}
        elif tag in _SSC_CHART_TAGS:
            pending[tag] = value
        elif tag in _NOTES_TAGS:
            yield Chart(
                difficulty=normalize_difficulty(pending.get('DIFFICULTY', '')),
                meter=int(parse_float(pending.get('METER', '0'))),
                stepstype=pending.get('STEPSTYPE', ''),
                description=pending.get('DESCRIPTION', ''),
                raw_notes=value,
            )
            pending = {}


def _sm_timing(tags: Sequence[MsdTag], path: Path) -> TimingData:
    """
    Build timing from a ``.sm`` or ``.ssc`` file's tags.

    Parameters
    ----------
    tags : :py:class:`~collections.abc.Sequence`
        All tags parsed from the file.
    path : :py:class:`~pathlib.Path`
        Location of the file, used in error messages.

    Returns
    -------
    TimingData
        The parsed timing.

    Raises
    ------
    SimfileError
        If the file declares no usable tempo.
    """
    bpms = tuple(
        BpmSegment(beat, value)
        for beat, value in parse_beat_value_list(_text(tags, 'BPMS'))
        if value > 0
    )
    if not bpms:
        msg = f'{path}: no usable #BPMS.'
        raise SimfileError(msg)
    stops = [
        StopSegment(beat, seconds)
        for name in ('STOPS', 'FREEZES')
        for beat, seconds in parse_beat_value_list(_text(tags, name))
    ]
    stops.extend(
        StopSegment(beat, seconds, is_delay=True)
        for beat, seconds in parse_beat_value_list(_text(tags, 'DELAYS'))
    )
    # NotesLoaderSM.cpp:92 assigns #OFFSET straight through.
    return TimingData(bpms=bpms, offset=parse_float(_text(tags, 'OFFSET')), stops=tuple(stops))


def _load_sm_ssc(path: Path, text: str, *, is_ssc: bool) -> Simfile:
    """
    Parse a ``.sm`` or ``.ssc`` file.

    Parameters
    ----------
    path : :py:class:`~pathlib.Path`
        Location of the file, retained on the result.
    text : str
        The decoded file contents.
    is_ssc : bool
        Whether to apply ``.ssc`` semantics, in which charts are split into
        ``#NOTEDATA`` blocks.

    Returns
    -------
    Simfile
        The parsed simfile.
    """
    tags = list(parse_msd(text))
    display_bpm = next((':'.join(params) for tag, params in tags if tag == 'DISPLAYBPM'), '')
    return Simfile(
        path=path,
        artist=_text(tags, 'ARTIST'),
        charts=list(_ssc_charts(tags) if is_ssc else _sm_charts(tags)),
        display_bpm=display_bpm,
        file_format='ssc' if is_ssc else 'sm',
        music=_text(tags, 'MUSIC'),
        offset_declared=any(tag == 'OFFSET' for tag, _ in tags),
        sample_length=parse_float(_text(tags, 'SAMPLELENGTH')),
        sample_start=parse_float(_text(tags, 'SAMPLESTART')),
        subtitle=_text(tags, 'SUBTITLE'),
        timing=_sm_timing(tags, path),
        title=_text(tags, 'TITLE'),
    )


def load_simfile(path: Path) -> Simfile:
    """
    Parse a ``.sm``, ``.ssc``, or ``.dwi`` file.

    Parameters
    ----------
    path : :py:class:`~pathlib.Path`
        The simfile to load.

    Returns
    -------
    Simfile
        The parsed simfile.

    Raises
    ------
    SimfileError
        If the extension is unsupported or the file declares no usable tempo.
    """
    text = read_simfile_text(path)
    match path.suffix.lower():
        case '.dwi':
            return _load_dwi(path, text)
        case '.sm':
            return _load_sm_ssc(path, text, is_ssc=False)
        case '.ssc':
            return _load_sm_ssc(path, text, is_ssc=True)
        case unsupported:
            msg = f'{path}: unsupported extension {unsupported!r}.'
            raise SimfileError(msg)
