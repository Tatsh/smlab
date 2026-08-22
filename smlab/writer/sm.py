"""Serialisation of generated charts to StepMania ``.sm`` files."""

from __future__ import annotations

from typing import TYPE_CHECKING

from smlab.chart import DIFFICULTIES

from .common import by_measure, measure_text

if TYPE_CHECKING:
    from collections.abc import Sequence

    from smlab.timing import TimingData

    from .common import SongMetadata

__all__ = ('chart_text', 'render_simfile')


def chart_text(rows: Sequence[tuple[int, Sequence[int]]]) -> str:
    """
    Render every measure of a chart.

    Parameters
    ----------
    rows : :py:class:`~collections.abc.Sequence`
        Grid slot and panel codes for each non-empty row.

    Returns
    -------
    str
        Measures separated by commas, as ``.sm`` requires.
    """
    grouped = by_measure(rows)
    if not grouped:
        return measure_text({})
    return '\n,\n'.join(measure_text(grouped.get(index, {})) for index in range(max(grouped) + 1))


def render_simfile(
    metadata: SongMetadata,
    timing: TimingData,
    charts: Sequence[tuple[str, int, Sequence[tuple[int, Sequence[int]]]]],
) -> str:
    """
    Render a complete ``.sm`` file.

    Every header tag is written even when empty, which is what hand-authored simfiles do and what
    makes the result straightforward to edit afterwards.

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
            chart_text(rows),
            ';',
            '',
        ])
    return '\n'.join(lines)
