"""Tests for drawing a chart."""

from __future__ import annotations

from typing import TYPE_CHECKING
import xml.etree.ElementTree as ET  # noqa: S405

from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

from smlab.chart_image import MEASURES_PER_COLUMN, Heading, render_chart, write_chart

_MEASURE = 48
_HEADING = Heading('Song', 'Hard', 9, 120.0)


def _parse(document: str) -> ET.Element:
    return ET.fromstring(document)  # noqa: S314


def test_an_empty_chart_still_draws_a_document() -> None:
    assert _parse(render_chart([], _HEADING)) is not None


def test_every_tap_becomes_one_arrow() -> None:
    rows = [(0, [1, 0, 0, 0]), (12, [0, 1, 1, 0]), (24, [0, 0, 0, 1])]
    root = _parse(render_chart(rows, _HEADING))
    arrows = [node for node in root.iter() if node.tag.endswith('polygon')]
    assert len(arrows) == 4


def test_each_lane_is_turned_a_different_way() -> None:
    # An up arrow rotated per lane is what makes the four panels legible; if two
    # lanes shared a rotation the picture would be wrong in a way that still
    # renders.
    rows = [(0, [1, 1, 1, 1])]
    root = _parse(render_chart(rows, _HEADING))
    turns = {node.get('points', '') for node in root.iter() if node.tag.endswith('polygon')}
    assert len(turns) == 4


def test_subdivision_decides_colour() -> None:
    # Quarter red, eighth blue, sixteenth yellow. Sharing a colour would hide
    # the rhythm the picture exists to show.
    rows = [(0, [1, 0, 0, 0]), (6, [1, 0, 0, 0]), (3, [1, 0, 0, 0])]
    root = _parse(render_chart(rows, _HEADING))
    colours = {node.get('fill') for node in root.iter() if node.tag.endswith('polygon')}
    assert len(colours) == 3


def test_a_freeze_draws_a_body_between_head_and_tail() -> None:
    rows = [(0, [2, 0, 0, 0]), (24, [3, 0, 0, 0])]
    root = _parse(render_chart(rows, _HEADING))
    bodies = [node for node in root.iter() if node.tag.endswith('rect') and node.get('rx')]
    assert bodies


def test_a_freeze_crossing_a_column_break_is_drawn_in_both() -> None:
    # Columns are separate places on the page, so one tall rectangle would run
    # through unrelated measures. It has to be split per measure instead.
    start = (MEASURES_PER_COLUMN - 1) * _MEASURE
    rows = [(start, [2, 0, 0, 0]), (start + 3 * _MEASURE, [3, 0, 0, 0])]
    root = _parse(render_chart(rows, _HEADING))
    bodies = [node for node in root.iter() if node.tag.endswith('rect') and node.get('rx')]
    assert len(bodies) > 1


def test_the_page_widens_with_the_song() -> None:
    short = _parse(render_chart([(0, [1, 0, 0, 0])], _HEADING))
    long = _parse(
        render_chart(
            [(0, [1, 0, 0, 0]), (MEASURES_PER_COLUMN * _MEASURE, [1, 0, 0, 0])],
            _HEADING,
        )
    )
    assert int(long.get('width', '0')) > int(short.get('width', '0'))


def test_the_title_is_escaped() -> None:
    document = render_chart([(0, [1, 0, 0, 0])], Heading('Rock & <Roll>', 'Hard', 9, 120.0))
    assert '<Roll>' not in document
    assert _parse(document) is not None


def test_a_png_is_written_when_the_suffix_says_so(tmp_path: Path) -> None:
    # PNG is the default output, so the raster path has to work without a
    # system font present.
    destination = tmp_path / 'chart.png'
    write_chart(destination, [(0, [1, 0, 0, 0]), (6, [0, 0, 0, 1])], _HEADING)
    assert destination.stat().st_size > 0
    with Image.open(destination) as image:
        assert image.size[0] > 0


def test_an_svg_is_written_when_the_suffix_says_so(tmp_path: Path) -> None:
    destination = tmp_path / 'chart.svg'
    write_chart(destination, [(0, [1, 0, 0, 0])], _HEADING)
    assert destination.read_text().startswith('<svg')
