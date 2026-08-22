"""Tests for drawing a chart."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import xml.etree.ElementTree as ET  # noqa: S405

from PIL import Image, ImageFont

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

from smlab.chart.image import MEASURES_PER_COLUMN, Heading, render_chart, write_chart

_MEASURE = 48
_HEADING = Heading('Song', 'Hard', 9, 120.0)


def _parse(document: str) -> ET.Element:
    return ET.fromstring(document)  # noqa: S314


def _arrow_color(root: ET.Element) -> str | None:
    return next(node.get('fill') for node in root.iter() if node.tag.endswith('polygon'))


def _bodies(document: str) -> list[float]:
    return [
        float(node.get('height', '0'))
        for node in _parse(document).iter()
        if node.tag.endswith('rect') and node.get('rx')
    ]


def test_an_empty_chart_still_draws_a_document() -> None:
    assert _parse(render_chart([], _HEADING)) is not None


def test_every_tap_becomes_one_arrow() -> None:
    rows = [(0, [1, 0, 0, 0]), (12, [0, 1, 1, 0]), (24, [0, 0, 0, 1])]
    root = _parse(render_chart(rows, _HEADING))
    arrows = [node for node in root.iter() if node.tag.endswith('polygon')]
    assert len(arrows) == 4


def test_each_lane_is_turned_a_different_way() -> None:
    # An up arrow rotated per lane is what makes the four panels legible; if two lanes shared a
    # rotation the picture would be wrong in a way that still renders.
    rows = [(0, [1, 1, 1, 1])]
    root = _parse(render_chart(rows, _HEADING))
    turns = {node.get('points', '') for node in root.iter() if node.tag.endswith('polygon')}
    assert len(turns) == 4


def test_subdivision_decides_color() -> None:
    # Quarter red, eighth blue, sixteenth yellow. Sharing a colour would hide the rhythm the
    # picture exists to show.
    rows = [(0, [1, 0, 0, 0]), (6, [1, 0, 0, 0]), (3, [1, 0, 0, 0])]
    root = _parse(render_chart(rows, _HEADING))
    colors = {node.get('fill') for node in root.iter() if node.tag.endswith('polygon')}
    assert len(colors) == 3


def test_a_freeze_draws_a_body_between_head_and_tail() -> None:
    rows = [(0, [2, 0, 0, 0]), (24, [3, 0, 0, 0])]
    root = _parse(render_chart(rows, _HEADING))
    bodies = [node for node in root.iter() if node.tag.endswith('rect') and node.get('rx')]
    assert bodies


def test_a_freeze_crossing_a_column_break_is_drawn_in_both() -> None:
    # Columns are separate places on the page, so one tall rectangle would run through unrelated
    # measures. It has to be split per measure instead.
    start = (MEASURES_PER_COLUMN - 1) * _MEASURE
    rows = [(start, [2, 0, 0, 0]), (start + 3 * _MEASURE, [3, 0, 0, 0])]
    root = _parse(render_chart(rows, _HEADING))
    bodies = [node for node in root.iter() if node.tag.endswith('rect') and node.get('rx')]
    assert len(bodies) > 1


def test_a_freeze_ending_on_a_bar_line_stays_inside_its_measure() -> None:
    # The tail slot ends the freeze rather than carrying any of it. Drawing the measure that bar
    # line opens took its extent from the previous column and produced one rectangle running the
    # whole height of the page.
    start = MEASURES_PER_COLUMN * _MEASURE - 12
    rows = [(start, [0, 2, 0, 0]), (start + 12, [0, 3, 0, 0])]
    bodies = _bodies(render_chart(rows, _HEADING))
    assert len(bodies) == 1
    # One beat of freeze cannot be taller than the four a whole measure covers.
    assert bodies[0] <= max(
        _bodies(render_chart([(0, [0, 2, 0, 0]), (_MEASURE, [0, 3, 0, 0])], _HEADING))
    )


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
    # PNG is the default output, so the raster path has to work without a system font present.
    destination = tmp_path / 'chart.png'
    write_chart(destination, [(0, [1, 0, 0, 0]), (6, [0, 0, 0, 1])], _HEADING)
    assert destination.stat().st_size > 0
    with Image.open(destination) as image:
        assert image.size[0] > 0


def test_an_svg_is_written_when_the_suffix_says_so(tmp_path: Path) -> None:
    destination = tmp_path / 'chart.svg'
    write_chart(destination, [(0, [1, 0, 0, 0])], _HEADING)
    assert destination.read_text().startswith('<svg')


def test_a_mine_is_drawn_as_a_ring() -> None:
    # A mine is not stepped on, so it must not read as an arrow.
    root = _parse(render_chart([(0, [5, 0, 0, 0])], _HEADING))
    assert [node for node in root.iter() if node.tag.endswith('circle')]
    assert not [node for node in root.iter() if node.tag.endswith('polygon')]


def test_a_roll_head_is_drawn_apart_from_a_tap() -> None:
    tap = _parse(render_chart([(0, [1, 0, 0, 0])], _HEADING))
    roll = _parse(render_chart([(0, [4, 0, 0, 0])], _HEADING))
    assert _arrow_color(tap) != _arrow_color(roll)


def test_measure_numbers_are_written_down_the_side() -> None:
    root = _parse(render_chart([(0, [1, 0, 0, 0])], _HEADING))
    texts = [node.text for node in root.iter() if node.tag.endswith('text')]
    assert '1' in texts


def test_a_png_falls_back_to_a_built_in_font(tmp_path: Path, mocker: MockerFixture) -> None:
    # A container with no system fonts still has to draw the measure numbers. Pillow's own default
    # font is loaded through truetype as well, so only a lookup by path may fail.
    real = ImageFont.truetype

    def missing(font: Any, size: int = 10, **kwargs: Any) -> ImageFont.FreeTypeFont:
        if isinstance(font, str):
            raise OSError
        return real(font, size, **kwargs)

    mocker.patch('PIL.ImageFont.truetype', missing)
    destination = tmp_path / 'chart.png'
    write_chart(destination, [(0, [1, 0, 0, 0]), (6, [0, 0, 0, 5])], _HEADING)
    with Image.open(destination) as image:
        assert image.size[0] > 0


def test_every_shape_kind_survives_the_raster_back_end(tmp_path: Path) -> None:
    # The two back ends draw the same list of shapes, so a kind the raster one cannot handle would
    # only show up here.
    destination = tmp_path / 'chart.png'
    rows = [(0, [2, 0, 0, 0]), (12, [3, 0, 0, 5]), (24, [4, 1, 0, 0])]
    write_chart(destination, rows, _HEADING)
    assert destination.stat().st_size > 0


def test_a_quarter_note_sits_inside_the_bar_rather_than_on_its_line() -> None:
    # Drawn where the arithmetic puts it, the bar line runs through the arrow and the measure reads
    # as starting a note early.
    root = _parse(render_chart([(0, [1, 0, 0, 0])], _HEADING))
    arrow = next(node for node in root.iter() if node.tag.endswith('polygon'))
    tops = [float(pair.split(',')[1]) for pair in (arrow.get('points') or '').split()]
    box = next(node for node in root.iter() if node.tag.endswith('rect') and node.get('stroke'))
    assert min(tops) > float(box.get('y') or 0)


def test_an_eighth_note_lands_on_a_beat_line() -> None:
    # Half a beat below the downbeat is where the first beat line is drawn.
    root = _parse(render_chart([(6, [1, 0, 0, 0])], _HEADING))
    arrow = next(node for node in root.iter() if node.tag.endswith('polygon'))
    centre = sum(float(p.split(',')[1]) for p in (arrow.get('points') or '').split()) / 7
    lines = [float(node.get('y1') or 0) for node in root.iter() if node.tag.endswith('line')]
    assert min(abs(centre - y) for y in lines) < 1.0
