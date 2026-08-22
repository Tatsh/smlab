"""
Render a chart as a picture, in the style step artists read charts in.

Measures run down a column and columns run left to right, so a whole song fits on one page and its
shape — where the streams are, where it rests, whether one panel is being leaned on — is visible at
a glance. Note colour follows the StepMania convention, where the colour says which subdivision of
the beat a note falls on.

The page is laid out once as a list of shapes and then handed to one of two back ends, so the PNG
and the SVG are the same drawing rather than two drawings that have to be kept in step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple
from xml.sax.saxutils import escape
import math

from PIL import Image, ImageDraw, ImageFont

from smlab.dataset import SUBDIVISIONS_PER_BEAT

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

__all__ = ('MEASURES_PER_COLUMN', 'Heading', 'render_chart', 'write_chart')

MEASURES_PER_COLUMN = 16
"""Measures stacked in one column before starting the next."""

_BEATS_PER_MEASURE = 4
_SLOTS_PER_MEASURE = SUBDIVISIONS_PER_BEAT * _BEATS_PER_MEASURE
_LANES = 4
_LANE = 22
_BEAT_HEIGHT = 26
_MEASURE_HEIGHT = _BEAT_HEIGHT * _BEATS_PER_MEASURE
_COLUMN_GAP = 18
_MARGIN = 14
_HEADER = 34
_NOTE_SHIFT = _BEAT_HEIGHT / 2
"""
How far below its beat a note is drawn, in pixels.

Half a beat. Drawn where the arithmetic puts them, quarter notes sit exactly on the bar line and the
beat lines, so the line runs through the arrow and the eye reads the measure as starting a note
early. Nudging every note down by an eighth puts quarters inside the bar and eighths on the lines,
which is how a chart is read.
"""
_TAP = 1
_HOLD_HEAD = 2
_TAIL = 3
_ROLL_HEAD = 4
_MINE = 5
_HEADS = frozenset({_HOLD_HEAD, _ROLL_HEAD})
_ROTATION = (270, 180, 0, 90)
"""Degrees to turn an up arrow by for each of the left, down, up, right lanes."""
_COLORS = {
    0: '#e8443c',
    6: '#3d6fe0',
    4: '#a259e6',
    8: '#a259e6',
    3: '#e8c53c',
    9: '#e8c53c',
    2: '#e864b4',
    10: '#e864b4',
}
"""
Note colour by position within the beat, on the twelfth-of-a-beat grid.

Quarters are red, eighths blue, twelfths purple, sixteenths yellow and twenty-fourths pink,
following StepMania. Anything finer falls through to :py:data:`_FINE_COLOR`.
"""
_FINE_COLOR = '#4cc76a'
_MINE_COLOR = '#8d8d99'
_HOLD_COLOR = '#4cc76a'
_ROLL_COLOR = '#c8c8d2'
_PAGE = '#111116'
_ODD_MEASURE = '#1b1b22'
_EVEN_MEASURE = '#212129'
_MEASURE_EDGE = '#33333f'
_BEAT_LINE = '#2b2b35'
_NUMBER_COLOR = '#6a6a7a'
_TITLE_COLOR = '#e6e6ee'
_ARROW_POINTS = ((0, -7), (7, 0), (3, 0), (3, 7), (-3, 7), (-3, 0), (-7, 0))
_ARROW_SCALE = 1.25
_FONTS = (
    '/usr/share/fonts/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/TTF/DejaVuSans.ttf',
)


class Rect(NamedTuple):
    """A filled rectangle, optionally outlined and rounded."""

    x: float
    y: float
    width: float
    height: float
    fill: str
    outline: str | None = None
    radius: float = 0.0
    opacity: float = 1.0


class Line(NamedTuple):
    """A single straight line."""

    x1: float
    y1: float
    x2: float
    y2: float
    color: str


class Arrow(NamedTuple):
    """An arrow glyph, turned to face one of the four panels."""

    x: float
    y: float
    rotation: int
    fill: str


class Ring(NamedTuple):
    """An outlined circle, used for a mine."""

    x: float
    y: float
    radius: float
    color: str


class Text(NamedTuple):
    """A run of text."""

    x: float
    y: float
    size: int
    color: str
    content: str
    anchor: str = 'start'


Shape = Rect | Line | Arrow | Ring | Text


class Heading(NamedTuple):
    """What a drawn chart says across the top of the page."""

    title: str
    """Song title."""
    difficulty: str
    """Difficulty name."""
    meter: int
    """Difficulty rating."""
    bpm: float
    """Tempo."""


class Page(NamedTuple):
    """A laid-out chart, ready for either back end."""

    width: int
    height: int
    shapes: tuple[Shape, ...]


def _column_of(measure: int) -> tuple[int, int]:
    """
    Locate a measure on the page.

    Parameters
    ----------
    measure : int
        Measure number from the start of the chart.

    Returns
    -------
    tuple[int, int]
        Its column, and its offset in measures down that column.
    """
    return divmod(measure, MEASURES_PER_COLUMN)


def _position(slot: int) -> tuple[float, float]:
    """
    Convert a grid slot into a point on the page.

    Parameters
    ----------
    slot : int
        Grid slot, in twelfths of a beat.

    Returns
    -------
    tuple[float, float]
        Left edge of the lane block, and the vertical centre of the row.
    """
    column, offset = _column_of(slot // _SLOTS_PER_MEASURE)
    within = slot % _SLOTS_PER_MEASURE
    left = _MARGIN + column * (_LANES * _LANE + _COLUMN_GAP)
    top = (
        _HEADER
        + _NOTE_SHIFT
        + offset * _MEASURE_HEIGHT
        + within * _BEAT_HEIGHT / SUBDIVISIONS_PER_BEAT
    )
    return left, top


def _grid(measures: int) -> Iterable[Shape]:
    """
    Lay out the measure boxes and beat lines behind the notes.

    Parameters
    ----------
    measures : int
        Total measures in the chart.

    Yields
    ------
    Shape
        Background shapes.
    """
    for measure in range(measures):
        column, offset = _column_of(measure)
        left = _MARGIN + column * (_LANES * _LANE + _COLUMN_GAP)
        top = _HEADER + offset * _MEASURE_HEIGHT
        yield Rect(
            left,
            top,
            _LANES * _LANE,
            _MEASURE_HEIGHT,
            _ODD_MEASURE if measure % 2 else _EVEN_MEASURE,
            _MEASURE_EDGE,
        )
        for beat in range(1, _BEATS_PER_MEASURE):
            line = top + beat * _BEAT_HEIGHT
            yield Line(left, line, left + _LANES * _LANE, line, _BEAT_LINE)
        yield Text(left - 4, top + 11, 9, _NUMBER_COLOR, str(measure + 1), anchor='end')


def _note(lane: int, slot: int, code: int) -> Shape:
    """
    Lay out one note.

    Parameters
    ----------
    lane : int
        Panel index, ordered left, down, up, right.
    slot : int
        Grid slot the note falls on.
    code : int
        Panel code from the chart.

    Returns
    -------
    Shape
        The note's shape.
    """
    left, top = _position(slot)
    centre = left + lane * _LANE + _LANE / 2
    if code == _MINE:
        return Ring(centre, top, 7, _MINE_COLOR)
    color = _COLORS.get(slot % SUBDIVISIONS_PER_BEAT, _FINE_COLOR)
    if code == _ROLL_HEAD:
        color = '#f0f0f4'
    return Arrow(centre, top, _ROTATION[lane], color)


def _body(lane: int, start: int, stop: int, code: int) -> Iterable[Shape]:
    """
    Lay out the body of a freeze, which may run across a column break.

    Parameters
    ----------
    lane : int
        Panel index.
    start : int
        Slot the freeze begins on.
    stop : int
        Slot the freeze ends on.
    code : int
        Whether it is a hold or a roll.

    Yields
    ------
    Shape
        One rectangle per measure the freeze covers.
    """
    color = _ROLL_COLOR if code == _ROLL_HEAD else _HOLD_COLOR
    measure = start // _SLOTS_PER_MEASURE
    # The tail slot ends the freeze rather than carrying any of it, so a freeze stopping on a bar
    # line covers nothing of the measure that line opens.
    while measure <= (stop - 1) // _SLOTS_PER_MEASURE:
        head = max(start, measure * _SLOTS_PER_MEASURE)
        tail = min(stop, (measure + 1) * _SLOTS_PER_MEASURE)
        left, top = _position(head)
        _, bottom = _position(tail) if tail % _SLOTS_PER_MEASURE else _position(tail - 1)
        if tail % _SLOTS_PER_MEASURE == 0:
            bottom += _BEAT_HEIGHT / SUBDIVISIONS_PER_BEAT
        centre = left + lane * _LANE + _LANE / 2
        yield Rect(centre - 5, top, 10, max(bottom - top, 1.0), color, radius=4.0, opacity=0.55)
        measure += 1


def layout(rows: Sequence[tuple[int, Sequence[int]]], heading: Heading) -> Page:
    """
    Lay a chart out into shapes.

    Parameters
    ----------
    rows : Sequence[tuple[int, Sequence[int]]]
        Grid slot and four panel codes for each row, in ascending slot order.
    heading : Heading
        What to write across the top.

    Returns
    -------
    Page
        The laid-out page.
    """
    if not rows:
        return Page(1, 1, ())
    measures = rows[-1][0] // _SLOTS_PER_MEASURE + 1
    columns = (measures + MEASURES_PER_COLUMN - 1) // MEASURES_PER_COLUMN
    width = 2 * _MARGIN + columns * (_LANES * _LANE + _COLUMN_GAP)
    height = int(_HEADER + _NOTE_SHIFT + MEASURES_PER_COLUMN * _MEASURE_HEIGHT + _MARGIN)
    banner = (
        f'{heading.title} — {heading.difficulty} {heading.meter} — '
        f'{heading.bpm:.2f} BPM — {len(rows)} rows'
    )
    shapes: list[Shape] = [
        Rect(0, 0, width, height, _PAGE),
        Text(_MARGIN, 20, 14, _TITLE_COLOR, banner),
        *_grid(measures),
    ]
    open_freezes: dict[int, tuple[int, int]] = {}
    for slot, codes in rows:
        for lane, code in enumerate(codes):
            if code in _HEADS:
                open_freezes[lane] = (slot, code)
            elif code == _TAIL and lane in open_freezes:
                start, kind = open_freezes.pop(lane)
                shapes.extend(_body(lane, start, slot, kind))
    shapes.extend(
        _note(lane, slot, code)
        for slot, codes in rows
        for lane, code in enumerate(codes)
        if code in {_TAP, _HOLD_HEAD, _ROLL_HEAD, _MINE}
    )
    return Page(width, height, tuple(shapes))


def _arrow_points(shape: Arrow) -> list[tuple[float, float]]:
    """
    Turn an arrow glyph into absolute points.

    Parameters
    ----------
    shape : Arrow
        The arrow to place.

    Returns
    -------
    list[tuple[float, float]]
        The polygon's corners.
    """
    angle = math.radians(shape.rotation)
    cos, sin = math.cos(angle), math.sin(angle)
    return [
        (
            shape.x + _ARROW_SCALE * (x * cos - y * sin),
            shape.y + _ARROW_SCALE * (x * sin + y * cos),
        )
        for x, y in _ARROW_POINTS
    ]


def render_chart(rows: Sequence[tuple[int, Sequence[int]]], heading: Heading) -> str:
    """
    Draw a chart as an SVG document.

    Parameters
    ----------
    rows : Sequence[tuple[int, Sequence[int]]]
        Grid slot and four panel codes for each row, in ascending slot order.
    heading : Heading
        What to write across the top.

    Returns
    -------
    str
        A complete SVG document.
    """
    page = layout(rows, heading)
    if not page.shapes:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>'
    opening = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{page.width}" '
        f'height="{page.height}" viewBox="0 0 {page.width} {page.height}">'
    )
    parts = [opening]
    parts.extend(_svg_shape(shape) for shape in page.shapes)
    parts.append('</svg>')
    return '\n'.join(parts)


def _svg_shape(shape: Shape) -> str:
    """
    Write one shape as SVG.

    Parameters
    ----------
    shape : Shape
        The shape to write.

    Returns
    -------
    str
        One SVG element.
    """
    match shape:
        case Rect():
            edge = f' stroke="{shape.outline}"' if shape.outline else ''
            round_off = f' rx="{shape.radius}"' if shape.radius else ''
            fade = f' fill-opacity="{shape.opacity}"' if shape.opacity < 1 else ''
            return (
                f'<rect x="{shape.x:.1f}" y="{shape.y:.1f}" width="{shape.width:.1f}" '
                f'height="{shape.height:.1f}" fill="{shape.fill}"{edge}{round_off}{fade}/>'
            )
        case Line():
            return (
                f'<line x1="{shape.x1:.1f}" y1="{shape.y1:.1f}" x2="{shape.x2:.1f}" '
                f'y2="{shape.y2:.1f}" stroke="{shape.color}"/>'
            )
        case Arrow():
            points = ' '.join(f'{x:.1f},{y:.1f}' for x, y in _arrow_points(shape))
            return (
                f'<polygon points="{points}" fill="{shape.fill}" '
                f'stroke="#101014" stroke-width="1"/>'
            )
        case Ring():
            return (
                f'<circle cx="{shape.x:.1f}" cy="{shape.y:.1f}" r="{shape.radius}" '
                f'fill="none" stroke="{shape.color}" stroke-width="2.5" stroke-dasharray="3 2"/>'
            )
        case _:
            anchor = f' text-anchor="{shape.anchor}"' if shape.anchor != 'start' else ''
            family = 'monospace' if shape.anchor == 'end' else 'sans-serif'
            return (
                f'<text x="{shape.x:.1f}" y="{shape.y:.1f}" font-size="{shape.size}" '
                f'fill="{shape.color}"{anchor} font-family="{family}">'
                f'{escape(shape.content)}</text>'
            )


def write_chart(
    destination: Path, rows: Sequence[tuple[int, Sequence[int]]], heading: Heading
) -> None:
    """
    Draw a chart to a file, as SVG or PNG according to its suffix.

    Parameters
    ----------
    destination : :py:class:`~pathlib.Path`
        Where to write, ending in ``.svg`` or ``.png``.
    rows : Sequence[tuple[int, Sequence[int]]]
        Grid slot and four panel codes for each row, in ascending slot order.
    heading : Heading
        What to write across the top.
    """
    if destination.suffix.lower() == '.svg':
        destination.write_text(render_chart(rows, heading), encoding='utf-8')
        return
    _write_png(destination, layout(rows, heading))


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """
    Find a scalable font, falling back to the built-in bitmap one.

    Parameters
    ----------
    size : int
        Point size wanted.

    Returns
    -------
    :py:class:`PIL.ImageFont.ImageFont` or :py:class:`PIL.ImageFont.FreeTypeFont`
        A Pillow font.
    """
    for candidate in _FONTS:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _write_png(destination: Path, page: Page) -> None:
    """
    Draw a laid-out page to a PNG.

    Parameters
    ----------
    destination : :py:class:`~pathlib.Path`
        Where to write.
    page : Page
        The laid-out page.
    """
    image = Image.new('RGB', (page.width, page.height), _PAGE)
    draw = ImageDraw.Draw(image, 'RGBA')
    for shape in page.shapes:
        match shape:
            case Rect():
                box = (shape.x, shape.y, shape.x + shape.width, shape.y + shape.height)
                fill = _rgba(shape.fill, shape.opacity)
                if shape.radius:
                    draw.rounded_rectangle(box, radius=shape.radius, fill=fill)
                else:
                    draw.rectangle(box, fill=fill, outline=shape.outline)
            case Line():
                draw.line((shape.x1, shape.y1, shape.x2, shape.y2), fill=shape.color)
            case Arrow():
                draw.polygon(_arrow_points(shape), fill=shape.fill, outline='#101014')
            case Ring():
                draw.ellipse(
                    (
                        shape.x - shape.radius,
                        shape.y - shape.radius,
                        shape.x + shape.radius,
                        shape.y + shape.radius,
                    ),
                    outline=shape.color,
                    width=2,
                )
            case _:
                draw.text(
                    (shape.x, shape.y),
                    shape.content,
                    fill=shape.color,
                    font=_load_font(shape.size),
                    anchor='rs' if shape.anchor == 'end' else 'ls',
                )
    image.save(destination)


def _rgba(color: str, opacity: float) -> tuple[int, int, int, int]:
    """
    Turn a hex colour and an opacity into a Pillow colour.

    Parameters
    ----------
    color : str
        A ``#rrggbb`` string.
    opacity : float
        Alpha between zero and one.

    Returns
    -------
    tuple[int, int, int, int]
        Red, green, blue and alpha.
    """
    value = color.lstrip('#')
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return red, green, blue, round(255 * opacity)
