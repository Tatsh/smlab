"""
Deciding how a chart can physically be performed.

A keyboard offers four independent fingers with negligible travel, so any combination of panels is
reachable at any rate. A dance pad offers two feet and, at a stretch, two hands. Three separate
limits therefore decide what a chart demands, and they are measured separately here:

* **Geometry.** Two feet cannot be in three places, and a foot pinned by a hold cannot step
  elsewhere. Whether a valid assignment exists at all is decided by a dynamic program over foot
  positions rather than by pattern heuristics.
* **Chords.** Rows of three or four panels need hands as well as feet. That is a normal part of In
  The Groove charting, so it is reported as its own category rather than treated as a failure.
* **Stamina.** A sustained note rate can exceed what anyone holds with their legs while remaining
  comfortable under four fingers, so the note rate is measured over both a short and a long
  window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
import math

from .chart import HOLD_HEAD, LIFT, ROLL_HEAD, TAIL, TAP
from .dataset import CODE_BY_CHAR

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = (
    'BURST_WINDOW_SECONDS',
    'FOOT_STATES',
    'MAX_FEET',
    'PANEL_NAMES',
    'STREAM_SECONDS',
    'SUSTAINED_WINDOW_SECONDS',
    'PlayabilityReport',
    'Style',
    'analyze_rows',
    'is_crossover',
)

Style = Literal['feet', 'hands', 'keyboard']
"""How a chart can be performed: with feet alone, with hands too, or not on a pad."""

PANEL_NAMES = ('left', 'down', 'up', 'right')
"""Panel order used throughout, matching the ``.sm`` column order.

:meta hide-value:
"""
MAX_FEET = 2
"""Contact points a dancer has on the pad without using their hands.

:meta hide-value:
"""
MAX_LIMBS = 4
"""Contact points available when hands are used as well.

:meta hide-value:
"""
FOOT_STATES = tuple((left, right) for left in range(4) for right in range(4) if left != right)
"""Every position the two feet can occupy, as ``(left, right)`` panel indices.

:meta hide-value:
"""
BURST_WINDOW_SECONDS = 1.0
"""Window over which a short burst of notes is measured.

:meta hide-value:
"""
SUSTAINED_WINDOW_SECONDS = 10.0
"""Window over which sustained effort is measured.

:meta hide-value:
"""
MAX_BURST_NPS = 20.0
"""
Note rate a dancer can reach momentarily, in notes per second.

Beyond this a passage stops being a flourish and becomes a keyboard pattern.

:meta hide-value:
"""
MAX_SUSTAINED_NPS = 12.0
"""
Note rate a dancer can hold for :data:`SUSTAINED_WINDOW_SECONDS`.

Sixteenth notes at 180 beats per minute are twelve notes per second, which is already the territory
of the hardest pad charts ever written.

:meta hide-value:
"""
FAST_JACK_SECONDS = 0.09
"""
Shortest interval in which one foot can retap the same panel.

:meta hide-value:
"""
STREAM_SECONDS = 0.16
"""
Gap within which consecutive notes are danced as a run, in seconds.

Closer together than this the feet alternate rather than choosing a panel each, which is what makes
a step land crossed or not.

:meta hide-value:
"""
_LEFT_PANEL = 0
_RIGHT_PANEL = 3
_STEP_CODES = frozenset(CODE_BY_CHAR[character] for character in (TAP, HOLD_HEAD, ROLL_HEAD, LIFT))
_HOLD_START_CODES = frozenset({CODE_BY_CHAR[HOLD_HEAD], CODE_BY_CHAR[ROLL_HEAD]})
_TAIL_CODE = CODE_BY_CHAR[TAIL]
_CROSSOVER_COST = 4.0
_DOUBLE_STEP_COST = 2.0
_MOVE_COST = 0.25
_IMPOSSIBLE = math.inf
_MIN_ROWS_FOR_RATE = 2


def is_crossover(left: int, right: int) -> bool:
    """
    Return whether a foot position requires the legs to cross.

    Parameters
    ----------
    left : int
        Panel index under the left foot.
    right : int
        Panel index under the right foot.

    Returns
    -------
    bool
        True when the left foot is right of the right foot.
    """
    return left == _RIGHT_PANEL or right == _LEFT_PANEL


@dataclass(frozen=True, slots=True)
class PlayabilityReport:
    """How a chart can be performed, and the measurements behind the verdict."""

    burst_nps: float
    """Highest note rate over :data:`BURST_WINDOW_SECONDS`."""
    chord_rows: int
    """Rows needing three or four panels, which require hands."""
    crossovers: int
    """Steps a running dancer meets on a crossed foot."""
    fastest_jack: float
    """Shortest interval between repeats of one panel, in seconds."""
    geometrically_possible: bool
    """Whether any two-foot assignment covers the chart."""
    impossible_rows: int
    """Rows needing more than four panels, which nothing can cover."""
    max_simultaneous: int
    """Largest number of panels required at one instant."""
    reasons: tuple[str, ...]
    """Human-readable explanations for anything beyond feet alone."""
    style: Style
    """The least demanding way the chart can be performed."""
    sustained_nps: float
    """Highest note rate held over :data:`SUSTAINED_WINDOW_SECONDS`."""

    @property
    def pad_playable(self) -> bool:
        """Whether the chart can be performed on a pad at all, hands included."""
        return self.style != 'keyboard'


def _required_panels(rows: Sequence[tuple[float, Sequence[int]]]) -> list[tuple[float, set[int]]]:
    """
    Return the panels that must be occupied at each row.

    A panel under a hold stays occupied until its tail, so it counts against the limb budget of
    every row in between. This is what makes an otherwise ordinary note impossible: with two panels
    held, no foot remains free.

    Parameters
    ----------
    rows : :py:class:`~collections.abc.Sequence`
        Time in seconds and panel codes for each row.

    Returns
    -------
    list[tuple[float, set[int]]]
        Time and occupied panel indices for each row.
    """
    held: set[int] = set()
    output: list[tuple[float, set[int]]] = []
    for time, codes in rows:
        for panel, code in enumerate(codes):
            if code == _TAIL_CODE:
                held.discard(panel)
        stepped = {panel for panel, code in enumerate(codes) if code in _STEP_CODES}
        if occupied := stepped | held:
            output.append((time, occupied))
        for panel, code in enumerate(codes):
            if code in _HOLD_START_CODES:
                held.add(panel)
    return output


def _window_rate(times: Sequence[float], window: float) -> float:
    """
    Return the highest note rate sustained over a sliding window.

    Parameters
    ----------
    times : :py:class:`~collections.abc.Sequence`
        Row times in seconds, ordered.
    window : float
        Window length in seconds.

    Returns
    -------
    float
        Notes per second, or zero when the chart is shorter than the window.
    """
    if len(times) < _MIN_ROWS_FOR_RATE or times[-1] - times[0] < window:
        return 0.0
    best = 0.0
    start = 0
    for end, time in enumerate(times):
        while time - times[start] > window:
            start += 1
        best = max(best, (end - start + 1) / window)
    return best


def _transition_cost(previous: tuple[int, int], current: tuple[int, int], gap: float) -> float:
    """
    Return the effort of moving between two foot positions.

    Parameters
    ----------
    previous : tuple[int, int]
        Foot position before the row.
    current : tuple[int, int]
        Foot position at the row.
    gap : float
        Seconds since the previous row.

    Returns
    -------
    float
        Movement cost, or infinity when a foot would have to move impossibly fast.
    """
    moved = (previous[0] != current[0]) + (previous[1] != current[1])
    if moved == MAX_FEET and gap < FAST_JACK_SECONDS:
        # Both feet cannot reposition within one very short interval.
        return _IMPOSSIBLE
    cost = _MOVE_COST * moved
    if is_crossover(*current):
        cost += _CROSSOVER_COST
    if moved == 0:
        cost += _DOUBLE_STEP_COST
    return cost


def _crossed_steps(rows: Sequence[tuple[float, Sequence[int]]]) -> int:
    """
    Count the steps a running dancer meets on a crossed foot.

    Inside a run the feet alternate, so which panel a step lands on decides whether the legs cross:
    it is the panel sequence that crosses, not any one row. Between runs the feet reset to whichever
    suits the note, because there is time to choose.

    Asking the feasibility search instead would always answer none. It prices a crossed stance at
    :py:data:`_CROSSOVER_COST`, so the cheapest assignment it retains avoids crossing wherever it
    can, which across four hundred corpus charts is everywhere.

    Parameters
    ----------
    rows : :py:class:`~collections.abc.Sequence`
        Time in seconds and panel codes for each row.

    Returns
    -------
    int
        Steps landing on a foot that has crossed over the other.
    """
    foot = 0
    crossings = 0
    previous_time: float | None = None
    for time, codes in rows:
        stepped = {panel for panel, code in enumerate(codes) if code in _STEP_CODES}
        if not stepped:
            # A row that only releases a freeze is not a step, so it neither takes a turn in the
            # alternation nor breaks the run around it.
            continue
        # The opening note has no predecessor to alternate away from, so it sets the foot rather
        # than inheriting one.
        in_run = previous_time is not None and time - previous_time <= STREAM_SECONDS
        previous_time = time
        if in_run:
            foot ^= 1
        if len(stepped) != 1:
            continue
        landed = next(iter(stepped))
        if not in_run:
            foot = 1 if landed == _RIGHT_PANEL else 0
        elif (foot == 0 and landed == _RIGHT_PANEL) or (foot == 1 and landed == _LEFT_PANEL):
            crossings += 1
    return crossings


def _foot_search(occupied: Sequence[tuple[float, set[int]]]) -> bool:
    """
    Decide whether two feet can cover every row, and count the crossovers.

    Feet must cover as many of a row's panels as they can reach, which is two for a chord and all of
    them otherwise; anything left over is attributed to hands.

    Parameters
    ----------
    occupied : :py:class:`~collections.abc.Sequence`
        Time and occupied panels for each row.

    Returns
    -------
    bool
        Whether a valid assignment exists.
    """
    costs = dict.fromkeys(FOOT_STATES, 0.0)
    previous_time = occupied[0][0] if occupied else 0.0
    for time, panels in occupied:
        needed = min(len(panels), MAX_FEET)
        gap = max(time - previous_time, 0.0)
        updated: dict[tuple[int, int], float] = {}
        for state in FOOT_STATES:
            if len(panels & set(state)) != needed:
                continue
            best = min(
                (costs[origin] + _transition_cost(origin, state, gap) for origin in costs),
                default=_IMPOSSIBLE,
            )
            if best < _IMPOSSIBLE:
                updated[state] = best
        if not updated:
            return False
        costs = updated
        previous_time = time
    return True


def _classify(report: dict[str, float | int | bool]) -> tuple[Style, tuple[str, ...]]:
    """
    Turn measurements into a style verdict.

    Parameters
    ----------
    report : dict[str, float | int | bool]
        Measurements gathered by :func:`analyze_rows`.

    Returns
    -------
    tuple[Style, tuple[str, ...]]
        The verdict and the reasons supporting anything beyond feet alone.
    """
    reasons: list[str] = []
    if report['impossible_rows']:
        reasons.append(f'{report["impossible_rows"]} rows need more than four panels at once')
    if not report['geometrically_possible']:
        reasons.append('no two-foot assignment covers the chart')
    if report['fastest_jack'] < FAST_JACK_SECONDS:
        reasons.append(f'a panel repeats every {float(report["fastest_jack"]) * 1000:.0f} ms')
    if report['sustained_nps'] > MAX_SUSTAINED_NPS:
        reasons.append(
            f'{report["sustained_nps"]:.1f} notes per second held for '
            f'{SUSTAINED_WINDOW_SECONDS:.0f} s exceeds pad stamina'
        )
    if report['burst_nps'] > MAX_BURST_NPS:
        reasons.append(f'bursts reach {report["burst_nps"]:.1f} notes per second')
    if reasons:
        return 'keyboard', tuple(reasons)
    if report['chord_rows']:
        return 'hands', (f'{report["chord_rows"]} rows need three or four panels together',)
    return 'feet', ()


def analyze_rows(rows: Sequence[tuple[float, Sequence[int]]]) -> PlayabilityReport:
    """
    Decide how a chart can be performed.

    Parameters
    ----------
    rows : :py:class:`~collections.abc.Sequence`
        Time in seconds and four panel codes for each non-empty row, ordered by time.

    Returns
    -------
    PlayabilityReport
        The verdict together with the measurements behind it.
    """
    occupied = _required_panels(rows)
    if not occupied:
        return PlayabilityReport(
            burst_nps=0.0,
            chord_rows=0,
            crossovers=0,
            fastest_jack=math.inf,
            geometrically_possible=True,
            impossible_rows=0,
            max_simultaneous=0,
            reasons=(),
            style='feet',
            sustained_nps=0.0,
        )
    times = [time for time, _ in occupied]
    max_simultaneous = max(len(panels) for _, panels in occupied)
    chord_rows = sum(1 for _, panels in occupied if MAX_FEET < len(panels) <= MAX_LIMBS)
    impossible_rows = sum(1 for _, panels in occupied if len(panels) > MAX_LIMBS)
    # Only a fresh tap counts. A panel under a freeze is occupied on every row until its tail,
    # which is what pins a foot there, but the foot is resting rather than retapping. Reading the
    # jack rate off that set reports a repeat on every row a freeze spans, and a chart with a
    # freeze under a sixteenth run then looks unplayable when it is nothing of the sort.
    last_seen: dict[int, float] = {}
    fastest_jack = math.inf
    for time, codes in rows:
        for panel, code in enumerate(codes):
            if code not in _STEP_CODES:
                continue
            if (previous := last_seen.get(panel)) is not None:
                fastest_jack = min(fastest_jack, time - previous)
            last_seen[panel] = time
    possible = _foot_search(occupied)
    crossovers = _crossed_steps(rows)
    measurements: dict[str, float | int | bool] = {
        'burst_nps': _window_rate(times, BURST_WINDOW_SECONDS),
        'chord_rows': chord_rows,
        'fastest_jack': fastest_jack,
        'geometrically_possible': possible,
        'impossible_rows': impossible_rows,
        'sustained_nps': _window_rate(times, SUSTAINED_WINDOW_SECONDS),
    }
    style, reasons = _classify(measurements)
    return PlayabilityReport(
        burst_nps=float(measurements['burst_nps']),
        chord_rows=chord_rows,
        crossovers=crossovers,
        fastest_jack=fastest_jack,
        geometrically_possible=possible,
        impossible_rows=impossible_rows,
        max_simultaneous=max_simultaneous,
        reasons=reasons,
        style=style,
        sustained_nps=float(measurements['sustained_nps']),
    )
