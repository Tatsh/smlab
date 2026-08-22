"""Tests for deciding how a chart can be performed."""

from __future__ import annotations

from smlab.playability import FOOT_STATES, MAX_FEET, analyze_rows, is_crossover
import pytest

TAP = 1
HOLD = 2
TAIL = 3


def _alternating(count: int, interval: float) -> list[tuple[float, list[int]]]:
    panels = (0, 3, 1, 2)
    return [
        (index * interval, [TAP if panel == panels[index % 4] else 0 for panel in range(4)])
        for index in range(count)
    ]


def test_alternating_singles_are_danceable() -> None:
    report = analyze_rows(_alternating(64, 0.25))
    assert report.style == 'feet'
    assert report.max_simultaneous == 1
    assert report.chord_rows == 0
    assert report.reasons == ()


def test_two_panel_jumps_are_danceable() -> None:
    rows = [(index * 0.5, [TAP, 0, 0, TAP]) for index in range(16)]
    assert analyze_rows(rows).style == 'feet'


def test_three_panel_rows_need_hands() -> None:
    rows = [(index * 0.5, [TAP, TAP, TAP, 0]) for index in range(16)]
    report = analyze_rows(rows)
    assert report.style == 'hands'
    assert report.max_simultaneous == 3
    assert report.chord_rows == 16


def test_quads_need_hands_but_remain_pad_playable() -> None:
    rows = [(index * 1.0, [TAP, TAP, TAP, TAP]) for index in range(8)]
    report = analyze_rows(rows)
    assert report.style == 'hands'
    assert report.pad_playable


def test_impossibly_fast_repeats_are_keyboard_only() -> None:
    rows = [(index * 0.02, [TAP, 0, 0, 0]) for index in range(64)]
    report = analyze_rows(rows)
    assert report.style == 'keyboard'
    assert not report.pad_playable
    assert any('repeats' in reason for reason in report.reasons)


def test_superhuman_sustained_rate_is_keyboard_only() -> None:
    # Twenty notes per second held for twenty seconds, alternating panels so
    # that geometry alone would allow it.
    report = analyze_rows(_alternating(400, 0.05))
    assert report.style == 'keyboard'
    assert report.sustained_nps > 12.0
    assert any('stamina' in reason for reason in report.reasons)


def test_hold_on_two_panels_blocks_a_third_note() -> None:
    rows: list[tuple[float, list[int]]] = [
        (0.0, [HOLD, 0, 0, HOLD]),
        (1.0, [0, TAP, 0, 0]),
        (2.0, [TAIL, 0, 0, TAIL]),
    ]
    report = analyze_rows(rows)
    assert report.max_simultaneous == 3
    assert report.style != 'feet'


def test_empty_chart_is_trivially_danceable() -> None:
    report = analyze_rows([])
    assert report.style == 'feet'
    assert report.max_simultaneous == 0


@pytest.mark.parametrize(
    ('left', 'right', 'expected'),
    [
        (0, 3, False),
        (3, 0, True),
        (1, 2, False),
        (2, 1, False),
        (0, 1, False),
        (3, 1, True),
    ],
)
def test_crossover_detection(left: int, right: int, *, expected: bool) -> None:
    assert is_crossover(left, right) is expected


def test_note_rate_scales_with_tempo() -> None:
    slow = analyze_rows(_alternating(200, 0.125))
    fast = analyze_rows(_alternating(200, 0.0625))
    assert fast.sustained_nps > slow.sustained_nps
    assert slow.style == 'feet'
    assert fast.style == 'keyboard'


def test_a_chart_no_two_feet_can_cover_is_keyboard_only() -> None:
    # Both feet would have to jump across the pad every millisecond.
    rows = [(index * 0.001, codes) for index, codes in enumerate([[1, 1, 0, 0], [0, 0, 1, 1]] * 6)]
    report = analyze_rows(rows)
    assert not report.geometrically_possible
    assert report.style == 'keyboard'
    assert 'no two-foot assignment covers the chart' in report.reasons


def test_a_panel_repeating_faster_than_a_foot_moves_is_reported() -> None:
    rows = [(index * 0.05, [1, 0, 0, 0]) for index in range(12)]
    report = analyze_rows(rows)
    assert report.fastest_jack == pytest.approx(0.05)
    assert any('a panel repeats every 50 ms' in reason for reason in report.reasons)


def test_every_row_a_crossed_stance_covers_a_clean_one_covers_too() -> None:
    # This is why the search never has to report a forced crossover: the foot
    # states are symmetric under swapping, so a clean stance is reachable
    # wherever a crossed one is.
    for panels in ({0}, {1}, {2}, {3}, {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}):
        needed = min(len(panels), MAX_FEET)
        matching = [state for state in FOOT_STATES if len(panels & set(state)) == needed]
        assert any(not is_crossover(*state) for state in matching)


def test_a_row_needing_more_than_four_panels_is_reported() -> None:
    # Wider steps types reach this: no player has five limbs.
    report = analyze_rows([(0.0, [1, 1, 1, 1, 1, 1]), (1.0, [1, 0, 0, 0, 0, 0])])
    assert report.impossible_rows == 1
    assert 'rows need more than four panels at once' in report.reasons[0]
    assert report.style == 'keyboard'
