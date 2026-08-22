"""Tests for the rulebook a chart is decoded under."""

from __future__ import annotations

from smlab.constraints import (
    Budget,
    allowed,
    crowded,
    on_grid,
    panel_bias,
    panel_membership,
    subdivision_quota,
    thin_measures,
)
from smlab.encoder import MEASURE_SLOTS
from smlab.generate import GenerationConfig
from smlab.vocab import Vocabulary, encode_row
import numpy as np
import pytest

_LEFT, _DOWN, _UP, _RIGHT = range(4)
_PANEL_INDEX = {'L': _LEFT, 'D': _DOWN, 'U': _UP, 'R': _RIGHT}
_NO_RUN = 10.0
"""A gap far too wide for the notes either side of it to be a run."""
_ROOMY = 10.0
"""A gap long enough that no timing rule bites."""


@pytest.fixture
def vocabulary() -> Vocabulary:
    """Build a vocabulary holding the patterns these tests reason about."""
    rows = (
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
        (1, 0, 0, 1),  # a jump
        (0, 1, 1, 0),  # a jump
        (2, 0, 0, 0),  # a freeze head
        (3, 0, 0, 0),  # its tail
        (4, 0, 0, 0),  # a roll head
        (5, 0, 0, 0),  # a mine
        (1, 1, 1, 0),  # three at once
    )
    return Vocabulary([encode_row(row) for row in rows])


def _permitted(vocabulary: Vocabulary, mask: np.ndarray) -> set[tuple[int, ...]]:
    return {tuple(vocabulary.panels_of(int(i))) for i in np.flatnonzero(mask)}


def test_the_empty_pattern_is_never_offered() -> None:
    # Where the rests go is the placement head's decision. A vocabulary entry
    # for silence lets the selection head overrule it, which cost 28 per cent
    # of a chart's notes when it was reachable.
    vocabulary = Vocabulary([encode_row((0, 0, 0, 0)), encode_row((1, 0, 0, 0))])
    mask = allowed(vocabulary, GenerationConfig(), {}, frozenset(), _ROOMY, _ROOMY, frozenset())
    assert (0, 0, 0, 0) not in _permitted(vocabulary, mask)
    assert (1, 0, 0, 0) in _permitted(vocabulary, mask)


def test_mines_and_rolls_are_off_by_default(vocabulary: Vocabulary) -> None:
    mask = allowed(vocabulary, GenerationConfig(), {}, frozenset(), _ROOMY, _ROOMY, frozenset())
    permitted = _permitted(vocabulary, mask)
    assert (5, 0, 0, 0) not in permitted
    assert (4, 0, 0, 0) not in permitted


def test_mines_and_rolls_can_be_asked_for(vocabulary: Vocabulary) -> None:
    config = GenerationConfig(mines=True, rolls=True)
    mask = allowed(vocabulary, config, {}, frozenset(), _ROOMY, _ROOMY, frozenset())
    permitted = _permitted(vocabulary, mask)
    assert (5, 0, 0, 0) in permitted
    assert (4, 0, 0, 0) in permitted


def test_a_panel_cannot_be_retapped_faster_than_a_foot_moves(vocabulary: Vocabulary) -> None:
    # Real feet-style charts put only 0.4 per cent of their same-panel repeats
    # under 130 ms, so a sixteenth jack at speed is not something to generate.
    mask = allowed(vocabulary, GenerationConfig(), {}, frozenset({_LEFT}), 0.10, 0.10, frozenset())
    permitted = _permitted(vocabulary, mask)
    assert (1, 0, 0, 0) not in permitted
    assert (0, 1, 0, 0) in permitted


def test_a_keyboard_may_retap_as_fast_as_it_likes(vocabulary: Vocabulary) -> None:
    config = GenerationConfig(style='keyboard')
    mask = allowed(vocabulary, config, {}, frozenset({_LEFT}), 0.02, 0.02, frozenset())
    assert (1, 0, 0, 0) in _permitted(vocabulary, mask)


def test_a_jump_needs_room_on_both_sides(vocabulary: Vocabulary) -> None:
    # A jump is as hard to leave as to reach, so the tighter of the two gaps
    # decides. Passing a wide gap in and a narrow one out must still refuse.
    tight = allowed(vocabulary, GenerationConfig(), {}, frozenset(), _ROOMY, 0.05, frozenset())
    assert (1, 0, 0, 1) not in _permitted(vocabulary, tight)
    roomy = allowed(vocabulary, GenerationConfig(), {}, frozenset(), _ROOMY, _ROOMY, frozenset())
    assert (1, 0, 0, 1) in _permitted(vocabulary, roomy)


def test_feet_never_get_three_panels_at_once(vocabulary: Vocabulary) -> None:
    mask = allowed(vocabulary, GenerationConfig(), {}, frozenset(), _ROOMY, _ROOMY, frozenset())
    assert (1, 1, 1, 0) not in _permitted(vocabulary, mask)


def test_hands_may_take_three(vocabulary: Vocabulary) -> None:
    config = GenerationConfig(style='hands')
    mask = allowed(vocabulary, config, {}, frozenset(), _ROOMY, _ROOMY, frozenset())
    assert (1, 1, 1, 0) in _permitted(vocabulary, mask)


def test_a_held_panel_is_out_of_reach_and_its_tail_is_not(vocabulary: Vocabulary) -> None:
    held = {_LEFT: 0}
    mask = allowed(vocabulary, GenerationConfig(), held, frozenset(), _ROOMY, _ROOMY, frozenset())
    permitted = _permitted(vocabulary, mask)
    assert (1, 0, 0, 0) not in permitted
    assert (3, 0, 0, 0) in permitted


def test_an_orphan_tail_is_refused(vocabulary: Vocabulary) -> None:
    # A tail with no open freeze stops the file loading at all.
    mask = allowed(vocabulary, GenerationConfig(), {}, frozenset(), _ROOMY, _ROOMY, frozenset())
    assert (3, 0, 0, 0) not in _permitted(vocabulary, mask)


def test_a_crossed_panel_can_be_barred(vocabulary: Vocabulary) -> None:
    mask = allowed(
        vocabulary, GenerationConfig(), {}, frozenset(), _ROOMY, _ROOMY, frozenset({_RIGHT})
    )
    permitted = _permitted(vocabulary, mask)
    assert (0, 0, 0, 1) not in permitted
    assert (1, 0, 0, 0) in permitted


@pytest.mark.parametrize('relax', [2, 3])
def test_relaxing_gives_up_preferences_but_never_the_jack_limit(
    vocabulary: Vocabulary, relax: int
) -> None:
    # When every rule bites at once something must still be placed, but a chart
    # that retaps a panel in 100 ms cannot be danced, so that one is never
    # surrendered however far the others are relaxed.
    mask = allowed(
        vocabulary,
        GenerationConfig(),
        {},
        frozenset({_LEFT}),
        0.10,
        0.10,
        frozenset(),
        relax=relax,
    )
    assert (1, 0, 0, 0) not in _permitted(vocabulary, mask)


def test_off_grid_slots_are_out_of_reach() -> None:
    # Charts sit on the sixteenth grid: 97.3 per cent of corpus notes are
    # quarters, eighths or sixteenths and 74 per cent of charts never leave it.
    scores = np.ones(MEASURE_SLOTS, dtype=np.float32)
    ranked = on_grid(scores, triplets=False)
    assert np.isfinite(ranked[0])
    assert np.isfinite(ranked[3])
    assert np.isfinite(ranked[6])
    assert not np.isfinite(ranked[1])
    assert not np.isfinite(ranked[4])


def test_triplets_open_the_twelfth_grid() -> None:
    scores = np.ones(MEASURE_SLOTS, dtype=np.float32)
    ranked = on_grid(scores, triplets=True)
    assert np.isfinite(ranked[4])
    assert not np.isfinite(ranked[1])


@pytest.mark.parametrize(('rate', 'coarsest'), [(2.5, True), (9.5, False)])
def test_a_slower_chart_leans_on_quarters(rate: float, *, coarsest: bool) -> None:
    quarters, eighths, sixteenths = subdivision_quota(rate, 1000)
    assert quarters + eighths + sixteenths == 1000
    assert (quarters > 700) is coarsest


def test_a_freeze_needs_room_before_the_next_note() -> None:
    # Three quarters of corpus freezes have the other foot playing nothing at
    # all, which only holds if the freeze ends before the next note lands.
    tight = crowded([0, 3, 6, 60, 120])
    assert bool(tight[0])
    assert bool(tight[1])
    assert not bool(tight[3])


def test_the_panel_bias_pushes_away_from_an_over_used_panel(vocabulary: Vocabulary) -> None:
    membership = panel_membership(vocabulary)
    usage = np.zeros(4, dtype=np.float64)
    usage[_DOWN] = 40.0
    bias = panel_bias(membership, usage)
    down = next(i for i in range(len(vocabulary)) if vocabulary.panels_of(i) == (0, 1, 0, 0))
    left = next(i for i in range(len(vocabulary)) if vocabulary.panels_of(i) == (1, 0, 0, 0))
    assert bias[down] < bias[left]


def test_the_panel_record_fades() -> None:
    # Balancing lifetime totals leaves a single measure free to sit on one
    # arrow, since the totals barely move. The record has to be recent.
    budget = Budget()
    for _ in range(200):
        budget.record(frozenset({_DOWN}), [0, 1, 0, 0], in_run=False)
    early = float(budget.usage[_DOWN])
    for _ in range(200):
        budget.record(frozenset({_LEFT}), [1, 0, 0, 0], in_run=False)
    assert float(budget.usage[_DOWN]) < early / 10


def test_freezes_and_jumps_run_out(vocabulary: Vocabulary) -> None:
    budget = Budget()
    for _ in range(50):
        budget.record(frozenset({_LEFT}), [1, 0, 0, 0], in_run=False)
    assert not budget.freezes_spent(0.04)
    assert not budget.jumps_spent(0.08)
    for _ in range(20):
        budget.record(frozenset({_LEFT}), [2, 0, 0, 0], in_run=False)
        budget.record(frozenset({_LEFT, _RIGHT}), [1, 0, 0, 1], in_run=False)
    assert budget.freezes_spent(0.04)
    assert budget.jumps_spent(0.08)


def test_a_run_of_one_arrow_is_capped() -> None:
    # A keyboard has no jack limit to break up a run, and nothing else stops
    # one: a chart measured with no cap put nineteen identical arrows in a row.
    budget = Budget()
    assert budget.stale() == frozenset()
    for _ in range(10):
        budget.record(frozenset({_DOWN}), [0, 1, 0, 0], in_run=True)
    assert budget.stale() == frozenset({_DOWN})
    budget.record(frozenset({_UP}), [0, 0, 1, 0], in_run=True)
    assert budget.stale() == frozenset()


def test_only_the_emptiest_measures_are_rested() -> None:
    # Charts rated twelve to eighteen leave no measure empty inside their body
    # at the median, so resting has to stay rare.
    busy = [measure * MEASURE_SLOTS + slot for measure in range(20) for slot in (0, 12, 24, 36)]
    thin = [21 * MEASURE_SLOTS]
    assert thin_measures([*busy, *thin]) == {21}


def _stream(panels: str, gap: float, seconds_per_slot: float, share: float = 1.0) -> list[bool]:
    """
    Walk a run of single notes through a budget.

    Reports whether each step would have been barred as a crossover. The
    opening note starts the run rather than joining one, as it does when a
    chart is decoded, and the share defaults high so that the cap on
    consecutive crossings is what the result reflects.
    """
    budget = Budget()
    barred = []
    for index, name in enumerate(panels.split()):
        landed = _PANEL_INDEX[name]
        in_run = budget.enter_run(gap if index else _NO_RUN, seconds_per_slot)
        barred.append(landed in budget.crossed(share) | budget.overrun(share))
        codes = [0, 0, 0, 0]
        codes[landed] = 1
        budget.record(frozenset({landed}), codes, in_run=in_run)
    return barred


def test_a_sixteenth_stream_never_crosses_twice_running() -> None:
    # Up then down leaves the left foot due, so landing on right crosses, and
    # left on the right foot straight after would cross again. At a sixteenth
    # there is no time to recover between them, so the second one is barred.
    sixteenth = 0.1
    barred = _stream('U D R L R L', sixteenth, sixteenth / 3.0)
    assert barred[2] is False
    assert barred[3] is True


def test_an_eighth_run_may_cross_twice_but_not_three_times() -> None:
    # Slower crossovers read as a flourish rather than a scramble, and the
    # corpus writes two in a row for 14 per cent of its crossed stretches.
    eighth = 0.15
    barred = _stream('U D R L R L', eighth, eighth / 12.0)
    assert barred[2] is False
    assert barred[3] is False
    assert barred[4] is True


def test_the_crossover_budget_applies_to_a_keyboard_too() -> None:
    # A keyboard has no legs to cross, but the shape is as awkward under four
    # fingers, and the rule used to be skipped for it entirely.
    budget = Budget()
    assert budget.enter_run(0.05, 0.02) is True


def test_a_gap_too_wide_to_be_a_run_is_not_one() -> None:
    assert Budget().enter_run(1.0, 0.02) is False


def test_nothing_is_barred_while_budget_remains() -> None:
    budget = Budget()
    assert budget.crossed(0.5) == frozenset()
    assert budget.overrun(0.5) == frozenset()


def test_an_allowance_of_zero_bars_crossing_from_the_very_first_step() -> None:
    # Rationing by share always lets the first one through, since no share of
    # nothing has been spent yet. Asking for none has to mean none.
    barred = _stream('U D R L', 0.1, 0.1 / 3.0, share=0.0)
    assert barred[2] is True


def test_an_allowance_of_zero_bars_crossing_at_eighth_speed_too() -> None:
    barred = _stream('U D R L', 0.15, 0.15 / 12.0, share=0.0)
    assert barred[2] is True
