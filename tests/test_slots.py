"""Tests for deciding which grid slots carry a note."""

from __future__ import annotations

import numpy as np
import pytest

from smlab.encoder import MEASURE_SLOTS
from smlab.generate import GenerationConfig
from smlab.slots import choose_slots, fill, refill, seed_pulse, silence_threshold, tidy_rests
from smlab.timing import TimingData

_BPM = 150.0
_TIMING = TimingData.constant(_BPM, 0.0)
_QUARTER = 12


def _scores(measures: int, loudness: list[float] | None = None) -> np.ndarray:
    """Score every slot, optionally giving each measure its own loudness."""
    scores = np.zeros(measures * MEASURE_SLOTS, dtype=np.float32)
    levels = loudness if loudness is not None else [1.0] * measures
    for measure, level in enumerate(levels):
        start = measure * MEASURE_SLOTS
        scores[start : start + MEASURE_SLOTS] = level
        # Give the downbeat the edge so the seeded slot is predictable.
        scores[start] = level + 0.5
    return scores


def test_one_note_is_seeded_in_every_measure_that_plays() -> None:
    scores = _scores(8)
    taken = np.zeros(len(scores), dtype=np.bool_)
    playable, seeded = seed_pulse(scores, taken, wanted=64)
    assert len(seeded) == 8
    assert seeded == [measure * MEASURE_SLOTS for measure in range(8)]
    assert playable.all()


def test_a_measure_far_quieter_than_the_song_is_left_silent() -> None:
    # An intro or a breakdown rests; a song that never lets up stays full.
    scores = _scores(8, [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, -100.0])
    taken = np.zeros(len(scores), dtype=np.bool_)
    playable, seeded = seed_pulse(scores, taken, wanted=64)
    assert len(seeded) == 7
    assert not playable[7 * MEASURE_SLOTS :].any()
    assert playable[: 7 * MEASURE_SLOTS].all()


def test_a_song_too_short_to_have_measures_is_not_seeded() -> None:
    scores = _scores(1)
    taken = np.zeros(len(scores), dtype=np.bool_)
    playable, seeded = seed_pulse(scores, taken, wanted=64)
    assert seeded == []
    assert playable.all()


def test_a_budget_smaller_than_the_song_is_not_seeded() -> None:
    # One note per measure would already overrun the note count asked for.
    scores = _scores(8)
    taken = np.zeros(len(scores), dtype=np.bool_)
    _, seeded = seed_pulse(scores, taken, wanted=3)
    assert seeded == []


def test_a_song_with_no_music_at_all_is_not_seeded() -> None:
    scores = np.full(8 * MEASURE_SLOTS, -np.inf, dtype=np.float32)
    taken = np.zeros(len(scores), dtype=np.bool_)
    playable, seeded = seed_pulse(scores, taken, wanted=64)
    assert seeded == []
    assert playable.all()


def test_a_song_of_uniform_loudness_rests_nowhere() -> None:
    # With no spread there is no such thing as an anomalously quiet measure.
    assert silence_threshold(np.full(8, 3.0, dtype=np.float32)) == pytest.approx(2.0)


def test_the_silence_threshold_sits_below_the_typical_measure() -> None:
    loudest = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    assert silence_threshold(loudest) < float(np.median(loudest))


def test_filling_takes_the_highest_scores_first() -> None:
    scores = np.arange(48, dtype=np.float32)
    order = np.argsort(scores)[::-1]
    taken = np.zeros(48, dtype=np.bool_)
    family = np.ones(48, dtype=np.bool_)
    assert fill(order, taken, family, wanted=3, gap=0) == [47, 46, 45]


def test_filling_respects_the_minimum_spacing() -> None:
    scores = np.arange(48, dtype=np.float32)
    order = np.argsort(scores)[::-1]
    taken = np.zeros(48, dtype=np.bool_)
    family = np.ones(48, dtype=np.bool_)
    added = fill(order, taken, family, wanted=3, gap=4)
    assert all(abs(a - b) >= 4 for a in added for b in added if a != b)


def test_filling_skips_slots_outside_the_family() -> None:
    order = np.arange(48, dtype=np.int64)[::-1]
    taken = np.zeros(48, dtype=np.bool_)
    quarters = (np.arange(48) % _QUARTER) == 0
    assert all(slot % _QUARTER == 0 for slot in fill(order, taken, quarters, wanted=4, gap=0))


def test_filling_stops_when_it_has_enough() -> None:
    order = np.arange(48, dtype=np.int64)
    taken = np.zeros(48, dtype=np.bool_)
    assert len(fill(order, taken, np.ones(48, dtype=np.bool_), wanted=0, gap=0)) == 0


def test_a_chart_with_no_thin_measures_is_left_alone() -> None:
    chosen = [
        measure * MEASURE_SLOTS + step * _QUARTER for measure in range(4) for step in range(4)
    ]
    order = np.arange(4 * MEASURE_SLOTS, dtype=np.int64)
    taken = np.zeros(4 * MEASURE_SLOTS, dtype=np.bool_)
    playable = np.ones(4 * MEASURE_SLOTS, dtype=np.bool_)
    assert tidy_rests(list(chosen), order, taken, 0, playable) == sorted(chosen)


def test_an_empty_chart_tidies_to_nothing() -> None:
    order = np.arange(48, dtype=np.int64)
    taken = np.zeros(48, dtype=np.bool_)
    assert tidy_rests([], order, taken, 0, np.ones(48, dtype=np.bool_)) == []


def test_a_barely_used_measure_is_emptied_onto_its_downbeat() -> None:
    # A rest that begins wherever the previous measure trailed off starts off
    # the beat, so the downbeat is kept and everything after it dropped.
    # At most a tenth of the measures may rest, so the song has to be long
    # enough for that tenth to round up to one.
    measures = 16
    slots = measures * MEASURE_SLOTS
    busy = [
        measure * MEASURE_SLOTS + step * 3 for measure in range(measures - 1) for step in range(16)
    ]
    stray = (measures - 1) * MEASURE_SLOTS + 30
    chosen = [*busy, stray]
    order = np.argsort(np.zeros(slots, dtype=np.float32))[::-1]
    taken = np.zeros(slots, dtype=np.bool_)
    for slot in chosen:
        taken[slot] = True
    tidied = tidy_rests(list(chosen), order, taken, 0, np.ones(slots, dtype=np.bool_))
    assert (measures - 1) * MEASURE_SLOTS in tidied
    assert stray not in tidied
    assert len(tidied) == len(chosen)


def test_notes_freed_by_a_rest_go_to_measures_still_playing() -> None:
    slots = 4 * MEASURE_SLOTS
    order = np.arange(slots, dtype=np.int64)
    taken = np.zeros(slots, dtype=np.bool_)
    playable = np.ones(slots, dtype=np.bool_)
    added = refill(3, order, taken, {0}, 0, playable)
    assert len(added) == 3
    assert all(slot // MEASURE_SLOTS != 0 for slot in added)


def test_refilling_never_touches_a_resting_measure() -> None:
    slots = 2 * MEASURE_SLOTS
    order = np.arange(slots, dtype=np.int64)
    taken = np.zeros(slots, dtype=np.bool_)
    playable = np.ones(slots, dtype=np.bool_)
    # Everything but the resting measure is already spoken for.
    taken[MEASURE_SLOTS:] = True
    assert refill(5, order, taken, {0}, 0, playable) == []


def test_refilling_honours_the_minimum_spacing() -> None:
    slots = 4 * MEASURE_SLOTS
    order = np.arange(slots, dtype=np.int64)
    taken = np.zeros(slots, dtype=np.bool_)
    added = refill(4, order, taken, set(), 6, np.ones(slots, dtype=np.bool_))
    assert all(abs(a - b) >= 6 for a in added for b in added if a != b)


def test_refilling_skips_a_measure_that_never_plays() -> None:
    slots = 2 * MEASURE_SLOTS
    order = np.arange(slots, dtype=np.int64)
    taken = np.zeros(slots, dtype=np.bool_)
    playable = np.ones(slots, dtype=np.bool_)
    playable[:MEASURE_SLOTS] = False
    assert all(slot >= MEASURE_SLOTS for slot in refill(3, order, taken, set(), 0, playable))


def test_the_chosen_slots_come_back_in_order() -> None:
    chosen = choose_slots(_scores(16), _TIMING, GenerationConfig(nps=4.0))
    assert chosen == sorted(chosen)
    assert chosen


def test_a_faster_rating_chooses_more_slots() -> None:
    scores = _scores(16)
    slow = choose_slots(scores, _TIMING, GenerationConfig(nps=2.0))
    fast = choose_slots(scores, _TIMING, GenerationConfig(nps=8.0))
    assert len(fast) > len(slow)


def test_a_keyboard_chart_is_not_held_to_the_jack_limit() -> None:
    scores = _scores(16)
    feet = choose_slots(scores, _TIMING, GenerationConfig(nps=12.0))
    keys = choose_slots(scores, _TIMING, GenerationConfig(nps=12.0, style='keyboard'))
    assert len(keys) >= len(feet)


def test_at_least_one_slot_is_always_kept() -> None:
    # A chart with no notes at all is not a chart.
    assert len(choose_slots(_scores(2), _TIMING, GenerationConfig(density=0.0, nps=1e-9))) == 1


def test_a_song_with_no_slots_chooses_nothing() -> None:
    assert choose_slots(np.zeros(0, dtype=np.float32), _TIMING, GenerationConfig()) == []


def test_notes_stay_on_the_sixteenth_grid_by_default() -> None:
    # Ranking every subdivision together scatters notes onto twenty-fourths,
    # which is what a chart full of stray off-colour arrows is.
    assert all(
        slot % _QUARTER % 3 == 0
        for slot in choose_slots(_scores(16), _TIMING, GenerationConfig(nps=8.0))
    )


def test_a_rested_measure_that_never_plays_keeps_no_downbeat() -> None:
    # A measure the seeding already silenced has nothing to hit the downbeat
    # of, so emptying it must not put a note back.
    measures = 16
    slots = measures * MEASURE_SLOTS
    last = measures - 1
    chosen = [
        *(measure * MEASURE_SLOTS + step * 3 for measure in range(last) for step in range(16)),
        last * MEASURE_SLOTS + 30,
    ]
    order = np.argsort(np.zeros(slots, dtype=np.float32))[::-1]
    taken = np.zeros(slots, dtype=np.bool_)
    for slot in chosen:
        taken[slot] = True
    playable = np.ones(slots, dtype=np.bool_)
    playable[last * MEASURE_SLOTS :] = False
    tidied = tidy_rests(list(chosen), order, taken, 0, playable)
    assert last * MEASURE_SLOTS not in tidied


def test_slots_with_no_music_in_them_are_never_chosen() -> None:
    # Which measures rest is otherwise read off the placement scores, and over
    # dead air the network still returns a number. The audio says so outright.
    measures = 16
    scores = _scores(measures)
    loudness = np.full(len(scores), -30.0, dtype=np.float32)
    silent = 12 * MEASURE_SLOTS
    loudness[silent:] = -80.0
    chosen = choose_slots(scores, _TIMING, GenerationConfig(nps=6.0), loudness)
    assert chosen
    assert max(chosen) < silent


def test_a_fade_that_has_not_reached_the_floor_still_plays() -> None:
    scores = _scores(8)
    loudness = np.full(len(scores), -65.0, dtype=np.float32)
    assert choose_slots(scores, _TIMING, GenerationConfig(nps=4.0), loudness)


def test_loudness_shorter_than_the_grid_leaves_the_rest_playable() -> None:
    # The two grids are derived separately, so their lengths need not agree.
    scores = _scores(8)
    loudness = np.full(len(scores) - 5, -30.0, dtype=np.float32)
    assert choose_slots(scores, _TIMING, GenerationConfig(nps=4.0), loudness)


def test_a_song_that_is_silent_throughout_is_charted_at_all() -> None:
    # Nothing is audible, so nothing can be placed rather than everything.
    scores = _scores(8)
    loudness = np.full(len(scores), -80.0, dtype=np.float32)
    assert choose_slots(scores, _TIMING, GenerationConfig(nps=4.0), loudness) == []
