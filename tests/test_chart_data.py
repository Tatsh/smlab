"""Tests for the training windows drawn from the stem cache."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
import json

from smlab.cache import cache_path_for
from smlab.chart_data import (
    MIRRORS,
    WINDOW_MEASURES,
    WINDOW_STEPS,
    ChartWindows,
    measure_prior,
    pack_scales,
)
from smlab.encoder import MAX_METER, MAX_RATE, MEASURE_SLOTS
from smlab.features import TOTAL_CHANNELS
from smlab.vocab import Vocabulary, encode_row
import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from smlab.typing import SongRecord

_SLOTS = WINDOW_MEASURES * MEASURE_SLOTS
"""Exactly one window, so the random start is always zero and notes always land."""
_VOCABULARY = Vocabulary([encode_row(row) for row in ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1))])
_TRAINING = '/corpus/0.sm'
"""A corpus path that hashes into the training split."""
_HELD_OUT = '/corpus/18.sm'
"""A corpus path that hashes into the held-out split."""


def _record(pack: str, simfile: str, meters: tuple[int, ...] = (9,), **extra: Any) -> SongRecord:
    return cast(
        'SongRecord',
        {
            'charts': [
                {'difficulty': 'Challenge', 'meter': meter, 'rows': 100, 'style': 'feet'}
                for meter in meters
            ],
            'pack': pack,
            'primary_bpm': 150.0,
            'simfile': simfile,
            **extra,
        },
    )


def _write_entry(root: Path, record: SongRecord, *, rows: int = 64) -> Path:
    path = cache_path_for(root, record['simfile'])
    path.parent.mkdir(parents=True, exist_ok=True)
    slots = np.arange(rows, dtype=np.int32) * 12
    arrays: dict[str, Any] = {
        'features': np.zeros((2 * _SLOTS, TOTAL_CHANNELS), dtype=np.float16),
        'slots_0': slots,
        'panels_0': np.tile(np.array([[1, 0, 0, 0]], dtype=np.uint8), (rows, 1)),
        'meta': np.asarray(json.dumps([{'difficulty': 'Challenge', 'index': 0, 'meter': 9}])),
    }
    np.savez(path, **arrays)
    return path


def test_a_pack_that_never_exceeds_ten_is_on_the_classic_scale() -> None:
    scales = pack_scales([_record('Classic', 'a.sm', meters=(3, 7, 10))])
    assert scales['Classic'] == 0


def test_a_pack_reaching_thirteen_is_on_the_modern_scale() -> None:
    scales = pack_scales([_record('Modern', 'a.sm', meters=(9, 16))])
    assert scales['Modern'] == 1


def test_a_pack_that_straddles_the_two_is_left_out() -> None:
    # Eleven or twelve says nothing: it is too high for the classic scale and
    # too low to prove the modern one, so the conditioning would be a guess.
    assert pack_scales([_record('Ambiguous', 'a.sm', meters=(11,))]) == {}


def test_windows_are_drawn_for_every_chart_in_the_split(tmp_path: Path) -> None:
    record = _record('Modern', _TRAINING, meters=(9, 16))
    _write_entry(tmp_path, record)
    windows = ChartWindows(tmp_path, [record], _VOCABULARY)
    assert len(windows) + len(ChartWindows(tmp_path, [record], _VOCABULARY, validation=True)) == 2


def test_a_song_with_no_cache_entry_is_skipped(tmp_path: Path) -> None:
    record = _record('Modern', '/corpus/missing.sm', meters=(16,))
    assert len(ChartWindows(tmp_path, [record], _VOCABULARY)) == 0


def test_a_song_whose_pack_has_no_scale_is_skipped(tmp_path: Path) -> None:
    record = _record('Ambiguous', _TRAINING, meters=(11,))
    _write_entry(tmp_path, record)
    assert len(ChartWindows(tmp_path, [record], _VOCABULARY)) == 0


def test_the_split_is_stable_and_disjoint(tmp_path: Path) -> None:
    records = [_record('Modern', f'/corpus/{n}.sm', meters=(16,)) for n in range(20)]
    for record in records:
        _write_entry(tmp_path, record)
    train = ChartWindows(tmp_path, records, _VOCABULARY)
    held = ChartWindows(tmp_path, records, _VOCABULARY, validation=True)
    assert len(train) + len(held) == len(records)
    assert len(held) > 0
    assert {e.path for e in train.examples}.isdisjoint(e.path for e in held.examples)


def test_a_limit_stops_collecting_early(tmp_path: Path) -> None:
    records = [_record('Modern', f'/corpus/{n}.sm', meters=(16, 16)) for n in range(20)]
    for record in records:
        _write_entry(tmp_path, record)
    assert len(ChartWindows(tmp_path, records, _VOCABULARY, limit=3)) <= 4


def test_a_rating_beyond_the_embedding_is_clamped(tmp_path: Path) -> None:
    record = _record('Modern', _TRAINING, meters=(99,))
    _write_entry(tmp_path, record)
    windows = ChartWindows(tmp_path, [record], _VOCABULARY)
    assert all(example.meter < MAX_METER for example in windows.examples)


def test_a_window_carries_both_heads_targets(tmp_path: Path) -> None:
    record = _record('Modern', _TRAINING, meters=(16,))
    _write_entry(tmp_path, record)
    windows = ChartWindows(tmp_path, [record], _VOCABULARY)
    window = windows[0]
    assert window['features'].shape[1] == TOTAL_CHANNELS
    assert window['pattern_target'].shape == (WINDOW_STEPS,)
    assert window['delta'].shape == (WINDOW_STEPS,)
    assert int(window['rate']) < MAX_RATE
    # Slots the chart does not reach are ignored by the loss.
    assert int((window['pattern_target'] == -100).sum()) > 0


def test_a_chart_the_entry_does_not_hold_yields_empty_targets(tmp_path: Path) -> None:
    # The manifest and the cache can disagree; a window still has to be shaped
    # correctly rather than raising.
    record = _record('Modern', _TRAINING, meters=(16,))
    record['charts'][0]['difficulty'] = 'Beginner'
    _write_entry(tmp_path, record)
    windows = ChartWindows(tmp_path, [record], _VOCABULARY)
    assert int((windows[0]['pattern_target'] != -100).sum()) == 0
    # The prior counts only the charts it was asked about, so a mismatched
    # entry contributes slot totals but no steps.
    assert np.all(measure_prior(windows.examples) < 0.0)


def test_a_held_out_window_is_never_mirrored(tmp_path: Path) -> None:
    # Validation figures have to stay comparable across runs, so the identity
    # reflection is the only one a held-out window ever sees.
    records = [_record('Modern', f'/corpus/{n}.sm', meters=(16,)) for n in range(20)]
    for record in records:
        _write_entry(tmp_path, record)
    held = ChartWindows(tmp_path, records, _VOCABULARY, validation=True)
    assert MIRRORS[0] == (0, 1, 2, 3)
    # Every row in the entry steps on the left panel; a reflection would move
    # some of them to the right one.
    left = _VOCABULARY.token_for((1, 0, 0, 0))
    tokens = held[0]['pattern_target'].numpy()
    assert set(tokens[tokens != -100].tolist()) == {left}


def test_a_chart_whose_notes_share_one_slot_has_no_rate(tmp_path: Path) -> None:
    # Duration of zero would divide by zero when turning a count into a rate.
    record = _record('Modern', _TRAINING, meters=(16,))
    path = _write_entry(tmp_path, record)
    with np.load(path, allow_pickle=False) as data:
        arrays = dict(data)
    arrays['slots_0'] = np.zeros(8, dtype=np.int32)
    arrays['panels_0'] = np.tile(np.array([[1, 0, 0, 0]], dtype=np.uint8), (8, 1))
    np.savez(path, **arrays)
    windows = ChartWindows(tmp_path, [record], _VOCABULARY)
    assert int(windows[0]['rate']) == 0


def test_a_chart_with_one_note_has_no_gap_to_measure(tmp_path: Path) -> None:
    record = _record('Modern', _TRAINING, meters=(16,))
    _write_entry(tmp_path, record, rows=1)
    windows = ChartWindows(tmp_path, [record], _VOCABULARY)
    assert int(windows[0]['delta'].sum()) == 0


def test_the_prior_reflects_where_steps_actually_fall(tmp_path: Path) -> None:
    record = _record('Modern', _TRAINING, meters=(16,))
    _write_entry(tmp_path, record)
    windows = ChartWindows(tmp_path, [record], _VOCABULARY)
    prior = measure_prior(windows.examples)
    assert prior.shape == (MEASURE_SLOTS,)
    assert np.all(np.isfinite(prior))
    # Every note in the entry sits on a quarter, so quarters must outscore the
    # positions between them.
    assert prior[0] > prior[1]


def test_a_prior_over_nothing_is_still_finite() -> None:
    prior = measure_prior([])
    assert prior.shape == (MEASURE_SLOTS,)
    assert np.all(np.isfinite(prior))


def test_every_worker_draws_its_own_windows(tmp_path: Path) -> None:
    # A generator built in __init__ is copied into every worker along with its
    # state, so all of them would draw the same window in lockstep.
    record = _record('Modern', _TRAINING, meters=(16,))
    _write_entry(tmp_path, record)
    windows = ChartWindows(tmp_path, [record], _VOCABULARY)
    assert windows.rng is windows.rng


@pytest.mark.parametrize('mirror', MIRRORS)
def test_every_mirror_is_a_permutation_of_the_pad(mirror: tuple[int, ...]) -> None:
    assert sorted(mirror) == [0, 1, 2, 3]
