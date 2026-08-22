"""Tests for how the phase model's training data is selected and scored."""

from __future__ import annotations

from typing import TYPE_CHECKING

from smlab.offset import BEATS_PER_MEASURE, PHASE_BINS, phase_of_offset
from smlab.train_offset import (
    FoldedProfiles,
    OffsetTrainingConfig,
    build_envelope_cache,
    cyclic_error,
    usable,
)
import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

_RATE = 172.265625
_BPM = 120.0
_BANDS = 4


def _record(**changes: object) -> dict[str, object]:
    base: dict[str, object] = {
        'constant_bpm': True,
        'offset_declared': True,
        'stops': False,
        'primary_bpm': 128.0,
    }
    return base | changes


@pytest.mark.parametrize(
    ('changes', 'wanted'),
    [
        ({}, True),
        # A tempo that moves means the beat grid moves, so a single phase label
        # would be wrong for most of the song.
        ({'constant_bpm': False}, False),
        # An offset the scanner guessed is not evidence of anything.
        ({'offset_declared': False}, False),
        # Stops shift everything after them off the grid the label assumes.
        ({'stops': True}, False),
        ({'primary_bpm': 0}, False),
    ],
)
def test_only_songs_with_a_trustworthy_grid_are_used(
    changes: dict[str, object], *, wanted: bool
) -> None:
    assert usable(_record(**changes)) is wanted


@pytest.mark.parametrize(
    ('predicted', 'actual', 'wanted'),
    [
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 1),
        # The bar wraps, so bin 95 and bin 0 are neighbours rather than 95
        # apart. Measuring that the long way round would make a near-perfect
        # answer look like the worst possible one.
        (95, 0, 1),
        (0, 95, 1),
        (0, 48, 48),
    ],
)
def test_phase_error_is_measured_the_short_way_round(
    predicted: int, actual: int, wanted: int
) -> None:
    got = cyclic_error(np.array([predicted]), np.array([actual]))
    assert int(got[0]) == wanted


def test_phase_error_never_exceeds_half_a_bar() -> None:
    every = np.arange(PHASE_BINS)
    gaps = cyclic_error(every, np.zeros(PHASE_BINS, dtype=np.int64))
    assert gaps.max() <= PHASE_BINS // 2


def _write_song(path: Path, offset: float, seconds: float = 60.0) -> None:
    period = BEATS_PER_MEASURE * 60.0 / _BPM
    frames = int(_RATE * seconds)
    envelopes = np.zeros((_BANDS, frames), dtype=np.float32)
    time = (-offset) % period
    while time < seconds:
        envelopes[:, round(time * _RATE)] = 1.0
        time += period
    np.savez(
        path,
        envelopes=envelopes,
        bpm=np.float64(_BPM),
        offset=np.float64(offset),
        rate=np.float64(_RATE),
    )


def test_the_label_matches_the_authored_offset(tmp_path: Path) -> None:
    offset = -0.35
    _write_song(tmp_path / '00000000cafe.npz', offset)
    data = FoldedProfiles(tmp_path, OffsetTrainingConfig(), validation=True)
    profile, phase = data[0]
    assert profile.shape == (_BANDS, PHASE_BINS)
    assert int(phase) == phase_of_offset(offset, _BPM)


def test_the_label_does_not_move_with_the_excerpt(tmp_path: Path) -> None:
    # Training draws a fresh excerpt from each song every epoch. If the label
    # depended on where the excerpt began, most of them would be wrong.
    _write_song(tmp_path / '00000003cafe.npz', -0.35)
    data = FoldedProfiles(tmp_path, OffsetTrainingConfig())
    labels = {int(data[index][1]) for index in range(6)}
    assert len(labels) == 1


def test_the_split_is_disjoint(tmp_path: Path) -> None:
    for index in range(16):
        _write_song(tmp_path / f'{index:016x}.npz', -0.1 * index)
    train = FoldedProfiles(tmp_path, OffsetTrainingConfig())
    valid = FoldedProfiles(tmp_path, OffsetTrainingConfig(), validation=True)
    assert not set(train.paths) & set(valid.paths)
    assert len(train.paths) + len(valid.paths) == 16


def test_each_worker_draws_its_own_excerpts(tmp_path: Path, mocker: MockerFixture) -> None:
    # A generator built in __init__ is copied into every worker along with its
    # state, so all of them draw identically and the model sees a fraction of
    # the variety the epoch count implies. Deferring to the per-worker seed
    # PyTorch supplies is what separates them.
    _write_song(tmp_path / '00000003cafe.npz', -0.35)
    drawn = []
    for seed in (11, 22, 33):
        mocker.patch('torch.utils.data.get_worker_info', return_value=mocker.Mock(seed=seed))
        data = FoldedProfiles(tmp_path, OffsetTrainingConfig())
        drawn.append(tuple(int(x) for x in data.rng.integers(1_000_000, size=8)))
    assert len(set(drawn)) == len(drawn)


def test_the_main_process_draws_reproducibly(tmp_path: Path, mocker: MockerFixture) -> None:
    _write_song(tmp_path / '00000003cafe.npz', -0.35)
    mocker.patch('torch.utils.data.get_worker_info', return_value=None)
    first = FoldedProfiles(tmp_path, OffsetTrainingConfig()).rng.integers(1_000_000, size=8)
    second = FoldedProfiles(tmp_path, OffsetTrainingConfig()).rng.integers(1_000_000, size=8)
    assert list(first) == list(second)


def test_unusable_songs_are_not_cached(tmp_path: Path) -> None:
    written = build_envelope_cache([_record(constant_bpm=False, audio='missing.mp3')], tmp_path)
    assert written == 0
    assert not list(tmp_path.glob('*.npz'))
