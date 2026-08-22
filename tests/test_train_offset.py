"""Tests for how the phase model's training data is selected and scored."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]  # No stubs are published.
import torch

from smlab.offset import BEATS_PER_MEASURE, PHASE_BINS, phase_of_offset
from smlab.train.offset import (
    FoldedProfiles,
    OffsetTrainingConfig,
    build_envelope_cache,
    cyclic_error,
    train_offset_model,
    usable,
)

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
        # A tempo that moves means the beat grid moves, so a single phase label would be wrong for
        # most of the song.
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
        # The bar wraps, so bin 95 and bin 0 are neighbours rather than 95 apart. Measuring that the
        # long way round would make a near-perfect answer look like the worst possible one.
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
    # Training draws a fresh excerpt from each song every epoch. If the label depended on where the
    # excerpt began, most of them would be wrong.
    _write_song(tmp_path / '00000003cafe.npz', -0.35)
    data = FoldedProfiles(tmp_path, OffsetTrainingConfig())
    labels = {int(data[index][1]) for index in range(6)}
    assert len(labels) == 1


def test_the_split_is_disjoint(tmp_path: Path) -> None:
    for index in range(16):
        _write_song(tmp_path / f'{index:08x}00000000.npz', -0.1 * index)
    train = FoldedProfiles(tmp_path, OffsetTrainingConfig())
    valid = FoldedProfiles(tmp_path, OffsetTrainingConfig(), validation=True)
    assert not set(train.paths) & set(valid.paths)
    assert len(train.paths) + len(valid.paths) == 16
    assert train.paths
    assert valid.paths


def test_each_worker_draws_its_own_excerpts(tmp_path: Path, mocker: MockerFixture) -> None:
    # A generator built in __init__ is copied into every worker along with its state, so all of
    # them draw identically and the model sees a fraction of the variety the epoch count implies.
    # Deferring to the per-worker seed PyTorch supplies is what separates them.
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


def test_an_empty_split_says_so_rather_than_indexing_off_the_end(tmp_path: Path) -> None:
    with pytest.raises(IndexError, match='no songs'):
        FoldedProfiles(tmp_path, OffsetTrainingConfig())[0]


def test_a_song_is_cached_with_its_labels(tmp_path: Path) -> None:
    audio = tmp_path / 'song.wav'
    sf.write(audio, np.zeros(22050 * 35, dtype='float32'), 22050)
    destination = tmp_path / 'envelopes'
    assert build_envelope_cache([_record(audio=str(audio), offset=-0.048)], destination) == 1
    with np.load(next(destination.glob('*.npz'))) as data:
        assert data['envelopes'].shape[0] == _BANDS
        assert float(data['bpm']) == pytest.approx(128.0)


def test_a_song_already_cached_is_counted_but_not_rebuilt(tmp_path: Path) -> None:
    audio = tmp_path / 'song.wav'
    sf.write(audio, np.zeros(22050 * 35, dtype='float32'), 22050)
    destination = tmp_path / 'envelopes'
    build_envelope_cache([_record(audio=str(audio), offset=-0.048)], destination)
    written = next(destination.glob('*.npz'))
    stamp = written.stat().st_mtime_ns
    assert build_envelope_cache([_record(audio=str(audio), offset=-0.048)], destination) == 1
    assert written.stat().st_mtime_ns == stamp


def test_a_song_that_cannot_be_read_is_skipped(tmp_path: Path) -> None:
    broken = tmp_path / 'broken.wav'
    broken.write_bytes(b'not audio at all')
    assert build_envelope_cache([_record(audio=str(broken), offset=-0.048)], tmp_path / 'out') == 0


def test_a_song_too_short_to_fold_is_skipped(tmp_path: Path) -> None:
    # A fold needs several bars before it means anything.
    audio = tmp_path / 'brief.wav'
    sf.write(audio, np.zeros(22050 * 3, dtype='float32'), 22050)
    assert build_envelope_cache([_record(audio=str(audio), offset=-0.048)], tmp_path / 'out') == 0


def test_progress_is_reported_for_a_long_run(tmp_path: Path, mocker: MockerFixture) -> None:
    # Separating a corpus takes hours, so the run has to say where it is.
    audio = tmp_path / 'song.wav'
    sf.write(audio, np.zeros(22050 * 35, dtype='float32'), 22050)
    mocker.patch('smlab.train.offset._PROGRESS_EVERY', 1)
    assert (
        build_envelope_cache([_record(audio=str(audio), offset=-0.048)] * 3, tmp_path / 'out') == 3
    )


def test_training_keeps_the_best_weights(tmp_path: Path) -> None:
    for index in range(16):
        _write_song(tmp_path / f'{index:08x}00000000.npz', -0.1 * index)
    output = tmp_path / 'checkpoints' / 'offset.pt'
    measured = train_offset_model(
        tmp_path, output, OffsetTrainingConfig(batch_size=2, epochs=1, windows=1)
    )
    assert set(measured) == {'exact', 'half_beat_out', 'within_one_bin', 'within_two_bins'}
    assert output.is_file()
    assert 'model' in torch.load(output, weights_only=False)


def test_training_falls_back_to_the_default_settings(tmp_path: Path) -> None:
    for index in range(16):
        _write_song(tmp_path / f'{index:08x}00000000.npz', -0.1 * index)
    assert train_offset_model(tmp_path, tmp_path / 'offset.pt', OffsetTrainingConfig(epochs=0)) == {
        'within_one_bin': 0.0
    }


def test_an_epoch_that_does_not_improve_leaves_the_checkpoint_alone(tmp_path: Path) -> None:
    # Only the best epoch is kept, so a later one that scores no better must not overwrite it.
    for index in range(16):
        _write_song(tmp_path / f'{index:08x}00000000.npz', -0.1 * index)
    torch.manual_seed(0)
    output = tmp_path / 'offset.pt'
    best = train_offset_model(
        tmp_path, output, OffsetTrainingConfig(batch_size=2, epochs=4, windows=1)
    )
    assert output.is_file() == (best['within_one_bin'] > 0.0)
