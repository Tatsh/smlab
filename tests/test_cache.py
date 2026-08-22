"""Tests for the beat-grid feature cache."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]  # No stubs are published.

from smlab.cache import cache_path_for, iter_cached, load_cached, write_song_cache
from smlab.simfile import load_simfile

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from smlab.typing import SongRecord

_RATE = 22050
_MEASURE = '1000\n0100\n0010\n0001'
SM_TEXT = """#TITLE:Test Song;
#MUSIC:song.wav;
#OFFSET:0.000;
#BPMS:0.000=150.000;

#NOTES:
     dance-single:
     :
     Challenge:
     9:
     0,0,0,0,0:
{};
""".format('\n,\n'.join([_MEASURE] * 8))
"""Eight measures of quarter notes, which clears the cache's minimum row count."""


def _record(tmp_path: Path, *, text: str = SM_TEXT, seconds: float = 16.0) -> SongRecord:
    simfile = tmp_path / 'song.sm'
    simfile.write_text(text)
    audio = tmp_path / 'song.wav'
    times = np.arange(int(_RATE * seconds), dtype=np.float32) / _RATE
    sf.write(audio, np.sin(2.0 * np.pi * 440.0 * times).astype(np.float32), _RATE)
    return cast('SongRecord', {'simfile': str(simfile), 'audio': str(audio)})


def test_a_cache_path_is_derived_from_the_simfile(tmp_path: Path) -> None:
    first = cache_path_for(tmp_path, '/songs/a/a.sm')
    second = cache_path_for(tmp_path, '/songs/b/b.sm')
    assert first != second
    assert first == cache_path_for(tmp_path, '/songs/a/a.sm')
    # Sharded into subdirectories so no one directory holds the whole corpus.
    assert first.parent.parent == tmp_path
    assert first.suffix == '.npz'


def test_a_song_is_cached_and_read_back(tmp_path: Path) -> None:
    record = _record(tmp_path)
    root = tmp_path / 'cache'
    assert write_song_cache((record, str(root))) == record['simfile']
    entries = list(iter_cached(root))
    assert len(entries) == 1
    song = load_cached(entries[0])
    assert song is not None
    assert len(song) > 0
    assert song.features.shape[1] > 0
    assert song.charts[0]['difficulty'] == 'Challenge'
    assert song.charts[0]['meter'] == 9


def test_an_existing_entry_is_not_rebuilt(tmp_path: Path) -> None:
    record = _record(tmp_path)
    root = tmp_path / 'cache'
    write_song_cache((record, str(root)))
    written = next(iter_cached(root))
    stamp = written.stat().st_mtime_ns
    assert write_song_cache((record, str(root))) == record['simfile']
    assert written.stat().st_mtime_ns == stamp


def test_a_malformed_simfile_is_not_cached(tmp_path: Path) -> None:
    record = _record(tmp_path, text='not a simfile at all')
    assert write_song_cache((record, str(tmp_path / 'cache'))) is None


def test_a_simfile_without_timing_is_not_cached(tmp_path: Path, mocker: MockerFixture) -> None:
    # The loaders raise rather than return a Simfile carrying no tempo, so the guard against one
    # has to be reached directly.
    record = _record(tmp_path)
    parsed = load_simfile(Path(record['simfile']))
    mocker.patch('smlab.cache.load_simfile', return_value=replace(parsed, timing=None))
    assert write_song_cache((record, str(tmp_path / 'cache'))) is None


def test_a_chart_with_too_few_rows_is_left_out(tmp_path: Path) -> None:
    # A handful of notes teaches the model nothing, and the song is still worth caching for its
    # other charts.
    stub = """
#NOTES:
     dance-single:
     :
     Easy:
     2:
     0,0,0,0,0:
1000
0100
0010
0001
;
"""
    root = tmp_path / 'cache'
    write_song_cache((_record(tmp_path, text=SM_TEXT + stub), str(root)))
    song = load_cached(next(iter_cached(root)))
    assert song is not None
    assert [chart['difficulty'] for chart in song.charts] == ['Challenge']


def test_audio_too_short_to_grid_is_not_cached(tmp_path: Path) -> None:
    # Fewer than the minimum slots means there is nothing to learn from.
    record = _record(tmp_path, seconds=0.2)
    assert write_song_cache((record, str(tmp_path / 'cache'))) is None


def test_a_song_whose_charts_are_all_unusable_is_not_cached(tmp_path: Path) -> None:
    # A dance-double chart is not a dance-single one, so nothing is left.
    text = SM_TEXT.replace('dance-single', 'dance-double')
    assert write_song_cache((_record(tmp_path, text=text), str(tmp_path / 'cache'))) is None


def test_missing_audio_is_not_cached(tmp_path: Path) -> None:
    record = _record(tmp_path)
    (tmp_path / 'song.wav').unlink()
    assert write_song_cache((record, str(tmp_path / 'cache'))) is None


def test_an_unreadable_entry_reads_back_as_nothing(tmp_path: Path) -> None:
    broken = tmp_path / 'broken.npz'
    broken.write_bytes(b'not an npz archive')
    assert load_cached(broken) is None


def test_an_entry_missing_its_arrays_reads_back_as_nothing(tmp_path: Path) -> None:
    incomplete = tmp_path / 'incomplete.npz'
    np.savez(incomplete, features=np.zeros((4, 4), dtype=np.float16))
    assert load_cached(incomplete) is None


def test_an_empty_cache_yields_nothing(tmp_path: Path) -> None:
    assert list(iter_cached(tmp_path)) == []


def test_entries_come_back_in_a_stable_order(tmp_path: Path) -> None:
    for name in ('cc', 'aa', 'bb'):
        shard = tmp_path / name
        shard.mkdir()
        np.savez(shard / f'{name}.npz', features=np.zeros(1))
    assert [path.stem for path in iter_cached(tmp_path)] == ['aa', 'bb', 'cc']


@pytest.mark.parametrize('seconds', [16.0, 20.0])
def test_the_cache_holds_one_feature_row_per_slot(tmp_path: Path, seconds: float) -> None:
    record = _record(tmp_path, seconds=seconds)
    root = tmp_path / 'cache'
    write_song_cache((record, str(root)))
    song = load_cached(next(iter_cached(root)))
    assert song is not None
    assert len(song) == song.features.shape[0]
