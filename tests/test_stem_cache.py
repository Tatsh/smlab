"""Tests for building the stem-based feature cache."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

from smlab.features import TOTAL_CHANNELS
from smlab.simfile import load_simfile
from smlab.stem_cache import build_stem_cache, cache_channels, stem_cache_entry
from smlab.stems import STEM_NAMES, SeparationError
import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]  # No stubs are published.
import torch

if TYPE_CHECKING:
    from pytest_mock import MockerFixture
    from smlab.typing import SongRecord

_RATE = 22050
_CPU = torch.device('cpu')
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


class _Model:
    """Stands in for a demucs model, which is far too slow to run in a test."""

    def __init__(self) -> None:
        self.samplerate = _RATE
        self.sources = list(STEM_NAMES)


def _record(tmp_path: Path, *, text: str = SM_TEXT, seconds: float = 16.0) -> SongRecord:
    simfile = tmp_path / 'song.sm'
    simfile.write_text(text)
    audio = tmp_path / 'song.wav'
    times = np.arange(int(_RATE * seconds), dtype=np.float32) / _RATE
    sf.write(audio, np.sin(2.0 * np.pi * 440.0 * times).astype(np.float32), _RATE)
    return cast('SongRecord', {'simfile': str(simfile), 'audio': str(audio)})


@pytest.fixture
def separated(mocker: MockerFixture) -> None:
    mocker.patch(
        'smlab.stem_cache.separate',
        side_effect=lambda _model, path, _device: dict.fromkeys(
            STEM_NAMES, np.asarray(sf.read(path)[0], dtype=np.float32)
        ),
    )


def test_the_feature_width_matches_the_extractor() -> None:
    assert cache_channels() == TOTAL_CHANNELS


@pytest.mark.usefixtures('separated')
def test_an_entry_holds_features_and_one_chart(tmp_path: Path) -> None:
    arrays = stem_cache_entry(_record(tmp_path), _Model(), _CPU)
    assert arrays is not None
    assert arrays['features'].shape[1] == TOTAL_CHANNELS
    assert 'slots_0' in arrays
    assert 'panels_0' in arrays
    assert 'meta' in arrays


@pytest.mark.usefixtures('separated')
def test_note_targets_sit_on_half_the_feature_grid(tmp_path: Path) -> None:
    # Audio is sampled twice as finely as notes are placed.
    arrays = stem_cache_entry(_record(tmp_path), _Model(), _CPU)
    assert arrays is not None
    slots = np.asarray(arrays['slots_0'], dtype=np.int64)
    assert int(slots.max()) < arrays['features'].shape[0] // 2


def test_a_malformed_simfile_yields_no_entry(tmp_path: Path) -> None:
    record = _record(tmp_path, text='not a simfile at all')
    assert stem_cache_entry(record, _Model(), _CPU) is None


def test_a_song_that_cannot_be_separated_yields_no_entry(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    mocker.patch('smlab.stem_cache.separate', side_effect=SeparationError('no'))
    assert stem_cache_entry(_record(tmp_path), _Model(), _CPU) is None


@pytest.mark.usefixtures('separated')
def test_audio_too_short_to_grid_yields_no_entry(tmp_path: Path) -> None:
    assert stem_cache_entry(_record(tmp_path, seconds=0.2), _Model(), _CPU) is None


@pytest.mark.usefixtures('separated')
def test_a_song_whose_charts_are_all_unusable_yields_no_entry(tmp_path: Path) -> None:
    text = SM_TEXT.replace('dance-single', 'dance-double')
    assert stem_cache_entry(_record(tmp_path, text=text), _Model(), _CPU) is None


@pytest.mark.usefixtures('separated')
def test_a_chart_with_too_few_rows_is_left_out(tmp_path: Path) -> None:
    stub = """
#NOTES:
     dance-single:
     :
     Easy:
     2:
     0,0,0,0,0:
1000
0100
;
"""
    arrays = stem_cache_entry(_record(tmp_path, text=SM_TEXT + stub), _Model(), _CPU)
    assert arrays is not None
    assert 'slots_1' not in arrays


@pytest.mark.usefixtures('separated')
def test_the_corpus_is_walked_and_written(tmp_path: Path, mocker: MockerFixture) -> None:
    mocker.patch('smlab.stem_cache.load_separator', return_value=_Model())
    root = tmp_path / 'cache'
    results = list(build_stem_cache([_record(tmp_path)], root, device=_CPU))
    assert results == [(str(tmp_path / 'song.sm'), True)]
    assert list(root.glob('*/*.npz'))


@pytest.mark.usefixtures('separated')
def test_an_entry_already_written_is_not_rebuilt(tmp_path: Path, mocker: MockerFixture) -> None:
    mocker.patch('smlab.stem_cache.load_separator', return_value=_Model())
    root = tmp_path / 'cache'
    record = _record(tmp_path)
    list(build_stem_cache([record], root, device=_CPU))
    written = next(iter(root.glob('*/*.npz')))
    stamp = written.stat().st_mtime_ns
    assert list(build_stem_cache([record], root, device=_CPU)) == [(record['simfile'], True)]
    assert written.stat().st_mtime_ns == stamp


def test_a_song_that_yields_nothing_is_reported_as_unwritten(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    mocker.patch('smlab.stem_cache.load_separator', return_value=_Model())
    mocker.patch('smlab.stem_cache.separate', side_effect=SeparationError('no'))
    record = _record(tmp_path)
    assert list(build_stem_cache([record], tmp_path / 'cache', device=_CPU)) == [
        (record['simfile'], False)
    ]


@pytest.mark.usefixtures('separated')
def test_a_device_is_chosen_when_none_is_given(tmp_path: Path, mocker: MockerFixture) -> None:
    mocker.patch('smlab.stem_cache.load_separator', return_value=_Model())
    mocker.patch('torch.cuda.is_available', return_value=False)
    assert list(build_stem_cache([_record(tmp_path)], tmp_path / 'cache'))


@pytest.mark.usefixtures('separated')
def test_a_simfile_without_timing_yields_no_entry(tmp_path: Path, mocker: MockerFixture) -> None:
    # The loaders raise rather than return a Simfile carrying no tempo, so the
    # guard against one has to be reached directly.
    record = _record(tmp_path)
    parsed = load_simfile(Path(record['simfile']))
    mocker.patch('smlab.stem_cache.load_simfile', return_value=replace(parsed, timing=None))
    assert stem_cache_entry(record, _Model(), _CPU) is None
