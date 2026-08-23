"""Tests for the command line interface."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING
import json

import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]  # No stubs are published.
import torch

from smlab.encoder import MEASURE_SLOTS, EncoderConfig
from smlab.features import TOTAL_CHANNELS
from smlab.generate import DEFAULT_SCALE, SCALES, target_nps
from smlab.heads import ChartModel
from smlab.main import (
    DEFAULT_METERS,
    IMAGE_DIRECTORY,
    LATENCY_VARIABLE,
    current_user,
    default_meter,
    main,
)
from smlab.offset import OffsetModel
from smlab.simfile import load_simfile
from smlab.stems import STEM_NAMES, SeparationError
from smlab.timing import TimingData
from smlab.vocab import Vocabulary, encode_row
from smlab.warp import TempoReading, Warp, WarpFit
from smlab.weights import CHART_WEIGHTS, OFFSET_WEIGHTS, REPOSITORY_VARIABLE

if TYPE_CHECKING:
    from pathlib import Path

    from click.testing import CliRunner
    from pytest_mock import MockerFixture

from tests.conftest import SM_TEXT

_RATE = 22050
_BPM = 150.0
_SMALL = EncoderConfig(
    channels=16, model_dimension=24, local_blocks=1, slot_layers=1, measure_layers=1, heads=2
)
_VOCABULARY = Vocabulary([
    encode_row(row) for row in ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
])


class _Separator:
    """Stands in for a demucs model, which is far too slow to run in a test."""

    def __init__(self) -> None:
        self.samplerate = _RATE
        self.sources = list(STEM_NAMES)


def _chart_model() -> ChartModel:
    torch.manual_seed(0)
    return ChartModel(len(_VOCABULARY), _SMALL).eval()


def _offset_model() -> OffsetModel:
    torch.manual_seed(0)
    return OffsetModel().eval()


def _audio(path: Path, seconds: float = 6.0) -> Path:
    times = np.arange(int(_RATE * seconds), dtype=np.float32) / _RATE
    samples = np.sin(2.0 * np.pi * 440.0 * times).astype(np.float32)
    samples[:: int(_RATE * 60.0 / _BPM)] = 1.0
    sf.write(path, samples, _RATE)
    return path


def _checkpoints(tmp_path: Path) -> Path:
    """Write a chart checkpoint small enough to load in a test."""
    directory = tmp_path / 'checkpoints'
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = ChartModel(len(_VOCABULARY), _SMALL)
    torch.save(
        {
            'model': model.state_dict(),
            'prior': np.zeros(MEASURE_SLOTS, dtype=np.float32),
            'vocabulary': len(_VOCABULARY),
        },
        directory / CHART_WEIGHTS,
    )
    return directory


@pytest.fixture
def generation(mocker: MockerFixture) -> None:
    """Stand in for the parts of generation that need a GPU or a download."""
    mocker.patch('smlab.main.EncoderConfig', return_value=_SMALL)
    mocker.patch('smlab.main.load_vocabulary', return_value=_VOCABULARY)
    mocker.patch('smlab.main._load_offset_model', return_value=None)
    mocker.patch(
        'smlab.chart.gen.separate',
        side_effect=lambda _m, path, _d: dict.fromkeys(
            STEM_NAMES, np.asarray(sf.read(path)[0], dtype=np.float32)
        ),
    )

    mocker.patch('smlab.main.load_separator', return_value=_Separator())


def test_a_username_is_reported(mocker: MockerFixture) -> None:
    mocker.patch('getpass.getuser', return_value='charter')
    assert current_user() == 'charter'


def test_a_missing_username_falls_back_to_the_tool_name(mocker: MockerFixture) -> None:
    # A container with no passwd entry still has to credit the chart to something.
    mocker.patch('getpass.getuser', side_effect=KeyError)
    assert current_user() == 'smlab'


@pytest.mark.parametrize('name', list(DEFAULT_METERS))
def test_the_classic_scale_uses_the_table_directly(name: str) -> None:
    assert default_meter(name, 10) == DEFAULT_METERS[name]


def test_another_scale_translates_through_the_note_rate() -> None:
    # The table is written on the classic scale, so asking for another one has to land on whichever
    # rating implies the same speed.
    wanted = target_nps(DEFAULT_METERS['Hard'], 10)
    chosen = default_meter('Hard', 20)
    assert abs(target_nps(chosen, 20) - wanted) <= abs(target_nps(chosen + 2, 20) - wanted)


def test_an_unknown_difficulty_gets_a_middling_rating() -> None:
    assert default_meter('Ultra', 10) == 6


def test_the_default_scale_is_used_when_none_is_given() -> None:
    assert default_meter('Hard') == default_meter('Hard', DEFAULT_SCALE)


def test_help_lists_every_command(runner: CliRunner) -> None:
    result = runner.invoke(main, ['-h'])
    assert result.exit_code == 0
    for command in ('analyze', 'envelopes', 'generate', 'image', 'publish', 'scan', 'stems'):
        assert command in result.output


def test_debug_logging_can_be_turned_on(runner: CliRunner, tmp_path: Path) -> None:
    assert runner.invoke(main, ['-d', 'analyze', str(_simfile(tmp_path))]).exit_code == 0


def _simfile(tmp_path: Path, text: str = SM_TEXT) -> Path:
    path = tmp_path / 'song.sm'
    path.write_text(text)
    (tmp_path / 'test.ogg').write_bytes(b'')
    return path


def test_a_songs_tree_is_scanned_into_a_manifest(runner: CliRunner, tmp_path: Path) -> None:
    song = tmp_path / 'Songs' / 'Pack' / 'Song'
    song.mkdir(parents=True)
    (song / 'Song.sm').write_text(SM_TEXT)
    (song / 'test.ogg').write_bytes(b'')
    manifest = tmp_path / 'manifest.json'
    result = runner.invoke(
        main, ['scan', str(tmp_path / 'Songs'), '-o', str(manifest), '-w', '1', '-x', 'Nothing']
    )
    assert result.exit_code == 0
    assert 'Wrote 1 records' in result.output
    assert json.loads(manifest.read_text())[0]['title'] == 'Test Song'


def test_a_vocabulary_is_collected_from_the_cache(runner: CliRunner, tmp_path: Path) -> None:
    shard = tmp_path / 'cache' / 'aa'
    shard.mkdir(parents=True)
    np.savez(
        shard / 'aa.npz',
        features=np.zeros((8, 4), dtype=np.float16),
        slots_0=np.arange(3, dtype=np.int32),
        panels_0=np.array([[1, 0, 0, 0]] * 3, dtype=np.uint8),
        meta=np.asarray(json.dumps([{'difficulty': 'Challenge', 'index': 0, 'meter': 9}])),
    )
    output = tmp_path / 'vocabulary.json'
    result = runner.invoke(
        main, ['vocab', '-c', str(tmp_path / 'cache'), '-o', str(output), '-l', '8']
    )
    assert result.exit_code == 0
    assert 'Wrote 1 patterns' in result.output


def test_envelopes_are_cached_from_a_manifest(
    runner: CliRunner, tmp_path: Path, mocker: MockerFixture
) -> None:
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps([{'audio': 'song.wav'}]))
    cache = mocker.patch('smlab.main.build_envelope_cache', return_value=3)
    result = runner.invoke(main, ['envelopes', '-m', str(manifest), '-o', str(tmp_path / 'env')])
    assert result.exit_code == 0
    assert 'Cached 3 songs' in result.output
    cache.assert_called_once()


def test_the_offset_model_is_trained(
    runner: CliRunner, tmp_path: Path, mocker: MockerFixture
) -> None:
    cache = tmp_path / 'env'
    cache.mkdir()
    mocker.patch('smlab.main.train_offset_model', return_value={'exact': 0.5})
    result = runner.invoke(
        main,
        ['train-offset', '-c', str(cache), '-o', str(tmp_path / 'out'), '-e', '1', '-b', '2'],
    )
    assert result.exit_code == 0
    assert 'exact: 0.5000' in result.output


def test_stems_are_separated_for_every_song(
    runner: CliRunner, tmp_path: Path, mocker: MockerFixture
) -> None:
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps([{'simfile': f'{n}.sm'} for n in range(101)]))
    mocker.patch(
        'smlab.main.build_stem_cache',
        return_value=[(f'{n}.sm', n % 2 == 0) for n in range(101)],
    )
    result = runner.invoke(main, ['stems', '-m', str(manifest), '-o', str(tmp_path / 'cache')])
    assert result.exit_code == 0
    # Progress is reported every hundred songs, and the totals at the end.
    assert '100 of 101' in result.output
    assert 'Cached 51 of 101 songs' in result.output


def test_the_chart_model_is_trained(
    runner: CliRunner, tmp_path: Path, mocker: MockerFixture
) -> None:
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps([]))
    (tmp_path / 'vocabulary.json').write_text(json.dumps([1, 2, 3]))
    mocker.patch('smlab.main.train_chart_model', return_value={'quarter_auc': 0.75})
    result = runner.invoke(
        main,
        [
            'train',
            '-m',
            str(manifest),
            '-c',
            str(tmp_path),
            '-v',
            str(tmp_path / 'vocabulary.json'),
            '-o',
            str(tmp_path / 'out'),
            '-e',
            '1',
            '-b',
            '1',
            '-w',
            '0',
        ],
    )
    assert result.exit_code == 0
    assert 'quarter_auc: 0.7500' in result.output


@pytest.mark.usefixtures('generation')
def test_a_song_folder_is_generated(runner: CliRunner, tmp_path: Path) -> None:
    audio = _audio(tmp_path / 'song.wav')
    result = runner.invoke(
        main,
        [
            'generate',
            str(audio),
            '-o',
            str(tmp_path / 'out'),
            '-T',
            'Song',
            '-A',
            'Artist',
            '-c',
            str(_checkpoints(tmp_path)),
            '--bpm',
            str(_BPM),
            '--offset',
            '0',
            '-D',
            'Challenge:16',
            '--nps',
            '4',
        ],
    )
    assert result.exit_code == 0, result.output
    assert 'Using supplied timing' in result.output
    written = tmp_path / 'out' / 'Song'
    assert (written / 'Song.ssc').is_file()
    assert (written / 'Song.wav').is_file()


@pytest.mark.usefixtures('generation')
def test_generation_detects_the_timing_when_none_is_given(
    runner: CliRunner, tmp_path: Path
) -> None:
    result = runner.invoke(
        main,
        [
            'generate',
            str(_audio(tmp_path / 'song.wav')),
            '-o',
            str(tmp_path / 'out'),
            '-T',
            'Song',
            '-c',
            str(_checkpoints(tmp_path)),
            '-D',
            'Easy',
            '--nps',
            '2',
        ],
    )
    assert result.exit_code == 0, result.output
    assert 'Detected' in result.output


@pytest.mark.usefixtures('generation')
def test_a_supplied_tempo_reaches_the_offset_estimator(
    runner: CliRunner, tmp_path: Path, mocker: MockerFixture
) -> None:
    # Overriding the tempo after the estimator has already fitted a phase to its own leaves the
    # two describing different grids, and the offset then sits half the accumulated drift out.
    spy = mocker.patch(
        'smlab.main.estimate_timing',
        return_value={'bpm': 128.199, 'confidence': 0.0, 'offset': 0.0},
    )
    result = runner.invoke(
        main,
        [
            'generate',
            str(_audio(tmp_path / 'song.wav')),
            '-o',
            str(tmp_path / 'out'),
            '-T',
            'Song',
            '-c',
            str(_checkpoints(tmp_path)),
            '-D',
            'Easy',
            '--nps',
            '2',
            '--bpm',
            '128.199',
        ],
    )
    assert result.exit_code == 0, result.output
    assert spy.call_args.kwargs['bpm'] == pytest.approx(128.199)
    assert 'Using 128.199 BPM' in result.output


@pytest.mark.usefixtures('generation')
def test_a_warp_writes_a_second_tempo_segment(runner: CliRunner, tmp_path: Path) -> None:
    # A song whose tempo moves can only be charted correctly by saying where it moves, so the
    # marker has to reach the file rather than being averaged away into one number.
    result = runner.invoke(
        main,
        [
            'generate',
            str(_audio(tmp_path / 'song.wav')),
            '-o',
            str(tmp_path / 'out'),
            '-T',
            'Song',
            '-c',
            str(_checkpoints(tmp_path)),
            '-D',
            'Easy',
            '--nps',
            '2',
            '--bpm',
            str(_BPM),
            '--offset',
            '0',
            '--warp',
            '2:151.5',
        ],
    )
    assert result.exit_code == 0, result.output
    assert 'Warped to 151.500 BPM at 2 s' in result.output
    timing = load_simfile(tmp_path / 'out' / 'Song' / 'Song.ssc').timing
    assert timing is not None
    assert [segment.bpm for segment in timing.bpms] == pytest.approx([_BPM, 151.5])
    # Two seconds at 150 BPM is five beats, and the marker is left exactly there rather than
    # rounded onto a beat, which would move it by up to half a beat.
    assert timing.bpms[1].beat == pytest.approx(5.0, abs=0.01)


@pytest.mark.usefixtures('generation')
def test_a_bare_warp_fits_the_tempo_changes_itself(
    runner: CliRunner, tmp_path: Path, mocker: MockerFixture
) -> None:
    # The option with no value asks for the segments to be found, and a segment at the very start
    # is the tempo the song opens at rather than a marker part way through it.
    fit = mocker.patch(
        'smlab.main.fit_warps',
        return_value=WarpFit(
            warps=[Warp(seconds=0.0, bpm=_BPM), Warp(seconds=2.0, bpm=151.5)], splices=[]
        ),
    )
    result = runner.invoke(
        main,
        [
            'generate',
            str(_audio(tmp_path / 'song.wav')),
            '-o',
            str(tmp_path / 'out'),
            '-T',
            'Song',
            '-c',
            str(_checkpoints(tmp_path)),
            '-D',
            'Easy',
            '--nps',
            '2',
            '--bpm',
            str(_BPM),
            '--offset',
            '0',
            '--warp',
            '--warp-slip',
            '0.05',
        ],
    )
    assert result.exit_code == 0, result.output
    assert fit.call_args.kwargs['tolerance'] == pytest.approx(0.05)
    assert 'Fitted 2 tempo segments' in result.output
    assert f'Opening tempo is {_BPM:.3f} BPM' in result.output
    timing = load_simfile(tmp_path / 'out' / 'Song' / 'Song.ssc').timing
    assert timing is not None
    assert [segment.bpm for segment in timing.bpms] == pytest.approx([_BPM, 151.5])


@pytest.mark.usefixtures('generation')
def test_a_bare_warp_leaves_a_steady_song_alone(
    runner: CliRunner, tmp_path: Path, mocker: MockerFixture
) -> None:
    mocker.patch(
        'smlab.main.fit_warps',
        return_value=WarpFit(warps=[Warp(seconds=0.0, bpm=_BPM)], splices=[]),
    )
    result = runner.invoke(
        main,
        [
            'generate',
            str(_audio(tmp_path / 'song.wav')),
            '-o',
            str(tmp_path / 'out'),
            '-T',
            'Song',
            '-c',
            str(_checkpoints(tmp_path)),
            '-D',
            'Easy',
            '--nps',
            '2',
            '--bpm',
            str(_BPM),
            '--offset',
            '0',
            '--warp',
        ],
    )
    assert result.exit_code == 0, result.output
    assert 'nothing was warped' in result.output
    timing = load_simfile(tmp_path / 'out' / 'Song' / 'Song.ssc').timing
    assert timing is not None
    assert [segment.bpm for segment in timing.bpms] == pytest.approx([_BPM])


@pytest.mark.usefixtures('generation')
def test_a_warp_that_is_not_seconds_and_a_tempo_is_refused(
    runner: CliRunner, tmp_path: Path
) -> None:
    result = runner.invoke(
        main,
        [
            'generate',
            str(_audio(tmp_path / 'song.wav')),
            '-o',
            str(tmp_path / 'out'),
            '-T',
            'Song',
            '--warp',
            'halfway',
        ],
    )
    assert result.exit_code != 0
    assert 'SECONDS:BPM' in result.output


@pytest.mark.usefixtures('generation')
def test_a_warp_to_a_tempo_of_nothing_is_refused(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(
        main,
        [
            'generate',
            str(_audio(tmp_path / 'song.wav')),
            '-o',
            str(tmp_path / 'out'),
            '-T',
            'Song',
            '--warp',
            '10:0',
        ],
    )
    assert result.exit_code != 0
    assert 'tempo above zero' in result.output


def test_drift_reports_every_place_a_song_wanders(
    runner: CliRunner, tmp_path: Path, mocker: MockerFixture
) -> None:
    # One marker is not an answer for a song that moves more than once, so every fitted segment has
    # to reach the command line the user is meant to paste back.
    mocker.patch(
        'smlab.main.measure_tempo',
        return_value=[
            TempoReading(seconds=20.0, bpm=128.0, slip=0.001),
            TempoReading(seconds=60.0, bpm=127.5, slip=0.085),
        ],
    )
    mocker.patch(
        'smlab.main.fit_warps',
        return_value=WarpFit(
            warps=[
                Warp(seconds=0.0, bpm=128.0),
                Warp(seconds=60.0, bpm=127.5),
                Warp(seconds=90.0, bpm=129.25),
            ],
            splices=[],
        ),
    )
    result = runner.invoke(main, ['drift', str(_audio(tmp_path / 'song.wav')), '--bpm', '128'])
    assert result.exit_code == 0, result.output
    assert '127.500 to 128.000 BPM' in result.output
    assert '<- warp' in result.output
    assert 'needs 3 tempo segments' in result.output
    assert 'Generate with: --bpm 128.000 --warp 60:127.500 --warp 90:129.250' in result.output


def test_drift_says_when_the_beat_jumps_rather_than_changing_speed(
    runner: CliRunner, tmp_path: Path, mocker: MockerFixture
) -> None:
    # An edit in the audio is not a tempo change and no warp will repair it, so it has to be called
    # out rather than quietly fitted around.
    mocker.patch(
        'smlab.main.measure_tempo',
        return_value=[TempoReading(seconds=20.0, bpm=128.0, slip=0.0)],
    )
    mocker.patch(
        'smlab.main.fit_warps',
        return_value=WarpFit(warps=[Warp(seconds=0.0, bpm=128.0)], splices=[61.5]),
    )
    result = runner.invoke(main, ['drift', str(_audio(tmp_path / 'song.wav')), '--bpm', '128'])
    assert result.exit_code == 0, result.output
    assert 'The beat jumps at 61.5 s' in result.output
    assert 'a warp will not put it right' in result.output


@pytest.mark.usefixtures('generation')
def test_generating_says_when_the_beat_jumps(
    runner: CliRunner, tmp_path: Path, mocker: MockerFixture
) -> None:
    mocker.patch(
        'smlab.main.fit_warps',
        return_value=WarpFit(warps=[Warp(seconds=0.0, bpm=_BPM)], splices=[12.0, 34.0]),
    )
    result = runner.invoke(
        main,
        [
            'generate',
            str(_audio(tmp_path / 'song.wav')),
            '-o',
            str(tmp_path / 'out'),
            '-T',
            'Song',
            '-c',
            str(_checkpoints(tmp_path)),
            '-D',
            'Easy',
            '--nps',
            '2',
            '--bpm',
            str(_BPM),
            '--offset',
            '0',
        ],
    )
    assert result.exit_code == 0, result.output
    assert 'The beat jumps at 12 s, 34 s' in result.output


def test_drift_calls_a_steady_song_steady(
    runner: CliRunner, tmp_path: Path, mocker: MockerFixture
) -> None:
    mocker.patch(
        'smlab.main.measure_tempo',
        return_value=[TempoReading(seconds=20.0, bpm=128.0, slip=0.001)],
    )
    mocker.patch(
        'smlab.main.fit_warps',
        return_value=WarpFit(warps=[Warp(seconds=0.0, bpm=128.0)], splices=[]),
    )
    result = runner.invoke(main, ['drift', str(_audio(tmp_path / 'song.wav')), '--bpm', '128'])
    assert result.exit_code == 0, result.output
    assert 'Steady enough for one tempo of 128.000 BPM' in result.output


def test_drift_falls_back_to_the_average_when_nothing_can_be_fitted(
    runner: CliRunner, tmp_path: Path, mocker: MockerFixture
) -> None:
    # Stretches long enough to read a tempo from can still be too few to fit segments across.
    mocker.patch(
        'smlab.main.measure_tempo',
        return_value=[
            TempoReading(seconds=20.0, bpm=128.0, slip=0.0),
            TempoReading(seconds=60.0, bpm=127.0, slip=0.0),
        ],
    )
    mocker.patch('smlab.main.fit_warps', return_value=WarpFit(warps=[], splices=[]))
    result = runner.invoke(main, ['drift', str(_audio(tmp_path / 'song.wav')), '--bpm', '128'])
    assert result.exit_code == 0, result.output
    assert 'Steady enough for one tempo of 127.500 BPM' in result.output


def test_drift_detects_a_tempo_when_none_is_given(
    runner: CliRunner, tmp_path: Path, mocker: MockerFixture
) -> None:
    mocker.patch(
        'smlab.main.estimate_timing', return_value={'bpm': 150.0, 'confidence': 1.0, 'offset': 0.0}
    )
    mocker.patch(
        'smlab.main.measure_tempo',
        return_value=[TempoReading(seconds=20.0, bpm=150.0, slip=0.0)],
    )
    mocker.patch(
        'smlab.main.fit_warps',
        return_value=WarpFit(warps=[Warp(seconds=0.0, bpm=150.0)], splices=[]),
    )
    result = runner.invoke(main, ['drift', str(_audio(tmp_path / 'song.wav'))])
    assert result.exit_code == 0, result.output
    assert 'Measuring against the detected 150.000 BPM' in result.output


def test_drift_gives_up_on_a_song_it_cannot_track(
    runner: CliRunner, tmp_path: Path, mocker: MockerFixture
) -> None:
    mocker.patch('smlab.main.measure_tempo', return_value=[])
    result = runner.invoke(main, ['drift', str(_audio(tmp_path / 'song.wav')), '--bpm', '128'])
    assert result.exit_code != 0
    assert 'too short or too quiet' in result.output


@pytest.mark.usefixtures('generation')
def test_the_tempo_can_be_scaled_and_the_grid_shifted(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(LATENCY_VARIABLE, '0.06')
    result = runner.invoke(
        main,
        [
            'generate',
            str(_audio(tmp_path / 'song.wav')),
            '-o',
            str(tmp_path / 'out'),
            '-T',
            'Song',
            '-c',
            str(_checkpoints(tmp_path)),
            '-D',
            'Easy',
            '--nps',
            '2',
            '--bpm',
            '120',
            '--offset',
            '0',
            '--bpm-multiplier',
            '2',
            '--shift-beats',
            '1',
        ],
    )
    assert result.exit_code == 0, result.output
    assert 'Tempo scaled by 2' in result.output
    assert 'Shifted beat 0' in result.output
    assert 'Compensated +60 ms' in result.output


@pytest.mark.usefixtures('generation')
def test_a_rating_above_the_scale_is_reported(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(
        main,
        [
            'generate',
            str(_audio(tmp_path / 'song.wav')),
            '-o',
            str(tmp_path / 'out'),
            '-T',
            'Song',
            '-c',
            str(_checkpoints(tmp_path)),
            '--scale',
            '10',
            '-D',
            'Challenge:15',
            '--bpm',
            str(_BPM),
            '--offset',
            '0',
            '--nps',
            '3',
        ],
    )
    assert result.exit_code == 0, result.output
    assert 'above the 10-point scale' in result.output


@pytest.mark.usefixtures('generation')
def test_pictures_are_drawn_alongside_the_simfile(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(
        main,
        [
            'generate',
            str(_audio(tmp_path / 'song.wav')),
            '-o',
            str(tmp_path / 'out'),
            '-T',
            'Song',
            '-c',
            str(_checkpoints(tmp_path)),
            '-D',
            'Easy',
            '--nps',
            '2',
            '--bpm',
            str(_BPM),
            '--offset',
            '0',
            '--image',
            '--svg',
        ],
    )
    assert result.exit_code == 0, result.output
    assert list((tmp_path / 'out' / 'Song' / IMAGE_DIRECTORY).glob('*.svg'))


@pytest.mark.usefixtures('generation')
def test_the_weights_repository_can_be_overridden(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(REPOSITORY_VARIABLE, raising=False)
    result = runner.invoke(
        main,
        [
            'generate',
            str(_audio(tmp_path / 'song.wav')),
            '-o',
            str(tmp_path / 'out'),
            '-T',
            'Song',
            '-c',
            str(_checkpoints(tmp_path)),
            '-D',
            'Easy',
            '--nps',
            '2',
            '--bpm',
            str(_BPM),
            '--offset',
            '0',
            '--weights-repo',
            'someone/weights',
            '--weights-revision',
            'v1',
        ],
    )
    assert result.exit_code == 0, result.output


def test_generation_without_weights_stops(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Resolution falls back to ./checkpoints, so the working directory has to be somewhere without
    # one.
    empty = tmp_path / 'checkpoints'
    empty.mkdir()
    (tmp_path / 'elsewhere').mkdir()
    monkeypatch.chdir(tmp_path / 'elsewhere')
    result = runner.invoke(
        main,
        [
            'generate',
            str(_audio(tmp_path / 'song.wav')),
            '-o',
            str(tmp_path / 'out'),
            '-T',
            'Song',
            '-c',
            str(empty),
            '--bpm',
            str(_BPM),
            '--offset',
            '0',
        ],
    )
    assert result.exit_code != 0
    assert 'Could not load the chart model' in result.output


@pytest.mark.usefixtures('generation')
def test_generation_without_the_stems_extra_stops(
    runner: CliRunner, tmp_path: Path, mocker: MockerFixture
) -> None:
    mocker.patch('smlab.main.load_separator', side_effect=SeparationError('install the extra'))
    result = runner.invoke(
        main,
        [
            'generate',
            str(_audio(tmp_path / 'song.wav')),
            '-o',
            str(tmp_path / 'out'),
            '-T',
            'Song',
            '-c',
            str(_checkpoints(tmp_path)),
            '--bpm',
            str(_BPM),
            '--offset',
            '0',
        ],
    )
    assert result.exit_code != 0
    assert 'install the extra' in result.output


def test_a_simfile_is_analyzed(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(main, ['analyze', str(_simfile(tmp_path))])
    assert result.exit_code == 0
    assert 'Test Song' in result.output
    assert 'Challenge' in result.output
    assert 'BPM 150.00-150.00' in result.output


def test_analyzing_a_simfile_without_timing_stops(runner: CliRunner, tmp_path: Path) -> None:
    path = tmp_path / 'broken.sm'
    path.write_text('#TITLE:No Timing;')
    result = runner.invoke(main, ['analyze', str(path)])
    assert result.exit_code != 0


def test_an_existing_simfile_is_drawn(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(main, ['image', str(_simfile(tmp_path))])
    assert result.exit_code == 0
    assert list((tmp_path / IMAGE_DIRECTORY).glob('*.png'))


def test_a_single_chart_can_be_drawn_to_a_named_file(runner: CliRunner, tmp_path: Path) -> None:
    destination = tmp_path / 'chart.svg'
    result = runner.invoke(
        main, ['image', str(_simfile(tmp_path)), '-o', str(destination), '--svg']
    )
    assert result.exit_code == 0
    assert destination.is_file()


def test_several_charts_go_into_a_named_directory(runner: CliRunner, tmp_path: Path) -> None:
    two = SM_TEXT + SM_TEXT.split('#NOTES:')[1].join(('#NOTES:', '')).replace('Challenge', 'Hard')
    folder = tmp_path / 'pictures'
    result = runner.invoke(main, ['image', str(_simfile(tmp_path, two)), '-o', str(folder)])
    assert result.exit_code == 0
    assert len(list(folder.glob('*.png'))) == 2


def test_drawing_a_simfile_with_no_charts_stops(runner: CliRunner, tmp_path: Path) -> None:
    path = tmp_path / 'empty.sm'
    path.write_text('#TITLE:Nothing;\n#BPMS:0.000=150.000;\n#OFFSET:0.0;\n')
    result = runner.invoke(main, ['image', str(path)])
    assert result.exit_code != 0
    assert 'no dance-single charts' in result.output


def test_publishing_lists_what_it_would_upload(runner: CliRunner, tmp_path: Path) -> None:
    directory = tmp_path / 'checkpoints'
    directory.mkdir()
    for name in (CHART_WEIGHTS, OFFSET_WEIGHTS):
        (directory / name).write_bytes(b'0' * 2048)
    result = runner.invoke(main, ['publish', '-c', str(directory), '-n', '-r', 'someone/weights'])
    assert result.exit_code == 0
    assert 'Would upload' in result.output
    assert CHART_WEIGHTS in result.output


def test_publishing_without_the_weights_stops(runner: CliRunner, tmp_path: Path) -> None:
    directory = tmp_path / 'checkpoints'
    directory.mkdir()
    result = runner.invoke(main, ['publish', '-c', str(directory), '-n'])
    assert result.exit_code != 0
    assert 'missing' in result.output


def test_publishing_uploads_every_checkpoint(
    runner: CliRunner, tmp_path: Path, mocker: MockerFixture
) -> None:
    directory = tmp_path / 'checkpoints'
    directory.mkdir()
    for name in (CHART_WEIGHTS, OFFSET_WEIGHTS):
        (directory / name).write_bytes(b'0' * 2048)
    api = mocker.patch('huggingface_hub.HfApi')
    result = runner.invoke(main, ['publish', '-c', str(directory), '-r', 'someone/weights'])
    assert result.exit_code == 0
    assert api.return_value.upload_file.call_count == 2


def test_a_failed_upload_is_reported(
    runner: CliRunner, tmp_path: Path, mocker: MockerFixture
) -> None:
    directory = tmp_path / 'checkpoints'
    directory.mkdir()
    for name in (CHART_WEIGHTS, OFFSET_WEIGHTS):
        (directory / name).write_bytes(b'0' * 2048)
    api = mocker.patch('huggingface_hub.HfApi')
    api.return_value.upload_file.side_effect = RuntimeError('no write token')
    result = runner.invoke(main, ['publish', '-c', str(directory), '-r', 'someone/weights'])
    assert result.exit_code != 0
    assert 'Upload failed' in result.output


def test_the_scales_are_offered_by_name() -> None:
    assert set(SCALES) == {10, 15, 20}
    assert TimingData.constant(_BPM, 0.0).primary_bpm == pytest.approx(_BPM)
    assert TOTAL_CHANNELS > 0


def test_an_unknown_difficulty_name_is_refused(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(main, ['generate', str(_audio(tmp_path / 'song.wav')), '-D', 'Ultra'])
    assert result.exit_code != 0
    assert 'is not one of' in result.output


@pytest.mark.parametrize('rating', ['zero', '0', '-2'])
def test_a_rating_that_is_not_a_rating_is_refused(
    runner: CliRunner, tmp_path: Path, rating: str
) -> None:
    result = runner.invoke(
        main, ['generate', str(_audio(tmp_path / 'song.wav')), '-D', f'Hard:{rating}']
    )
    assert result.exit_code != 0
    assert 'is not a rating' in result.output


def test_a_trained_offset_model_places_the_downbeat(
    runner: CliRunner, tmp_path: Path, mocker: MockerFixture
) -> None:
    # Folding picks its phase from the loudest point of the envelope, which is not where a downbeat
    # is, so the model replaces that offset.
    mocker.patch('smlab.main.EncoderConfig', return_value=_SMALL)
    mocker.patch('smlab.main.load_vocabulary', return_value=_VOCABULARY)
    mocker.patch('smlab.main.refine_offset', return_value=(-0.123, 0.9))
    mocker.patch('smlab.main.OffsetModel', return_value=_offset_model())
    mocker.patch('smlab.main.resolve_weights', return_value=tmp_path / 'offset.pt')
    mocker.patch('torch.load', return_value={'model': _offset_model().state_dict()})
    mocker.patch(
        'smlab.chart.gen.separate',
        side_effect=lambda _m, path, _d: dict.fromkeys(
            STEM_NAMES, np.asarray(sf.read(path)[0], dtype=np.float32)
        ),
    )
    checkpoints = _checkpoints(tmp_path)
    mocker.patch(
        'smlab.main._load_chart_model',
        return_value=(
            _chart_model(),
            _VOCABULARY,
            torch.device('cpu'),
        ),
    )

    mocker.patch('smlab.main.load_separator', return_value=_Separator())
    result = runner.invoke(
        main,
        [
            'generate',
            str(_audio(tmp_path / 'song.wav')),
            '-o',
            str(tmp_path / 'out'),
            '-T',
            'Song',
            '-c',
            str(checkpoints),
            '-D',
            'Easy',
            '--nps',
            '2',
        ],
    )
    assert result.exit_code == 0, result.output
    assert 'Placed the downbeat at offset -0.1230' in result.output


def test_weights_that_will_not_load_leave_the_offset_alone(
    runner: CliRunner, tmp_path: Path, mocker: MockerFixture
) -> None:
    # A missing offset model is not fatal: the tempo estimator supplies one of its own, just a
    # worse one.
    mocker.patch('smlab.main.EncoderConfig', return_value=_SMALL)
    mocker.patch('smlab.main.load_vocabulary', return_value=_VOCABULARY)
    mocker.patch('smlab.main.resolve_weights', side_effect=OSError)
    mocker.patch(
        'smlab.chart.gen.separate',
        side_effect=lambda _m, path, _d: dict.fromkeys(
            STEM_NAMES, np.asarray(sf.read(path)[0], dtype=np.float32)
        ),
    )
    mocker.patch(
        'smlab.main._load_chart_model',
        return_value=(_chart_model(), _VOCABULARY, torch.device('cpu')),
    )

    mocker.patch('smlab.main.load_separator', return_value=_Separator())
    result = runner.invoke(
        main,
        [
            'generate',
            str(_audio(tmp_path / 'song.wav')),
            '-o',
            str(tmp_path / 'out'),
            '-T',
            'Song',
            '-c',
            str(_checkpoints(tmp_path)),
            '-D',
            'Easy',
            '--nps',
            '2',
        ],
    )
    assert result.exit_code == 0, result.output
    assert 'Placed the downbeat' not in result.output


@pytest.mark.usefixtures('generation')
def test_a_preview_start_is_predicted_when_none_is_given(
    runner: CliRunner, tmp_path: Path, mocker: MockerFixture
) -> None:
    # Without a preview model the start falls back to a fixed fraction of the song rather than
    # failing the whole run.
    mocker.patch('smlab.main.load_state_dict', side_effect=OSError)
    result = runner.invoke(
        main,
        [
            'generate',
            str(_audio(tmp_path / 'song.wav')),
            '-o',
            str(tmp_path / 'out'),
            '-T',
            'Song',
            '-c',
            str(_checkpoints(tmp_path)),
            '-D',
            'Easy',
            '--nps',
            '2',
            '--bpm',
            str(_BPM),
            '--offset',
            '0',
        ],
    )
    assert result.exit_code == 0, result.output
    assert 'Preview starts at' in result.output


@pytest.mark.usefixtures('generation')
def test_an_untitled_song_is_named_after_its_file(runner: CliRunner, tmp_path: Path) -> None:
    audio = _audio(tmp_path / 'Nameless Track.wav')
    result = runner.invoke(
        main,
        [
            'generate',
            str(audio),
            '-o',
            str(tmp_path / 'out'),
            '-c',
            str(_checkpoints(tmp_path)),
            '-D',
            'Easy',
            '--nps',
            '2',
            '--bpm',
            str(_BPM),
            '--offset',
            '0',
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / 'out' / 'Nameless Track' / 'Nameless Track.ssc').is_file()


def test_analyzing_reports_why_a_chart_is_not_danceable(runner: CliRunner, tmp_path: Path) -> None:
    # A row needing three panels at once cannot be danced, and the report has to say which rule it
    # broke.
    hands = SM_TEXT.replace('1000\n0100\n0010\n0001', '1110\n0100\n0010\n0001')
    result = runner.invoke(main, ['analyze', str(_simfile(tmp_path, hands))])
    assert result.exit_code == 0
    assert 'style=hands' in result.output
    assert 'need three or four panels together' in result.output


@pytest.mark.usefixtures('generation')
def test_a_preview_start_can_be_given_outright(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(
        main,
        [
            'generate',
            str(_audio(tmp_path / 'song.wav')),
            '-o',
            str(tmp_path / 'out'),
            '-T',
            'Song',
            '-c',
            str(_checkpoints(tmp_path)),
            '-D',
            'Easy',
            '--nps',
            '2',
            '--bpm',
            str(_BPM),
            '--offset',
            '0',
            '--sample-start',
            '12.5',
        ],
    )
    assert result.exit_code == 0, result.output
    assert 'Preview starts at' not in result.output
    assert '#SAMPLESTART:12.500' in (tmp_path / 'out' / 'Song' / 'Song.ssc').read_text()


def test_analyzing_a_simfile_carrying_no_tempo_stops(
    runner: CliRunner, tmp_path: Path, mocker: MockerFixture
) -> None:
    # The loaders raise rather than return a Simfile with no tempo, so the guard against one has to
    # be reached directly.
    path = _simfile(tmp_path)
    mocker.patch('smlab.main.load_simfile', return_value=replace(load_simfile(path), timing=None))
    result = runner.invoke(main, ['analyze', str(path)])
    assert result.exit_code != 0
    assert 'No usable timing' in result.output


@pytest.mark.usefixtures('generation')
def test_a_sm_file_is_written_when_asked_for(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(
        main,
        [
            'generate',
            str(_audio(tmp_path / 'song.wav')),
            '-o',
            str(tmp_path / 'out'),
            '-T',
            'Song',
            '-c',
            str(_checkpoints(tmp_path)),
            '-D',
            'Easy',
            '--nps',
            '2',
            '--bpm',
            str(_BPM),
            '--offset',
            '0',
            '--format',
            'sm',
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / 'out' / 'Song' / 'Song.sm').is_file()
    assert not (tmp_path / 'out' / 'Song' / 'Song.ssc').exists()
