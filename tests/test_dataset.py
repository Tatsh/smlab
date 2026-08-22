"""Tests for resampling audio onto the beat grid and building targets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]  # No stubs are published.

from smlab.chart import DIFFICULTIES, Chart
from smlab.dataset import (
    CODE_BY_CHAR,
    FEATURE_DIMENSION,
    SUBDIVISIONS_PER_BEAT,
    beat_features,
    chart_targets,
    difficulty_index,
    load_song_features,
    measure_position,
    placement_vector,
)
from smlab.timing import TimingData

if TYPE_CHECKING:
    from pathlib import Path

_RATE = 22050
_BPM = 120.0
_TIMING = TimingData.constant(_BPM, 0.0)


def _tone(seconds: float = 4.0) -> np.ndarray:
    times = np.arange(int(_RATE * seconds), dtype=np.float32) / _RATE
    return np.sin(2.0 * np.pi * 440.0 * times).astype(np.float32)


def _chart(measures: str, difficulty: str = 'Challenge', meter: int = 9) -> Chart:
    return Chart(difficulty, meter, 'dance-single', raw_notes=measures)


@pytest.mark.parametrize('name', list(DIFFICULTIES))
def test_every_known_difficulty_has_its_own_index(name: str) -> None:
    assert difficulty_index(name) == DIFFICULTIES.index(name)


def test_an_unknown_difficulty_falls_back_to_the_edit_slot() -> None:
    assert difficulty_index('Ultra') == len(DIFFICULTIES) - 1


def test_features_are_one_row_per_grid_slot() -> None:
    seconds = 4.0
    features = beat_features(_tone(seconds), _TIMING, sample_rate=_RATE)
    # Four seconds at 120 BPM is eight beats.
    assert features.shape == (int(8 * SUBDIVISIONS_PER_BEAT), FEATURE_DIMENSION)
    assert features.dtype == np.float16


def test_a_slot_past_the_end_of_the_audio_stops_the_fill() -> None:
    # An offset far into the future puts every slot beyond the samples, so the loop must break
    # rather than index past the spectrogram.
    late = TimingData.constant(_BPM, -30.0)
    features = beat_features(_tone(2.0), late, sample_rate=_RATE)
    assert float(np.abs(features.astype(np.float32)).max()) == pytest.approx(0.0)


def test_silence_and_tone_produce_different_features() -> None:
    silent = beat_features(np.zeros(_RATE * 2, dtype=np.float32), _TIMING, sample_rate=_RATE)
    played = beat_features(_tone(2.0), _TIMING, sample_rate=_RATE)
    assert not np.array_equal(silent, played)


def test_rows_are_converted_onto_the_grid() -> None:
    targets = chart_targets(_chart('1000\n0100\n0010\n0001'), 96)
    assert len(targets) == 4
    # A measure of quarter notes is one row every twelve slots.
    assert list(targets.slots) == [0, 12, 24, 36]
    assert list(targets.panels[0]) == [1, 0, 0, 0]
    assert targets.difficulty == 'Challenge'
    assert targets.meter == 9


def test_rows_beyond_the_grid_are_dropped() -> None:
    # The audio ran out before the chart did.
    targets = chart_targets(_chart('1000\n0100\n0010\n0001'), 20)
    assert list(targets.slots) == [0, 12]


def test_a_chart_with_nothing_on_the_grid_yields_empty_targets() -> None:
    targets = chart_targets(_chart('1000\n0100\n0010\n0001'), 0)
    assert len(targets) == 0
    assert targets.panels.shape == (0, 4)
    assert targets.difficulty == 'Challenge'


def test_a_short_row_is_padded_to_four_panels() -> None:
    targets = chart_targets(_chart('10'), 96)
    assert list(targets.panels[0]) == [1, 0, 0, 0]


def test_placement_marks_only_the_slots_that_are_stepped_on() -> None:
    # A mine is not a step, so it must not appear in the placement target.
    targets = chart_targets(_chart('1000\n0000\n5000\n0000'), 96)
    vector = placement_vector(targets, 96)
    assert vector[0] == pytest.approx(1.0)
    assert vector[24] == pytest.approx(0.0, abs=1e-12)
    assert float(vector.sum()) == pytest.approx(1.0)


def test_placement_over_an_empty_chart_is_all_zero() -> None:
    vector = placement_vector(chart_targets(_chart('0000'), 96), 96)
    assert float(vector.sum()) == pytest.approx(0.0, abs=1e-12)


def test_releasing_a_freeze_is_not_a_new_step() -> None:
    # The head is stepped on, the tail is only the foot coming off it.
    targets = chart_targets(_chart('2000\n0000\n3000\n0000'), 96)
    vector = placement_vector(targets, 96)
    assert vector[0] == pytest.approx(1.0)
    assert vector[24] == pytest.approx(0.0, abs=1e-12)
    assert CODE_BY_CHAR['M'] == 5


def test_metric_position_repeats_every_measure() -> None:
    positions = measure_position(200)
    per_measure = 4 * SUBDIVISIONS_PER_BEAT
    assert positions[0] == 0
    assert positions[per_measure] == 0
    assert int(positions.max()) == per_measure - 1


def test_features_can_be_built_straight_from_a_file(tmp_path: Path) -> None:
    path = tmp_path / 'tone.wav'
    sf.write(path, _tone(2.0), _RATE)
    assert load_song_features(path, _TIMING).shape[1] == FEATURE_DIMENSION
