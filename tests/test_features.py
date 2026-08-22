"""Tests for the fine-grid stem features."""

from __future__ import annotations

from smlab.features import (
    FINE_SUBDIVISIONS,
    MIXTURE_MELS,
    STEM_CHANNELS,
    STEM_MELS,
    TOTAL_CHANNELS,
    fine_features,
    grid_times,
)
from smlab.stems import STEM_NAMES
from smlab.timing import TimingData
import numpy as np
import pytest

_RATE = 22050
_BPM = 120.0
_TIMING = TimingData.constant(_BPM, 0.0)


def _tone(seconds: float = 2.0, hertz: float = 440.0) -> np.ndarray:
    times = np.arange(int(_RATE * seconds), dtype=np.float32) / _RATE
    return np.sin(2.0 * np.pi * hertz * times).astype(np.float32)


def test_the_channel_width_accounts_for_every_layer() -> None:
    assert STEM_CHANNELS == STEM_MELS + 2
    assert len(STEM_NAMES) * STEM_CHANNELS + MIXTURE_MELS + 2 == TOTAL_CHANNELS


def test_the_grid_runs_at_twice_the_note_resolution() -> None:
    times = grid_times(_TIMING, 2.0)
    # Two seconds at 120 BPM is four beats.
    assert len(times) == 4 * FINE_SUBDIVISIONS
    assert times[0] == pytest.approx(0.0)
    assert float(np.diff(times).min()) > 0.0


def test_a_grid_never_comes_back_empty() -> None:
    # A song shorter than one slot still needs somewhere to put its features.
    assert len(grid_times(_TIMING, 0.0)) == 1


def test_an_offset_past_the_end_of_the_audio_still_grids() -> None:
    # A negative beat count must not produce a negative slot count.
    assert len(grid_times(TimingData.constant(_BPM, 5.0), 1.0)) >= 1


def test_features_span_every_stem_and_the_mixture() -> None:
    mixture = _tone()
    stems = {name: _tone(hertz=220.0) for name in STEM_NAMES}
    features = fine_features(stems, mixture, _TIMING, sample_rate=_RATE)
    assert features.shape == (4 * FINE_SUBDIVISIONS, TOTAL_CHANNELS)
    assert features.dtype == np.float16


def test_a_missing_stem_is_filled_with_silence() -> None:
    # Separation can fail for one layer; the block still has to be the right
    # width or every downstream offset shifts.
    mixture = _tone()
    partial = fine_features({STEM_NAMES[0]: _tone()}, mixture, _TIMING, sample_rate=_RATE)
    assert partial.shape[1] == TOTAL_CHANNELS


def test_a_loud_stem_reads_differently_from_a_silent_one() -> None:
    mixture = _tone()
    loud = fine_features(dict.fromkeys(STEM_NAMES, _tone()), mixture, _TIMING, sample_rate=_RATE)
    quiet = fine_features({}, mixture, _TIMING, sample_rate=_RATE)
    assert not np.array_equal(loud, quiet)
