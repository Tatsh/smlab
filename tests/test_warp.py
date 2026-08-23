"""Tests for measuring how a song's tempo wanders."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]  # No stubs are published.

from smlab.tempo import PHASE_PARAMS
from smlab.warp import measure_tempo

if TYPE_CHECKING:
    from pathlib import Path

_RATE = PHASE_PARAMS.sample_rate
_FINE = {'window': 4.0, 'hop': 1.0, 'span': 10.0}


def _clicks(path: Path, tempi: tuple[tuple[float, float], ...]) -> Path:
    """Write a click track, each entry giving a tempo and how many seconds to hold it."""
    samples: list[np.ndarray] = []
    for bpm, seconds in tempi:
        block = np.zeros(int(_RATE * seconds), dtype=np.float32)
        block[:: int(_RATE * 60.0 / bpm)] = 1.0
        samples.append(block)
    sf.write(path, np.concatenate(samples), _RATE)
    return path


def test_a_steady_song_reads_the_same_tempo_throughout(tmp_path: Path) -> None:
    readings = measure_tempo(_clicks(tmp_path / 'steady.wav', ((128.0, 40.0),)), 128.0, **_FINE)
    assert readings
    assert all(reading.bpm == pytest.approx(128.0, abs=0.3) for reading in readings)
    assert max(abs(reading.slip) for reading in readings) < 0.02


def test_a_song_that_changes_tempo_reads_the_change(tmp_path: Path) -> None:
    # Measured against the tempo it starts at, the second half has to come back faster.
    path = _clicks(tmp_path / 'changing.wav', ((128.0, 30.0), (136.0, 30.0)))
    readings = measure_tempo(path, 128.0, **_FINE)
    assert readings
    early = [reading.bpm for reading in readings if reading.seconds < 15.0]
    late = [reading.bpm for reading in readings if reading.seconds > 45.0]
    assert early
    assert late
    assert min(late) > max(early)


def test_the_slip_says_how_far_the_grid_has_moved(tmp_path: Path) -> None:
    # A grid running slow against the music loses time across the stretch, so the slip is signed
    # and grows with how wrong the reference is.
    path = _clicks(tmp_path / 'fast.wav', ((130.0, 40.0),))
    readings = measure_tempo(path, 128.0, **_FINE)
    assert readings
    assert all(reading.slip < 0 for reading in readings)


def test_a_song_too_short_to_fit_a_stretch_reads_nothing(tmp_path: Path) -> None:
    assert measure_tempo(_clicks(tmp_path / 'brief.wav', ((128.0, 6.0),)), 128.0, **_FINE) == []


def test_a_file_shorter_than_one_window_reads_nothing(tmp_path: Path) -> None:
    # There are not enough envelope frames to take a phase from at all.
    sf.write(tmp_path / 'tiny.wav', np.zeros(64, dtype=np.float32), _RATE)
    assert measure_tempo(tmp_path / 'tiny.wav', 128.0, **_FINE) == []


def test_silence_reads_nothing(tmp_path: Path) -> None:
    # Every window is flat, so no window has a phase to give.
    sf.write(tmp_path / 'silent.wav', np.zeros(int(_RATE * 40), dtype=np.float32), _RATE)
    assert measure_tempo(tmp_path / 'silent.wav', 128.0, **_FINE) == []
