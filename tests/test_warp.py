"""Tests for measuring how a song's tempo wanders."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]  # No stubs are published.

from smlab.tempo import PHASE_PARAMS
from smlab.warp import fit_warps, measure_tempo

if TYPE_CHECKING:
    from pathlib import Path

_RATE = PHASE_PARAMS.sample_rate
_FINE = {'window': 4.0, 'hop': 1.0, 'span': 10.0}
_FIT = {'window': 8.0, 'hop': 1.0, 'shortest': 20.0}


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


def test_one_tempo_is_fitted_to_a_song_that_holds_one(tmp_path: Path) -> None:
    path = _clicks(tmp_path / 'steady.wav', ((128.0, 120.0),))
    fit = fit_warps(path, 128.0, **_FIT)
    assert [warp.seconds for warp in fit.warps] == [0.0]
    assert fit.warps[0].bpm == pytest.approx(128.0, abs=0.05)
    assert fit.splices == []


def test_a_song_that_changes_tempo_is_fitted_a_segment_for_each(tmp_path: Path) -> None:
    path = _clicks(tmp_path / 'changing.wav', ((128.0, 60.0), (136.0, 60.0)))
    fit = fit_warps(path, 128.0, **_FIT)
    assert len(fit.warps) > 1
    assert fit.warps[0].seconds == pytest.approx(0.0, abs=1e-12)
    assert fit.warps[0].bpm < fit.warps[-1].bpm
    assert fit.warps[-1].bpm == pytest.approx(136.0, abs=0.5)
    # A change of speed is not a jump, so nothing here should be called an edit.
    assert fit.splices == []


def test_a_change_of_instruments_alone_is_not_read_as_a_change_of_tempo(tmp_path: Path) -> None:
    # Off-beat hits entering part way through move where the beat is measured without moving the
    # beat. That is a step in the phase rather than a change of slope, and writing it as a tempo
    # would put a tempo in the chart the song never plays.
    step = int(_RATE * 60.0 / 128.0)
    track = np.zeros(int(_RATE * 120.0), dtype=np.float32)
    track[::step] = 1.0
    track[step // 2 + len(track) // 2 :: step] = 0.7
    sf.write(path := tmp_path / 'rearranged.wav', track, _RATE)
    fit = fit_warps(path, 128.0, **_FIT)
    assert [warp.seconds for warp in fit.warps] == [0.0]
    # One segment is the point of this. The tempo is allowed a little slack because a squared fit
    # leans into the phase step the entry leaves behind, which is a known and measured bias.
    assert fit.warps[0].bpm == pytest.approx(128.0, abs=0.2)


def test_a_song_that_slips_once_but_keeps_its_tempo_is_still_one_tempo(tmp_path: Path) -> None:
    # A splice moves every beat after it by the same amount without changing the spacing between
    # them. Both halves hold the same tempo, so there is nothing for a tempo marker to say, and the
    # jump is worth reporting because no tempo will put it right.
    step = int(_RATE * 60.0 / 128.0)
    track = np.zeros(int(_RATE * 120.0), dtype=np.float32)
    track[: _RATE * 60 : step] = 1.0
    track[_RATE * 60 + int(_RATE * 0.06) :: step] = 1.0
    sf.write(path := tmp_path / 'spliced.wav', track, _RATE)
    fit = fit_warps(path, 128.0, **_FIT)
    assert [warp.seconds for warp in fit.warps] == [0.0]
    assert fit.warps[0].bpm == pytest.approx(128.0, abs=0.1)
    assert fit.splices == pytest.approx([60.0], abs=10.0)


def test_an_abrupt_change_of_tempo_is_kept_even_though_it_looks_like_a_jump(
    tmp_path: Path,
) -> None:
    # A tempo that leaves and comes back is treated as the beat jumping rather than as music, and
    # a clean change reads that way at first because the window straddling it reports a tempo
    # belonging to neither side. What separates them is that here the two sides genuinely differ,
    # so the change has to survive.
    path = _clicks(tmp_path / 'step.wav', ((128.0, 100.0), (130.0, 100.0)))
    fit = fit_warps(path, 128.0)
    assert len(fit.warps) > 1
    assert fit.warps[0].bpm == pytest.approx(128.0, abs=0.2)
    assert fit.warps[-1].bpm == pytest.approx(130.0, abs=0.2)
    assert fit.splices == []


def _wandering(
    path: Path, seconds: float, *, amplitude: float, cycle: float, steps: tuple[float, ...] = ()
) -> Path:
    """Write a click track at one tempo whose beats slide back and forth, and may also jump."""
    track = np.zeros(int(_RATE * seconds), dtype=np.float32)
    period = 60.0 / 128.0
    for index in range(int(seconds / period)):
        at = index * period + amplitude * np.sin(2.0 * np.pi * index * period / cycle)
        at += 0.05 * sum(1 for step in steps if index * period > step)
        start = int(at * _RATE)
        if 0 <= start < len(track) - 300:
            track[start : start + 300] = 1.0
    sf.write(path, track, _RATE)
    return path


def test_a_beat_that_wanders_and_returns_keeps_one_tempo(tmp_path: Path) -> None:
    # The grid cannot hold within the tolerance here, so the track is cut into pieces, but every
    # piece reads the same tempo and a tempo change that changes no tempo is not worth writing.
    path = _wandering(tmp_path / 'wander.wav', 150.0, amplitude=0.03, cycle=40.0)
    fit = fit_warps(path, 128.0)
    assert [warp.seconds for warp in fit.warps] == [0.0]
    assert fit.warps[0].bpm == pytest.approx(128.0, abs=0.1)
    assert fit.splices == []
    # It says so, rather than pretending the grid fits.
    assert fit.slack > 0.020


def test_a_bend_that_stops_paying_off_once_refitted_is_dropped(tmp_path: Path) -> None:
    # Where the bends go is decided on stretches judged separately, and the tempi then come off one
    # line fitted through all of them at once, which can leave a bend separating two tempi that no
    # longer differ enough to be worth a marker.
    path = _wandering(tmp_path / 'both.wav', 200.0, amplitude=0.02, cycle=35.0, steps=(60.0,))
    fit = fit_warps(path, 128.0)
    assert [warp.seconds for warp in fit.warps] == [0.0]
    assert fit.warps[0].bpm == pytest.approx(128.0, abs=0.2)


def test_a_song_too_short_to_fit_a_segment_is_fitted_nothing(tmp_path: Path) -> None:
    fit = fit_warps(_clicks(tmp_path / 'brief.wav', ((128.0, 6.0),)), 128.0, **_FIT)
    assert fit.warps == []
    assert fit.splices == []
