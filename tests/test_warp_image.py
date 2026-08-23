"""Tests for drawing the beat grid over the audio it belongs to."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image
import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]  # No stubs are published.

from smlab.tempo import PHASE_PARAMS
from smlab.warp import DEFAULT_ROW_SECONDS, Warp, WarpFit, write_drift

if TYPE_CHECKING:
    from pathlib import Path

_RATE = PHASE_PARAMS.sample_rate
_BPM = 120.0
_PERIOD = 60.0 / _BPM
_ORIGIN = 1.5
_LEFT = 76
_RIGHT = 16
_WIDTH = 1800
_TOP = 58
_ROW = 132
_BAR = (120, 150, 226)
_SPLICE = (224, 138, 32)


def _clicks(path: Path, seconds: float, *, origin: float = _ORIGIN, gap: float = _PERIOD) -> Path:
    """Write a click track whose beats start at a known moment."""
    track = np.zeros(int(_RATE * seconds), dtype=np.float32)
    for beat in np.arange(origin, seconds, gap):
        at = int(beat * _RATE)
        track[at : at + 300] = 0.5
    # Every row is drawn against the loudest sample in the song, so one louder sample keeps the
    # clicks short enough to leave the top of each row clear for the grid to be read off.
    track[0] = 1.0
    sf.write(path, track, _RATE)
    return path


def _columns_matching(picture: Image.Image, colour: tuple[int, int, int], row: int) -> list[float]:
    """Return one column position per run of the given colour across the top of a row."""
    pixels = np.asarray(picture.convert('RGB')).astype(int)
    top = _TOP + row * (_ROW + 16)
    # Below the bar number, which is written in the same colour as the line it labels, and above
    # where the waveform reaches.
    band = pixels[top + 18 : top + 22, :, :]
    found = np.flatnonzero(np.all(np.abs(band - np.array(colour)) < 14, axis=2).any(axis=0))
    runs: list[float] = []
    for column in found:
        if runs and column - runs[-1] <= 4:
            runs[-1] = float(column)
            continue
        runs.append(float(column))
    return runs


def _at(column: float) -> float:
    """Turn a pixel column into the moment the grid puts there."""
    return (column - _LEFT) / (_WIDTH - _LEFT - _RIGHT) * DEFAULT_ROW_SECONDS


def test_a_supplied_offset_is_where_the_grid_goes(tmp_path: Path) -> None:
    # The bar lines have to land on the moment asked for rather than near it, because the whole
    # point of the picture is judging a grid against attacks a few milliseconds wide.
    fit = WarpFit(warps=[Warp(seconds=0.0, bpm=_BPM)], splices=[])
    origin = write_drift(
        tmp_path / 'grid.png', _clicks(tmp_path / 'song.wav', 24.0), _BPM, fit, origin=_ORIGIN
    )
    assert origin == pytest.approx(_ORIGIN)
    bars = _columns_matching(Image.open(tmp_path / 'grid.png'), _BAR, 0)
    # A bar is four beats, so they fall every two seconds from the offset. The one before it lands
    # at minus half a second, which is off the front of the song and so is not drawn.
    assert [_at(column) for column in bars] == pytest.approx(
        [_ORIGIN, _ORIGIN + 2.0, _ORIGIN + 4.0, _ORIGIN + 6.0], abs=0.01
    )


def test_the_grid_reaches_back_before_beat_zero(tmp_path: Path) -> None:
    # An offset of a few seconds would otherwise leave the opening with no grid over it at all.
    fit = WarpFit(warps=[Warp(seconds=0.0, bpm=_BPM)], splices=[])
    write_drift(
        tmp_path / 'grid.png',
        _clicks(tmp_path / 'song.wav', 24.0, origin=3.5),
        _BPM,
        fit,
        origin=3.5,
    )
    bars = _columns_matching(Image.open(tmp_path / 'grid.png'), _BAR, 0)
    assert min(_at(column) for column in bars) < 3.5


def test_the_offset_is_found_when_it_is_not_given(tmp_path: Path) -> None:
    fit = WarpFit(warps=[Warp(seconds=0.0, bpm=_BPM)], splices=[])
    origin = write_drift(
        tmp_path / 'grid.png', _clicks(tmp_path / 'song.wav', 30.0), _BPM, fit, origin=None
    )
    assert origin % _PERIOD == pytest.approx(_ORIGIN % _PERIOD, abs=0.03)


def test_silence_is_drawn_without_a_beat_to_find(tmp_path: Path) -> None:
    sf.write(tmp_path / 'quiet.wav', np.zeros(int(_RATE * 20.0), dtype=np.float32), _RATE)
    fit = WarpFit(warps=[Warp(seconds=0.0, bpm=_BPM)], splices=[])
    origin = write_drift(tmp_path / 'grid.png', tmp_path / 'quiet.wav', _BPM, fit)
    assert origin == pytest.approx(0.0, abs=1e-12)


def test_a_jump_in_the_beat_is_marked(tmp_path: Path) -> None:
    fit = WarpFit(warps=[Warp(seconds=0.0, bpm=_BPM)], splices=[3.0])
    write_drift(
        tmp_path / 'grid.png', _clicks(tmp_path / 'song.wav', 20.0), _BPM, fit, origin=_ORIGIN
    )
    marks = _columns_matching(Image.open(tmp_path / 'grid.png'), _SPLICE, 0)
    assert [_at(column) for column in marks] == pytest.approx([3.0], abs=0.02)


def test_a_second_tempo_spaces_the_later_beats_differently(tmp_path: Path) -> None:
    # The grid follows the tempo segments, so a faster stretch has to draw more beats into a row.
    path = _clicks(tmp_path / 'song.wav', 40.0)
    steady = WarpFit(warps=[Warp(seconds=0.0, bpm=_BPM)], splices=[])
    warped = WarpFit(
        warps=[Warp(seconds=0.0, bpm=_BPM), Warp(seconds=8.0, bpm=_BPM * 2)], splices=[]
    )
    write_drift(tmp_path / 'one.png', path, _BPM, steady, origin=_ORIGIN)
    write_drift(tmp_path / 'two.png', path, _BPM, warped, origin=_ORIGIN)
    later = 1
    assert len(_columns_matching(Image.open(tmp_path / 'two.png'), _BAR, later)) > len(
        _columns_matching(Image.open(tmp_path / 'one.png'), _BAR, later)
    )


def test_a_tempo_is_used_when_the_fit_has_nothing_to_say(tmp_path: Path) -> None:
    write_drift(
        tmp_path / 'grid.png',
        _clicks(tmp_path / 'song.wav', 20.0),
        _BPM,
        WarpFit(warps=[], splices=[]),
        origin=_ORIGIN,
    )
    bars = _columns_matching(Image.open(tmp_path / 'grid.png'), _BAR, 0)
    # A bar is four beats, so they fall every two seconds from the offset. The one before it lands
    # at minus half a second, which is off the front of the song and so is not drawn.
    assert [_at(column) for column in bars] == pytest.approx(
        [_ORIGIN, _ORIGIN + 2.0, _ORIGIN + 4.0, _ORIGIN + 6.0], abs=0.01
    )


def test_every_second_of_the_song_gets_a_row(tmp_path: Path) -> None:
    fit = WarpFit(warps=[Warp(seconds=0.0, bpm=_BPM)], splices=[])
    write_drift(
        tmp_path / 'grid.png',
        _clicks(tmp_path / 'song.wav', 21.0),
        _BPM,
        fit,
        origin=_ORIGIN,
        row_seconds=8.0,
    )
    assert Image.open(tmp_path / 'grid.png').height == _TOP + 3 * (_ROW + 16)
