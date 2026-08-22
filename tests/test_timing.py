"""Tests for beat and time conversion."""

from __future__ import annotations

from smlab.timing import BpmSegment, StopSegment, TimingData, gap_ms_to_offset, offset_to_gap_ms
import pytest


@pytest.mark.parametrize('offset', [-0.048, 0.0, 0.5, -16.011, 3.974])
def test_beat_zero_sits_at_negative_offset(offset: float) -> None:
    timing = TimingData.constant(150.0, offset)
    assert timing.time_at_beat(0.0) == pytest.approx(-offset)


@pytest.mark.parametrize(
    ('bpm', 'beat', 'expected'),
    [
        (150.0, 4.0, 1.6),
        (120.0, 4.0, 2.0),
        (60.0, 1.0, 1.0),
        (200.0, 8.0, 2.4),
    ],
)
def test_time_advances_with_tempo(bpm: float, beat: float, expected: float) -> None:
    timing = TimingData.constant(bpm, 0.0)
    assert timing.time_at_beat(beat) == pytest.approx(expected)


@pytest.mark.parametrize('beat', [0.0, 1.0, 7.5, 64.0, 191.25])
def test_beat_and_time_round_trip(beat: float) -> None:
    timing = TimingData.constant(147.5, -0.032)
    assert timing.beat_at_time(timing.time_at_beat(beat)) == pytest.approx(beat)


@pytest.mark.parametrize(('gap_ms', 'offset'), [(48, -0.048), (0, 0.0), (-250, 0.25), (1000, -1.0)])
def test_dwi_gap_converts_to_offset(gap_ms: float, offset: float) -> None:
    assert gap_ms_to_offset(gap_ms) == pytest.approx(offset)
    assert offset_to_gap_ms(offset) == pytest.approx(gap_ms)


def test_stop_delays_later_beats() -> None:
    timing = TimingData(bpms=(BpmSegment(0.0, 120.0),), offset=0.0, stops=(StopSegment(4.0, 1.5),))
    assert timing.time_at_beat(4.0) == pytest.approx(2.0 + 1.5)
    assert timing.time_at_beat(8.0) == pytest.approx(4.0 + 1.5)
    assert timing.time_at_beat(2.0) == pytest.approx(1.0)


def test_tempo_change_alters_later_beats() -> None:
    timing = TimingData(bpms=(BpmSegment(0.0, 120.0), BpmSegment(4.0, 240.0)), offset=0.0)
    assert timing.time_at_beat(4.0) == pytest.approx(2.0)
    assert timing.time_at_beat(8.0) == pytest.approx(3.0)
    assert timing.bpm_at_beat(0.0) == pytest.approx(120.0)
    assert timing.bpm_at_beat(5.0) == pytest.approx(240.0)


def test_constant_bpm_detection() -> None:
    assert TimingData.constant(150.0, 0.0).is_constant_bpm
    assert not TimingData(bpms=(BpmSegment(0.0, 120.0), BpmSegment(4.0, 200.0))).is_constant_bpm
    assert not TimingData(
        bpms=(BpmSegment(0.0, 120.0),), stops=(StopSegment(1.0, 0.5),)
    ).is_constant_bpm


def test_shifting_moves_beat_zero_later() -> None:
    timing = TimingData.constant(150.0, -0.048)
    shifted = timing.shifted(0.1)
    assert shifted.time_at_beat(0.0) == pytest.approx(timing.time_at_beat(0.0) + 0.1)


def test_empty_bpms_is_rejected() -> None:
    with pytest.raises(ValueError, match='at least one BPM segment'):
        TimingData(bpms=())
