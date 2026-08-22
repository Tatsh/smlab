"""Tests for beat and time conversion."""

from __future__ import annotations

import pytest

from smlab.timing import BPMSegment, StopSegment, TimingData, gap_ms_to_offset, offset_to_gap_ms


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
    timing = TimingData(bpms=(BPMSegment(0.0, 120.0),), offset=0.0, stops=(StopSegment(4.0, 1.5),))
    assert timing.time_at_beat(4.0) == pytest.approx(2.0 + 1.5)
    assert timing.time_at_beat(8.0) == pytest.approx(4.0 + 1.5)
    assert timing.time_at_beat(2.0) == pytest.approx(1.0)


def test_tempo_change_alters_later_beats() -> None:
    timing = TimingData(bpms=(BPMSegment(0.0, 120.0), BPMSegment(4.0, 240.0)), offset=0.0)
    assert timing.time_at_beat(4.0) == pytest.approx(2.0)
    assert timing.time_at_beat(8.0) == pytest.approx(3.0)
    assert timing.bpm_at_beat(0.0) == pytest.approx(120.0)
    assert timing.bpm_at_beat(5.0) == pytest.approx(240.0)


def test_constant_bpm_detection() -> None:
    assert TimingData.constant(150.0, 0.0).is_constant_bpm
    assert not TimingData(bpms=(BPMSegment(0.0, 120.0), BPMSegment(4.0, 200.0))).is_constant_bpm
    assert not TimingData(
        bpms=(BPMSegment(0.0, 120.0),), stops=(StopSegment(1.0, 0.5),)
    ).is_constant_bpm


def test_shifting_moves_beat_zero_later() -> None:
    timing = TimingData.constant(150.0, -0.048)
    shifted = timing.shifted(0.1)
    assert shifted.time_at_beat(0.0) == pytest.approx(timing.time_at_beat(0.0) + 0.1)


def test_empty_bpms_is_rejected() -> None:
    with pytest.raises(ValueError, match='at least one BPM segment'):
        TimingData(bpms=())


def test_a_time_inside_a_stop_holds_the_beat_still() -> None:
    # The chart waits at the stop's beat for its whole duration.
    timing = TimingData((BPMSegment(0.0, 120.0),), stops=(StopSegment(4.0, 1.0),))
    # Beat 4 arrives a second late because the stop sits in front of it, so
    # the frozen window is the second before that moment.
    assert timing.time_at_beat(4.0) == pytest.approx(3.0)
    assert timing.beat_at_time(2.5) == pytest.approx(4.0)


def test_a_tempo_declared_after_beat_zero_still_governs_from_it() -> None:
    # StepMania treats the first tempo as governing from beat 0 whatever beat
    # the tag names, so the two files describe the same song.
    late = TimingData((BPMSegment(8.0, 120.0),))
    early = TimingData((BPMSegment(0.0, 120.0),))
    assert late.time_at_beat(4.0) == pytest.approx(early.time_at_beat(4.0))


def test_the_tempo_at_a_beat_is_reported() -> None:
    timing = TimingData((BPMSegment(0.0, 120.0), BPMSegment(4.0, 180.0)))
    assert timing.bpm_at_beat(0.0) == pytest.approx(120.0)
    assert timing.bpm_at_beat(6.0) == pytest.approx(180.0)


def test_a_time_before_beat_zero_extrapolates_backwards() -> None:
    # An intro shorter than the offset still needs a beat number.
    timing = TimingData((BPMSegment(0.0, 120.0),), offset=-1.0)
    assert timing.beat_at_time(0.0) == pytest.approx(-2.0)


def test_the_tempo_range_spans_every_segment() -> None:
    timing = TimingData((
        BPMSegment(0.0, 120.0),
        BPMSegment(4.0, 180.0),
        BPMSegment(8.0, 90.0),
    ))
    low, high = timing.bpm_range()
    assert low == pytest.approx(90.0)
    assert high == pytest.approx(180.0)


def test_the_primary_tempo_is_the_one_at_beat_zero() -> None:
    timing = TimingData((BPMSegment(0.0, 120.0), BPMSegment(4.0, 180.0)))
    assert timing.primary_bpm == pytest.approx(120.0)


def test_a_stop_and_a_tempo_change_on_one_beat_are_both_applied() -> None:
    # StepMania orders the tempo change first, so the pause is measured in
    # seconds and the beats after it run at the new tempo.
    timing = TimingData(
        (BPMSegment(0.0, 120.0), BPMSegment(4.0, 240.0)),
        stops=(StopSegment(4.0, 1.0),),
    )
    # Four beats at 120 BPM is two seconds, then a second of silence.
    assert timing.time_at_beat(4.0) == pytest.approx(3.0)
    # The four beats after it run at the doubled tempo.
    assert timing.time_at_beat(8.0) == pytest.approx(4.0)
