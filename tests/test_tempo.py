"""Tests for recovering tempo and offset from an onset envelope."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]  # No stubs are published.

from smlab.tempo import (
    ONSET_LATENCY_SECONDS,
    TEMPO_PARAMS,
    Envelopes,
    estimate_tempo,
    estimate_timing,
    estimate_timing_from_envelopes,
    snap_bpm,
)

if TYPE_CHECKING:
    from pathlib import Path

_RATE = TEMPO_PARAMS.frame_rate
_BPM = 120.0
_SECONDS = 40.0


def _pulses(bpm: float = _BPM, seconds: float = _SECONDS, phase: float = 0.0) -> np.ndarray:
    """Build an onset envelope carrying one impulse per beat."""
    envelope = np.zeros(int(seconds * _RATE), dtype=np.float32)
    period = 60.0 / bpm * _RATE
    beats = np.arange(phase * _RATE, len(envelope), period)
    envelope[beats.astype(np.int64)] = 1.0
    return envelope


@pytest.mark.parametrize(
    ('value', 'wanted'),
    [
        # Corpus tempi are overwhelmingly whole or half beats per minute.
        (169.99, 170.0),
        (169.6, 169.5),
        (169.3, 169.5),
    ],
)
def test_a_tempo_close_to_a_round_one_is_snapped(value: float, wanted: float) -> None:
    assert snap_bpm(value) == pytest.approx(wanted)


def test_a_tempo_too_far_from_a_round_one_is_left_alone() -> None:
    # Halves are 0.5 apart, so nothing sits more than 0.25 from one and the default tolerance snaps
    # everything. Only a narrower window can refuse.
    assert snap_bpm(169.3, tolerance=0.1) == pytest.approx(169.3)


def test_a_steady_pulse_recovers_its_tempo() -> None:
    bpm, confidence = estimate_tempo(
        _pulses().astype(np.float64), _RATE, min_bpm=60.0, max_bpm=300.0
    )
    assert bpm == pytest.approx(_BPM)
    assert confidence > 1.0


def test_an_envelope_shorter_than_the_excerpt_is_folded_whole() -> None:
    # The coarse search folds a 25 s excerpt; a shorter envelope must still work rather than
    # slicing itself away to nothing.
    bpm, _ = estimate_tempo(
        _pulses(seconds=12.0).astype(np.float64), _RATE, min_bpm=60.0, max_bpm=300.0
    )
    assert bpm == pytest.approx(_BPM)


def test_a_silent_envelope_has_no_tempo() -> None:
    silent = np.zeros(int(_SECONDS * _RATE), dtype=np.float32)
    assert estimate_tempo(silent.astype(np.float64), _RATE, min_bpm=60.0, max_bpm=300.0) == (
        0.0,
        0.0,
    )


def test_a_single_candidate_carries_no_confidence() -> None:
    # With one tempo on the grid there is no runner-up to divide by.
    _, confidence = estimate_tempo(
        _pulses().astype(np.float64), _RATE, min_bpm=120.0, max_bpm=120.0
    )
    assert confidence == pytest.approx(0.0, abs=1e-12)


def test_timing_puts_beat_zero_on_the_first_pulse() -> None:
    phase = 0.25
    envelope = _pulses(phase=phase)
    found = estimate_timing_from_envelopes(Envelopes(phase=envelope, tempo=envelope), _RATE)
    assert found['bpm'] == pytest.approx(_BPM)
    # Beat 0 sits at -OFFSET, and the detector reports onsets slightly late.
    assert -found['offset'] == pytest.approx(phase - ONSET_LATENCY_SECONDS, abs=0.02)


def test_a_supplied_tempo_is_what_the_phase_is_fitted_against() -> None:
    # Snapping rounds a tempo like this to 128.0, and a phase fitted at 128.0 describes a
    # different grid: over a long song the two drift apart and the offset lands half the drift
    # out. The supplied tempo has to reach the fit, not merely replace its answer afterwards.
    phase = 0.25
    envelope = _pulses(bpm=128.199, seconds=120.0, phase=phase)
    envelopes = Envelopes(phase=envelope, tempo=envelope)
    fitted = estimate_timing_from_envelopes(envelopes, _RATE, bpm=128.199)
    assert fitted['bpm'] == pytest.approx(128.199)
    assert -fitted['offset'] == pytest.approx(phase - ONSET_LATENCY_SECONDS, abs=0.02)
    # Left to itself the search snaps to a round number, and the offset moves with it.
    searched = estimate_timing_from_envelopes(envelopes, _RATE)
    assert searched['bpm'] == pytest.approx(128.0)
    assert searched['offset'] != pytest.approx(fitted['offset'], abs=0.02)


def test_a_supplied_tempo_reports_no_confidence() -> None:
    # Nothing was weighed against anything, so there is no ratio to report.
    envelope = _pulses()
    found = estimate_timing_from_envelopes(
        Envelopes(phase=envelope, tempo=envelope), _RATE, bpm=99.0
    )
    assert found['bpm'] == pytest.approx(99.0)
    assert found['confidence'] == pytest.approx(0.0, abs=1e-12)


def test_a_supplied_tempo_survives_an_envelope_carrying_nothing() -> None:
    # The caller's tempo is not the estimator's to discard, even when there is no phase to find.
    silence = np.zeros(4096, dtype=np.float32)
    found = estimate_timing_from_envelopes(Envelopes(phase=silence, tempo=silence), _RATE, bpm=99.0)
    assert found['bpm'] == pytest.approx(99.0)
    assert found['offset'] == pytest.approx(0.0, abs=1e-12)


def test_a_confidence_is_reported_alongside_the_timing() -> None:
    envelope = _pulses()
    assert (
        estimate_timing_from_envelopes(Envelopes(phase=envelope, tempo=envelope), _RATE)[
            'confidence'
        ]
        > 1.0
    )


@pytest.mark.parametrize(
    'envelope',
    [
        # Too few frames to fold at all.
        np.ones(8, dtype=np.float32),
        # Long enough, but carrying nothing.
        np.zeros(4096, dtype=np.float32),
    ],
)
def test_an_unusable_envelope_reports_no_timing(envelope: np.ndarray) -> None:
    found = estimate_timing_from_envelopes(Envelopes(phase=envelope, tempo=envelope), _RATE)
    assert found == {'bpm': 0.0, 'confidence': 0.0, 'offset': 0.0}


def test_a_tempo_outside_the_search_range_reports_nothing() -> None:
    # Squeezing the range to nothing leaves no candidate to score.
    found = estimate_timing_from_envelopes(
        Envelopes(phase=_pulses(), tempo=_pulses()), _RATE, min_bpm=300.0, max_bpm=60.0
    )
    assert found['bpm'] == pytest.approx(0.0, abs=1e-12)


def test_a_pulse_slower_than_the_excerpt_leaves_the_phase_alone() -> None:
    # One beat per fifty seconds cannot be phase-refined inside a shorter envelope, so the offset
    # stays where folding put it.
    envelope = _pulses(bpm=1.2, seconds=30.0)
    found = estimate_timing_from_envelopes(
        Envelopes(phase=envelope, tempo=_pulses()), _RATE, min_bpm=1.0, max_bpm=2.0
    )
    assert found['offset'] == pytest.approx(ONSET_LATENCY_SECONDS)


def test_a_flat_phase_envelope_still_yields_an_offset() -> None:
    # Every fold total identical makes the parabolic peak fit degenerate, which must fall back
    # rather than divide by zero.
    found = estimate_timing_from_envelopes(
        Envelopes(phase=np.ones(int(_SECONDS * _RATE), dtype=np.float32), tempo=_pulses()), _RATE
    )
    assert np.isfinite(found['offset'])


def test_timing_is_read_from_an_audio_file(tmp_path: Path) -> None:
    rate = TEMPO_PARAMS.sample_rate
    samples = np.zeros(int(rate * _SECONDS), dtype=np.float32)
    samples[:: int(rate * 60.0 / _BPM)] = 1.0
    path = tmp_path / 'clicks.wav'
    sf.write(path, samples, rate)
    assert estimate_timing(path)['bpm'] == pytest.approx(_BPM)
