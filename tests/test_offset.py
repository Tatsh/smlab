"""Tests for recovering the downbeat phase."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]  # No stubs are published.
import torch

from smlab.offset import (
    BEATS_PER_MEASURE,
    KICK_CLARITY,
    KICK_LAG_SECONDS,
    PHASE_BINS,
    OffsetModel,
    band_envelopes,
    fold_profile,
    half_beat_rivalry,
    kick_phase,
    offset_for_phase,
    phase_of_offset,
    predict_phase,
    refine_offset,
)

if TYPE_CHECKING:
    from pathlib import Path

_RATE = 172.265625
_BPM = 120.0
_BANDS = 4
_SAMPLE_RATE = 22050


class _FixedModel(OffsetModel):
    """A phase model that is always sure the downbeat sits at one bar position."""

    def __init__(self, peak: int) -> None:
        super().__init__(channels=1)
        self.peak = peak % PHASE_BINS

    @override
    def forward(self, profile: torch.Tensor) -> torch.Tensor:
        logits = torch.full((profile.shape[0], PHASE_BINS), -10.0)
        logits[:, self.peak] = 10.0
        return logits


def _kick_drums(first: float, amplitude: float = 1.0, seconds: float = 30.0) -> np.ndarray:
    """Build a drum stem with a 60 Hz kick on every beat, over light noise."""
    rng = np.random.default_rng(0)
    drums = rng.standard_normal(int(_SAMPLE_RATE * seconds)).astype(np.float32) * 0.05
    t = np.arange(int(_SAMPLE_RATE * 0.05)) / _SAMPLE_RATE
    burst = np.sin(2 * np.pi * 60 * t) * np.exp(-t / 0.015)
    for start in np.arange(first, seconds - 0.06, 60.0 / _BPM):
        index = int(start * _SAMPLE_RATE)
        drums[index : index + burst.size] += amplitude * burst
    return drums


def _silence(directory: Path) -> Path:
    path = directory / 'song.wav'
    sf.write(path, np.zeros(_SAMPLE_RATE * 30, dtype='float32'), _SAMPLE_RATE)
    return path


def _clicks(phase: int, bpm: float = _BPM, seconds: float = 60.0) -> np.ndarray:
    """Build envelopes with one impulse per bar at a known bar position."""
    period = BEATS_PER_MEASURE * 60.0 / bpm
    frames = int(_RATE * seconds)
    envelopes = np.zeros((_BANDS, frames), dtype=np.float32)
    time = (phase + 0.5) / PHASE_BINS * period
    while time < seconds:
        envelopes[:, round(time * _RATE)] = 1.0
        time += period
    return envelopes


@pytest.mark.parametrize('phase', [0, 1, 24, 47, 48, 72, 95])
def test_a_pulse_folds_onto_its_own_position(phase: int) -> None:
    profile = fold_profile(_clicks(phase), _RATE, _BPM)
    assert int(np.argmax(profile[0])) == phase


@pytest.mark.parametrize('bpm', [90.0, 128.0, 155.0, 210.0])
def test_offset_and_phase_are_inverse(bpm: float) -> None:
    # Generation converts a predicted bin back into an offset, so a bin that does not survive the
    # trip would silently shift the whole chart.
    assert all(
        phase_of_offset(offset_for_phase(bin_, bpm), bpm) == bin_ for bin_ in range(PHASE_BINS)
    )


def test_folding_is_measured_from_the_given_start() -> None:
    # Training folds excerpts taken from partway into a song and relies on the start argument to
    # keep the label the same as it would be from zero.
    envelopes = _clicks(30)
    whole = fold_profile(envelopes, _RATE, _BPM)
    skip = int(_RATE * 8.0)
    part = fold_profile(envelopes[:, skip:], _RATE, _BPM, start=-skip / _RATE)
    assert int(np.argmax(whole[0])) == int(np.argmax(part[0]))


def test_each_band_is_scaled_on_its_own() -> None:
    # A quiet band carries the downbeat as often as a loud one, so absolute level must not decide
    # which band the model listens to.
    samples = np.random.default_rng(0).standard_normal(22050 * 5).astype(np.float32)
    envelopes = band_envelopes(samples)
    assert envelopes.shape[0] == _BANDS
    assert np.allclose(envelopes.max(axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize('shift', [1, 7, 24, 48, 95])
def test_the_model_is_shift_equivariant(shift: int) -> None:
    # The whole design rests on this: turning the bar cannot change which position the model picks,
    # only relabel it. Were it merely approximate the model could learn that downbeats favour
    # particular bins, which is an artefact of how offsets happen to be authored rather than
    # anything audible.
    torch.manual_seed(0)
    model = OffsetModel().eval()
    profile = torch.randn(1, _BANDS, PHASE_BINS)
    with torch.no_grad():
        plain = model(profile)[0]
        turned = model(torch.roll(profile, shift, dims=2))[0]
    assert torch.allclose(turned, torch.roll(plain, shift, dims=0), atol=1e-5)


def test_the_model_scores_every_position() -> None:
    model = OffsetModel().eval()
    with torch.no_grad():
        scored = model(torch.randn(3, _BANDS, PHASE_BINS))
    assert scored.shape == (3, PHASE_BINS)
    assert torch.isfinite(scored).all()


def test_a_silent_song_folds_without_dividing_by_zero() -> None:
    profile = fold_profile(np.zeros((_BANDS, 1000), dtype=np.float32), _RATE, _BPM)
    assert profile.shape == (_BANDS, PHASE_BINS)
    assert np.isfinite(profile).all()


def test_the_phase_is_averaged_over_several_excerpts() -> None:
    # One excerpt can be misled by a fill or a break, so the distribution is averaged over as many
    # as the song affords.
    envelopes = _clicks(0, seconds=90.0)
    estimate = predict_phase(OffsetModel().eval(), envelopes, _RATE, _BPM)
    assert 0 <= estimate.phase < PHASE_BINS
    assert 0.0 < estimate.weight <= 1.0
    assert estimate.rivalry >= 0.0


def test_a_song_shorter_than_one_excerpt_still_predicts() -> None:
    # The excerpt stride would otherwise leave no starting points at all.
    estimate = predict_phase(OffsetModel().eval(), _clicks(0, seconds=5.0), _RATE, _BPM)
    assert 0 <= estimate.phase < PHASE_BINS


def test_an_offset_is_recovered_from_a_file(tmp_path: Path) -> None:
    estimate = refine_offset(_FixedModel(24).eval(), _silence(tmp_path), _BPM)
    # A quarter of the way through the bar at 120 BPM is one beat in.
    assert estimate.offset == pytest.approx(offset_for_phase(24, _BPM))
    assert estimate.source == 'model'
    assert estimate.clarity == pytest.approx(0.0, abs=1e-12)


def test_the_kick_is_found_wherever_it_lands() -> None:
    early = kick_phase(_kick_drums(0.1), _BPM)
    late = kick_phase(_kick_drums(0.3), _BPM)
    assert early.clarity > KICK_CLARITY
    assert late.seconds - early.seconds == pytest.approx(0.2, abs=0.005)
    assert early.seconds == pytest.approx(0.1 - KICK_LAG_SECONDS, abs=0.02)


def test_a_clear_kick_overrules_a_model_half_a_beat_out(tmp_path: Path) -> None:
    drums = _kick_drums(0.1)
    kick = kick_phase(drums, _BPM)
    period = 60.0 / _BPM
    kick_bin = int(kick.seconds / period * 24)
    found = refine_offset(_FixedModel(kick_bin + 12).eval(), _silence(tmp_path), _BPM, drums)
    assert found.source == 'kick'
    assert (-found.offset - kick.seconds) / period == pytest.approx(
        round((-found.offset - kick.seconds) / period)
    )
    assert found.rivalry == pytest.approx(0.0, abs=1e-12)


def test_the_model_picks_which_beat_begins_the_bar(tmp_path: Path) -> None:
    drums = _kick_drums(0.1)
    kick = kick_phase(drums, _BPM)
    period = 60.0 / _BPM
    kick_bin = int(kick.seconds / period * 24)
    # Five bins off the kick is outside what counts as agreeing, but still within the third beat.
    found = refine_offset(_FixedModel(kick_bin + 48 + 5).eval(), _silence(tmp_path), _BPM, drums)
    assert found.offset == pytest.approx(-(kick.seconds + 2 * period))


@pytest.mark.parametrize(('half_beat_away', 'source'), [(True, 'model'), (False, 'kick')])
def test_a_weak_kick_yields_to_a_model_that_disagrees(
    tmp_path: Path, *, half_beat_away: bool, source: str
) -> None:
    drums = _kick_drums(0.1, amplitude=0.05)
    kick = kick_phase(drums, _BPM)
    assert 2.0 < kick.clarity < KICK_CLARITY
    kick_bin = int(kick.seconds / (60.0 / _BPM) * 24)
    peak = kick_bin + (12 if half_beat_away else 0)
    found = refine_offset(_FixedModel(peak).eval(), _silence(tmp_path), _BPM, drums)
    assert found.source == source
    assert found.clarity == pytest.approx(kick.clarity)


def test_silent_drums_leave_the_model_in_charge(tmp_path: Path) -> None:
    silent = np.zeros(int(_SAMPLE_RATE * 30), dtype=np.float32)
    found = refine_offset(_FixedModel(30).eval(), _silence(tmp_path), _BPM, silent)
    assert found.source == 'model'
    assert found.offset == pytest.approx(offset_for_phase(30, _BPM))


@pytest.mark.parametrize(
    ('peaks', 'expected'),
    [
        ({30: 1.0}, 0.0),
        ({30: 0.5, 42: 0.5}, 1.0),
        ({30: 0.4, 54: 0.4, 66: 0.2}, 0.25),
        ({95: 0.8, 11: 0.2}, 0.25),
    ],
)
def test_half_beat_rivalry_pools_across_beats(peaks: dict[int, float], expected: float) -> None:
    weights = np.zeros(PHASE_BINS)
    for position, weight in peaks.items():
        weights[position] = weight
    assert half_beat_rivalry(weights, max(peaks, key=peaks.__getitem__)) == pytest.approx(expected)
