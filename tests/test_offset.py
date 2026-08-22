"""Tests for recovering the downbeat phase."""

from __future__ import annotations

from smlab.offset import (
    BEATS_PER_MEASURE,
    PHASE_BINS,
    OffsetModel,
    band_envelopes,
    fold_profile,
    offset_for_phase,
    phase_of_offset,
)
import numpy as np
import pytest
import torch

_RATE = 172.265625
_BPM = 120.0
_BANDS = 4


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
    # Generation converts a predicted bin back into an offset, so a bin that
    # does not survive the trip would silently shift the whole chart.
    assert all(
        phase_of_offset(offset_for_phase(bin_, bpm), bpm) == bin_ for bin_ in range(PHASE_BINS)
    )


def test_folding_is_measured_from_the_given_start() -> None:
    # Training folds excerpts taken from partway into a song and relies on the
    # start argument to keep the label the same as it would be from zero.
    envelopes = _clicks(30)
    whole = fold_profile(envelopes, _RATE, _BPM)
    skip = int(_RATE * 8.0)
    part = fold_profile(envelopes[:, skip:], _RATE, _BPM, start=-skip / _RATE)
    assert int(np.argmax(whole[0])) == int(np.argmax(part[0]))


def test_each_band_is_scaled_on_its_own() -> None:
    # A quiet band carries the downbeat as often as a loud one, so absolute
    # level must not decide which band the model listens to.
    samples = np.random.default_rng(0).standard_normal(22050 * 5).astype(np.float32)
    envelopes = band_envelopes(samples)
    assert envelopes.shape[0] == _BANDS
    assert np.allclose(envelopes.max(axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize('shift', [1, 7, 24, 48, 95])
def test_the_model_is_shift_equivariant(shift: int) -> None:
    # The whole design rests on this: turning the bar cannot change which
    # position the model picks, only relabel it. Were it merely approximate the
    # model could learn that downbeats favour particular bins, which is an
    # artefact of how offsets happen to be authored rather than anything
    # audible.
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
