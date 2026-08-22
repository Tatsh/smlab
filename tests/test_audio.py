"""Tests for audio loading and onset envelopes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]  # No stubs are published.

from smlab.audio import (
    DEFAULT_HOP_LENGTH,
    DEFAULT_SAMPLE_RATE,
    OnsetParams,
    audio_duration,
    envelope_rate,
    load_audio,
    onset_envelope,
)

if TYPE_CHECKING:
    from pathlib import Path

_SECONDS = 2.0


def _clicks(rate: int = DEFAULT_SAMPLE_RATE, every: float = 0.5) -> np.ndarray:
    samples = np.zeros(int(rate * _SECONDS), dtype=np.float32)
    samples[:: int(rate * every)] = 1.0
    return samples


def test_the_frame_rate_follows_the_hop_length() -> None:
    assert envelope_rate(22050, 128) == pytest.approx(172.265625)
    assert envelope_rate() == pytest.approx(DEFAULT_SAMPLE_RATE / DEFAULT_HOP_LENGTH)


def test_settings_report_their_own_frame_rate() -> None:
    assert OnsetParams(hop_length=256).frame_rate == pytest.approx(DEFAULT_SAMPLE_RATE / 256)


def test_audio_is_read_as_mono_floats(tmp_path: Path) -> None:
    path = tmp_path / 'stereo.wav'
    sf.write(path, np.zeros((DEFAULT_SAMPLE_RATE, 2), dtype='float32'), DEFAULT_SAMPLE_RATE)
    samples = load_audio(path)
    assert samples.ndim == 1
    assert samples.dtype == np.float32


def test_audio_is_resampled_to_the_rate_asked_for(tmp_path: Path) -> None:
    path = tmp_path / 'tone.wav'
    sf.write(path, np.zeros(44100, dtype='float32'), 44100)
    assert len(load_audio(path, sample_rate=11025)) == pytest.approx(11025, abs=64)


def test_a_duration_counts_the_samples_that_decode(tmp_path: Path) -> None:
    path = tmp_path / 'tone.wav'
    sf.write(path, np.zeros(int(44100 * 1.5), dtype='float32'), 44100)
    assert audio_duration(path) == pytest.approx(1.5, abs=0.01)


def test_an_envelope_peaks_where_the_clicks_are() -> None:
    envelope = onset_envelope(_clicks())
    rate = OnsetParams().frame_rate
    peaks = np.flatnonzero(envelope > envelope.max() / 2) / rate
    # Clicks land every half second, so consecutive peaks are half a second
    # apart give or take a frame.
    assert peaks.size >= 2
    assert float(np.diff(peaks).max()) == pytest.approx(0.5, abs=0.05)


def test_the_default_settings_are_used_when_none_are_given() -> None:
    assert np.array_equal(onset_envelope(_clicks()), onset_envelope(_clicks(), OnsetParams()))


def test_a_band_limit_is_honoured() -> None:
    # A limited band sees a different amount of the click's energy, so the two
    # envelopes must not be the same array.
    full = onset_envelope(_clicks(), OnsetParams())
    limited = onset_envelope(_clicks(), OnsetParams(fmax=4000.0))
    assert not np.array_equal(full, limited)


def test_silence_produces_a_flat_envelope() -> None:
    peak = float(onset_envelope(np.zeros(DEFAULT_SAMPLE_RATE, dtype=np.float32)).max())
    assert peak == pytest.approx(0.0, abs=1e-12)
