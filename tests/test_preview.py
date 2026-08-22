"""Tests for choosing a preview start."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from smlab.dataset import FEATURE_DIMENSION
from smlab.preview import (
    DEFAULT_SAMPLE_LENGTH,
    EARLIEST_FRACTION,
    LATEST_FRACTION,
    POOLED_DIMENSION,
    SLOTS_PER_MEASURE,
    PreviewModel,
    measure_features,
    predict_sample_start,
)
from smlab.timing import BEATS_PER_MEASURE, TimingData


def _features(measures: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((measures * SLOTS_PER_MEASURE, FEATURE_DIMENSION)).astype(np.float16)


@pytest.mark.parametrize('measures', [4, 16, 120])
def test_pooling_yields_one_row_per_measure(measures: int) -> None:
    pooled = measure_features(_features(measures))
    assert pooled.shape == (measures, POOLED_DIMENSION)


def test_pooling_appends_relative_position() -> None:
    pooled = measure_features(_features(32))
    assert pooled[0, -1] == pytest.approx(0.0)
    assert pooled[-1, -1] == pytest.approx(1.0)
    assert np.all(np.diff(pooled[:, -1]) > 0)


def test_short_song_yields_no_measures() -> None:
    assert measure_features(np.zeros((8, FEATURE_DIMENSION), dtype=np.float16)).shape[0] == 0


def test_padding_is_excluded_from_the_softmax() -> None:
    model = PreviewModel()
    features = torch.zeros(1, 10, POOLED_DIMENSION)
    mask = torch.zeros(1, 10, dtype=torch.bool)
    mask[0, :4] = True
    logits = model(features, mask)
    assert torch.isinf(logits[0, 4:]).all()
    assert torch.isfinite(logits[0, :4]).all()


def test_prediction_lands_on_a_measure_boundary() -> None:
    timing = TimingData.constant(150.0, -0.048)
    start = predict_sample_start(PreviewModel(), _features(96), timing)
    beat = timing.beat_at_time(start)
    assert beat / BEATS_PER_MEASURE == pytest.approx(round(beat / BEATS_PER_MEASURE), abs=1e-6)


def test_prediction_stays_within_the_allowed_span() -> None:
    timing = TimingData.constant(150.0, 0.0)
    measures = 200
    start = predict_sample_start(PreviewModel(), _features(measures), timing)
    duration = timing.time_at_beat(measures * BEATS_PER_MEASURE)
    assert EARLIEST_FRACTION * duration - 2.0 <= start <= LATEST_FRACTION * duration + 2.0


def test_too_short_a_song_starts_at_zero() -> None:
    timing = TimingData.constant(150.0, 0.0)
    assert predict_sample_start(PreviewModel(), _features(2), timing) == pytest.approx(0.0)


def test_default_length_matches_the_corpus_convention() -> None:
    assert pytest.approx(15.0) == DEFAULT_SAMPLE_LENGTH
