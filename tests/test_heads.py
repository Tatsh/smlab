"""Tests for the placement and selection heads."""

from __future__ import annotations

from smlab.encoder import MEASURE_SLOTS, EncoderConfig
from smlab.features import TOTAL_CHANNELS
from smlab.heads import (
    MAX_DELTA,
    ChartModel,
    PlacementHead,
    SelectionBatch,
    SelectionHead,
    metric_prior_logits,
)
import numpy as np
import pytest
import torch

_WIDTH = 24
_VOCABULARY = 7
_SMALL = EncoderConfig(
    channels=32, model_dimension=_WIDTH, local_blocks=1, slot_layers=1, measure_layers=1, heads=2
)


def _batch(steps: int, slots: int) -> SelectionBatch:
    return SelectionBatch(
        delta=torch.full((1, steps), 6),
        position=torch.zeros(1, steps, dtype=torch.long),
        previous=torch.zeros(1, steps, dtype=torch.long),
        slots=torch.arange(steps).unsqueeze(0) % slots,
    )


def test_a_position_never_stepped_on_scores_below_one_always_stepped_on() -> None:
    counts = np.array([100.0, 0.0], dtype=np.float64)
    totals = np.array([100.0, 100.0], dtype=np.float64)
    logits = metric_prior_logits(counts, totals)
    assert logits[0] > 0.0 > logits[1]


def test_the_prior_is_clipped_away_from_certainty() -> None:
    # Log-odds of a rate of exactly zero or one are infinite, which would
    # poison every score downstream.
    counts = np.array([0.0, 50.0], dtype=np.float64)
    totals = np.array([50.0, 50.0], dtype=np.float64)
    assert np.all(np.isfinite(metric_prior_logits(counts, totals)))


def test_a_position_never_observed_does_not_divide_by_zero() -> None:
    logits = metric_prior_logits(np.zeros(4), np.zeros(4))
    assert np.all(np.isfinite(logits))


def test_placement_scores_one_slot_at_a_time() -> None:
    head = PlacementHead(_WIDTH).eval()
    encoded = torch.randn(2, MEASURE_SLOTS, _WIDTH)
    position = torch.arange(MEASURE_SLOTS).unsqueeze(0).expand(2, -1)
    assert head(encoded, position).shape == (2, MEASURE_SLOTS)


def test_placement_adds_the_prior_it_was_given() -> None:
    prior = np.linspace(-2.0, 2.0, MEASURE_SLOTS, dtype=np.float32)
    head = PlacementHead(_WIDTH, prior).eval()
    encoded = torch.zeros(1, MEASURE_SLOTS, _WIDTH)
    position = torch.arange(MEASURE_SLOTS).unsqueeze(0)
    with torch.no_grad():
        full = head(encoded, position)
        none = head(encoded, position, weight=0.0)
    assert torch.allclose(full - none, torch.from_numpy(prior), atol=1e-5)


def test_the_prior_can_be_damped_for_ranking() -> None:
    # The prior is right about probability and wrong for ranking, so decoding
    # keeps only a quarter of it.
    prior = np.full(MEASURE_SLOTS, 3.0, dtype=np.float32)
    head = PlacementHead(_WIDTH, prior).eval()
    encoded = torch.zeros(1, MEASURE_SLOTS, _WIDTH)
    position = torch.zeros(1, MEASURE_SLOTS, dtype=torch.long)
    with torch.no_grad():
        damped = head(encoded, position, weight=0.25)
        full = head(encoded, position)
    assert float(full[0, 0] - damped[0, 0]) == pytest.approx(2.25, abs=1e-4)


def test_selection_scores_every_pattern_for_every_step() -> None:
    head = SelectionHead(_WIDTH, _VOCABULARY, layers=1, heads=2).eval()
    encoded = torch.randn(1, MEASURE_SLOTS, _WIDTH)
    with torch.no_grad():
        logits = head(encoded, _batch(12, MEASURE_SLOTS))
    assert logits.shape == (1, 12, _VOCABULARY)


def test_a_gap_beyond_the_embedding_is_clamped() -> None:
    # Long rests happen, and without the clamp the lookup runs off the end of
    # the embedding table.
    head = SelectionHead(_WIDTH, _VOCABULARY, layers=1, heads=2).eval()
    encoded = torch.randn(1, MEASURE_SLOTS, _WIDTH)
    batch = _batch(4, MEASURE_SLOTS)._replace(delta=torch.full((1, 4), MAX_DELTA * 10))
    with torch.no_grad():
        assert torch.isfinite(head(encoded, batch)).all()


def test_selection_cannot_read_ahead_of_itself() -> None:
    # Decoding walks the steps in order, so a step's score must not depend on
    # anything after it.
    head = SelectionHead(_WIDTH, _VOCABULARY, layers=1, heads=2).eval()
    encoded = torch.randn(1, MEASURE_SLOTS, _WIDTH)
    batch = _batch(8, MEASURE_SLOTS)
    later = batch._replace(previous=torch.tensor([[0, 0, 0, 0, 5, 5, 5, 5]]))
    with torch.no_grad():
        first = head(encoded, batch)
        second = head(encoded, later)
    assert torch.allclose(first[0, :4], second[0, :4], atol=1e-5)


def test_the_model_carries_both_heads_over_one_encoder() -> None:
    model = ChartModel(_VOCABULARY, _SMALL).eval()
    features = torch.randn(1, MEASURE_SLOTS * 2 * 2, TOTAL_CHANNELS)
    zeros = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        encoded = model.encode(
            features, zeros, torch.full((1,), 9), zeros, zeros, torch.full((1,), 4)
        )
        placed = model.placement(encoded, torch.zeros(1, encoded.shape[1], dtype=torch.long))
        chosen = model.selection(encoded, _batch(6, encoded.shape[1]))
    assert encoded.shape == (1, MEASURE_SLOTS * 2, _WIDTH)
    assert placed.shape == (1, MEASURE_SLOTS * 2)
    assert chosen.shape == (1, 6, _VOCABULARY)


def test_a_model_can_be_built_with_the_default_encoder_settings() -> None:
    assert ChartModel(_VOCABULARY).placement.prior.shape == (MEASURE_SLOTS,)
