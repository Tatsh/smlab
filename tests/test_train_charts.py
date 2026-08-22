"""Tests for the measures training reports."""

from __future__ import annotations

from pathlib import Path
import math

from smlab.chart_data import ChartExample
from smlab.train_charts import stratified_auc, style_sampler
import numpy as np
import pytest

_QUARTER = 12
_SLOTS = 96 * 8
_QUARTERS = _SLOTS // _QUARTER
_SCALE = np.float32(10.0)
_UNUSED = Path('unused.npz')
"""The sampler never opens a cache entry, so any path will do."""


def _positions() -> np.ndarray:
    return np.arange(_SLOTS, dtype=np.int64) % 96


def _half_the_quarters() -> np.ndarray:
    """Label half the quarter slots, so both classes are represented."""
    labels = np.zeros(_SLOTS, dtype=np.float32)
    labels[:: _QUARTER * 2] = 1.0
    return labels


def test_perfect_ranking_scores_one() -> None:
    positions = _positions()
    labels = _half_the_quarters()
    # Every quarter that carries a step is scored above every quarter that
    # does not, which is what an area under the curve of one means.
    assert stratified_auc(labels * _SCALE, labels, positions, _QUARTER) == pytest.approx(1.0)


def test_reversed_ranking_scores_zero() -> None:
    positions = _positions()
    labels = _half_the_quarters()
    score = stratified_auc(-labels * _SCALE, labels, positions, _QUARTER)
    assert score == pytest.approx(0.0, abs=1e-12)


def test_a_stratum_with_one_class_is_not_scored() -> None:
    # A window where every quarter carries a step, or none does, says nothing
    # about discrimination, and averaging a made-up number in would flatter it.
    positions = _positions()
    labels = np.ones(_SLOTS, dtype=np.float32)
    assert math.isnan(
        stratified_auc(
            np.random.default_rng(0).random(_SLOTS).astype(np.float32), labels, positions, _QUARTER
        )
    )


def test_only_the_named_stratum_counts() -> None:
    # The point of stratifying is to remove what the metric prior already
    # knows, so slots off the stratum must not reach the score however they
    # are ranked.
    positions = _positions()
    labels = _half_the_quarters()
    spoiled = (labels * _SCALE).copy()
    off = positions % _QUARTER != 0
    spoiled[off] = 1000.0
    labels_off = labels.copy()
    labels_off[off] = 1.0
    assert stratified_auc(spoiled, labels_off, positions, _QUARTER) == pytest.approx(1.0)


def test_a_rare_style_is_drawn_as_often_as_a_common_one() -> None:
    # The corpus is overwhelmingly feet-style, so uniform sampling would leave
    # the keyboard conditioning essentially unlearnt.
    examples = [
        ChartExample(bpm=128.0, difficulty=0, meter=5, path=_UNUSED, scale=0, style=style)
        for style in [0] * 95 + [2] * 5
    ]
    sampler = style_sampler(examples)
    weights = np.asarray(sampler.weights, dtype=np.float64)
    common = weights[:95].sum()
    rare = weights[95:].sum()
    assert math.isclose(common, rare, rel_tol=1e-6)
