"""Tests for the measures training reports."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast
import json
import math

import numpy as np
import pytest
import torch

from smlab.cache import cache_path_for
from smlab.chart.data import WINDOW_MEASURES, ChartExample
from smlab.encoder import MEASURE_SLOTS, EncoderConfig
from smlab.features import TOTAL_CHANNELS
from smlab.train.charts import (
    ChartTrainingConfig,
    stratified_auc,
    style_sampler,
    train_chart_model,
)
from smlab.vocab import Vocabulary, encode_row

if TYPE_CHECKING:
    from smlab.typing import SongRecord

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


def _corpus(tmp_path: Path, rows: int = 64) -> tuple[Path, list[SongRecord], Vocabulary]:
    """Two songs, one either side of the training split, with tiny cache entries."""
    records = [
        cast(
            'SongRecord',
            {
                'charts': [{'difficulty': 'Challenge', 'meter': 16, 'rows': 64, 'style': 'feet'}],
                'pack': 'Modern',
                'primary_bpm': 150.0,
                'simfile': name,
            },
        )
        for name in ('/corpus/0.sm', '/corpus/18.sm')
    ]
    for record in records:
        path = cache_path_for(tmp_path, record['simfile'])
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            features=np.zeros(
                (2 * WINDOW_MEASURES * MEASURE_SLOTS, TOTAL_CHANNELS), dtype=np.float16
            ),
            slots_0=np.arange(rows, dtype=np.int32) * 12,
            panels_0=np.tile(np.array([[1, 0, 0, 0]], dtype=np.uint8), (rows, 1)),
            meta=np.asarray(json.dumps([{'difficulty': 'Challenge', 'index': 0, 'meter': 16}])),
        )
    return tmp_path, records, Vocabulary([encode_row((1, 0, 0, 0)), encode_row((0, 1, 0, 0))])


def test_training_reports_what_it_measured_and_saves_the_best(tmp_path: Path) -> None:
    cache_root, records, vocabulary = _corpus(tmp_path)
    output = tmp_path / 'checkpoints' / 'chart.pt'
    measured = train_chart_model(
        cache_root,
        records,
        vocabulary,
        output,
        ChartTrainingConfig(batch_size=1, epochs=1, warmup=1, workers=0),
        EncoderConfig(
            channels=16,
            model_dimension=24,
            local_blocks=1,
            slot_layers=1,
            measure_layers=1,
            heads=2,
        ),
    )
    assert set(measured) == {'eighth_auc', 'pattern', 'placement', 'quarter_auc'}
    assert output.is_file()
    saved = torch.load(output, weights_only=False)
    assert saved['vocabulary'] == len(vocabulary)
    assert saved['prior'].shape == (MEASURE_SLOTS,)


def test_the_metric_prior_is_computed_once_and_reused(tmp_path: Path) -> None:
    # Counting steps across the corpus is slow, and it does not change between
    # runs over the same cache.
    cache_root, records, vocabulary = _corpus(tmp_path / 'cache')
    settings = ChartTrainingConfig(batch_size=1, epochs=1, warmup=1, workers=0)
    shape = EncoderConfig(
        channels=16, model_dimension=24, local_blocks=1, slot_layers=1, measure_layers=1, heads=2
    )
    output = tmp_path / 'chart.pt'
    train_chart_model(cache_root, records, vocabulary, output, settings, shape)
    cached = tmp_path / 'metric_prior.npy'
    assert cached.is_file()
    stamp = cached.stat().st_mtime_ns
    train_chart_model(cache_root, records, vocabulary, output, settings, shape)
    assert cached.stat().st_mtime_ns == stamp


def test_training_falls_back_to_default_settings(tmp_path: Path) -> None:
    # Both configuration objects are optional, and the defaults have to at
    # least construct a model.
    cache_root, records, vocabulary = _corpus(tmp_path)
    measured = train_chart_model(
        cache_root,
        records,
        vocabulary,
        tmp_path / 'chart.pt',
        ChartTrainingConfig(batch_size=1, epochs=0, workers=0),
    )
    assert measured == {'quarter_auc': 0.0}


def test_a_split_that_says_nothing_never_becomes_the_best(tmp_path: Path) -> None:
    # Every quarter of the window carries a step, so no stratum holds both
    # classes and the area under the curve is undefined. An undefined score
    # must not overwrite the checkpoint.
    cache_root, records, vocabulary = _corpus(tmp_path, rows=WINDOW_MEASURES * 4)
    output = tmp_path / 'chart.pt'
    measured = train_chart_model(
        cache_root,
        records,
        vocabulary,
        output,
        ChartTrainingConfig(batch_size=1, epochs=1, warmup=1, workers=0),
        EncoderConfig(
            channels=16,
            model_dimension=24,
            local_blocks=1,
            slot_layers=1,
            measure_layers=1,
            heads=2,
        ),
    )
    assert measured == {'quarter_auc': 0.0}
    assert not output.exists()
