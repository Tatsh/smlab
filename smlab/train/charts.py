"""
Training the shared encoder with both heads.

The headline metric from the first model is deliberately not reported here. An
area under the curve computed across all slots is dominated by the fact that
notes fall on beats, which a forty-eight entry lookup table already knows.
What is reported instead is the same measure computed *within* one metric
position class, where position carries no information and any discrimination
must come from the audio.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING
import logging
import math

from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import torch

from smlab.chart.data import ChartExample, ChartWindows, measure_prior
from smlab.encoder import EncoderConfig
from smlab.heads import ChartModel, SelectionBatch

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from smlab.typing import SongRecord
    from smlab.vocab import Vocabulary

__all__ = ('ChartTrainingConfig', 'stratified_auc', 'style_sampler', 'train_chart_model')

log = logging.getLogger(__name__)

_BEAT_SLOTS = 12
_EIGHTH_SLOTS = 6
_MIN_STRATUM = 8


@dataclass(frozen=True, slots=True)
class ChartTrainingConfig:
    """Hyperparameters for the chart model."""

    batch_size: int = 8
    """Windows per optimiser step."""
    epochs: int = 40
    """Passes over the chart list."""
    learning_rate: float = 3e-4
    """Peak learning rate."""
    limit: int = 0
    """Cap on charts loaded, for quick runs. Zero loads everything."""
    selection_weight: float = 0.5
    """How much the pattern loss counts against the placement loss."""
    warmup: int = 500
    """Optimiser steps spent ramping the learning rate up."""
    workers: int = 6
    """Loader worker processes."""


def stratified_auc(
    scores: NDArray[np.float32],
    labels: NDArray[np.float32],
    positions: NDArray[np.int64],
    stride: int,
) -> float:
    """
    Score discrimination within one metric position class.

    Restricting to slots that share a position removes the metric prior's
    contribution, so what remains reflects reading the audio.

    Parameters
    ----------
    scores : :py:class:`~numpy.ndarray`
        Predicted logits per slot.
    labels : :py:class:`~numpy.ndarray`
        One where a step occurs.
    positions : :py:class:`~numpy.ndarray`
        Position within the bar per slot.
    stride : int
        Slots between members of the stratum, twelve for quarter notes.

    Returns
    -------
    float
        Area under the curve, or not-a-number when a class is missing.
    """
    keep = positions % stride == 0
    chosen, marks = scores[keep], labels[keep]
    positive = float(marks.sum())
    negative = float((marks == 0).sum())
    if positive < _MIN_STRATUM or negative < _MIN_STRATUM:
        return float('nan')
    order = np.argsort(chosen)
    ranks = np.empty(len(chosen), dtype=np.float64)
    ranks[order] = np.arange(1, len(chosen) + 1)
    return float((ranks[marks == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def style_sampler(examples: list[ChartExample]) -> WeightedRandomSampler:
    """
    Draw performance styles evenly rather than as the corpus happens to hold them.

    The corpus is 95% charts danceable with two feet, so uniform sampling would
    give keyboard and hand-chord charts about three per cent of the gradient
    updates and their conditioning would go essentially unlearnt.

    Parameters
    ----------
    examples : list[ChartExample]
        Charts the sampler draws from.

    Returns
    -------
    :py:class:`~torch.utils.data.WeightedRandomSampler`
        A sampler weighting each chart by the inverse frequency of its style.
    """
    counts = Counter(example.style for example in examples)
    weights = [1.0 / counts[example.style] for example in examples]
    return WeightedRandomSampler(weights, num_samples=len(examples), replacement=True)


def _selection_batch(batch: dict[str, torch.Tensor], device: torch.device) -> SelectionBatch:
    """
    Move one batch's selection inputs onto the compute device.

    Parameters
    ----------
    batch : dict[str, :py:class:`~torch.Tensor`]
        A batch from :py:class:`~smlab.chart_data.ChartWindows`.
    device : :py:class:`~torch.device`
        Compute device.

    Returns
    -------
    SelectionBatch
        The batch as head inputs.
    """
    return SelectionBatch(
        delta=batch['delta'].to(device),
        position=batch['step_position'].to(device),
        previous=batch['previous'].to(device),
        slots=batch['step_slots'].to(device),
    )


def _evaluate(
    model: ChartModel, loader: DataLoader[dict[str, torch.Tensor]], device: torch.device
) -> dict[str, float]:
    """
    Measure held-out loss and stratified discrimination.

    Parameters
    ----------
    model : ChartModel
        The model under evaluation.
    loader : :py:class:`~torch.utils.data.DataLoader`
        Validation batches.
    device : :py:class:`~torch.device`
        Compute device.

    Returns
    -------
    dict[str, float]
        Placement loss, pattern loss, and stratified areas under the curve.
    """
    model.eval()
    placement_loss = nn.BCEWithLogitsLoss()
    pattern_loss = nn.CrossEntropyLoss(ignore_index=-100)
    totals = {'placement': 0.0, 'pattern': 0.0}
    quarters: list[float] = []
    eighths: list[float] = []
    seen = 0
    with (
        torch.no_grad(),
        torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'),
    ):
        for batch in loader:
            encoded = model.encode(
                batch['features'].to(device),
                batch['difficulty'].to(device),
                batch['meter'].to(device),
                batch['scale'].to(device),
                batch['style'].to(device),
                batch['rate'].to(device),
            )
            position = batch['position'].to(device)
            logits = model.placement(encoded, position)
            target = batch['placement'].to(device)
            totals['placement'] += float(placement_loss(logits.float(), target).item())
            patterns = model.selection(encoded, _selection_batch(batch, device))
            totals['pattern'] += float(
                pattern_loss(
                    patterns.float().reshape(-1, patterns.shape[-1]),
                    batch['pattern_target'].to(device).reshape(-1),
                ).item()
            )
            scores = logits.float().cpu().numpy()
            marks = batch['placement'].numpy()
            spots = batch['position'].numpy()
            for index in range(scores.shape[0]):
                quarters.append(
                    stratified_auc(scores[index], marks[index], spots[index], _BEAT_SLOTS)
                )
                eighths.append(
                    stratified_auc(scores[index], marks[index], spots[index], _EIGHTH_SLOTS)
                )
            seen += 1
    return {
        'eighth_auc': float(np.nanmedian(eighths)) if eighths else float('nan'),
        'pattern': totals['pattern'] / max(seen, 1),
        'placement': totals['placement'] / max(seen, 1),
        'quarter_auc': float(np.nanmedian(quarters)) if quarters else float('nan'),
    }


def train_chart_model(  # noqa: PLR0914
    cache_root: Path,
    records: list[SongRecord],
    vocabulary: Vocabulary,
    output: Path,
    config: ChartTrainingConfig | None = None,
    encoder: EncoderConfig | None = None,
) -> dict[str, float]:
    """
    Train the shared encoder and both heads.

    Parameters
    ----------
    cache_root : :py:class:`~pathlib.Path`
        Stem feature cache directory.
    records : list[SongRecord]
        Manifest records covering the corpus.
    vocabulary : Vocabulary
        Pattern vocabulary the selection head predicts over.
    output : :py:class:`~pathlib.Path`
        Where to write the checkpoint.
    config : ChartTrainingConfig | None
        Hyperparameters, or ``None`` for the defaults.
    encoder : EncoderConfig | None
        Encoder shape, or ``None`` for the defaults.

    Returns
    -------
    dict[str, float]
        The best validation measurements reached.
    """
    settings = config if config is not None else ChartTrainingConfig()
    shape = encoder if encoder is not None else EncoderConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_set = ChartWindows(
        cache_root, records, vocabulary, validation=False, limit=settings.limit
    )
    valid_set = ChartWindows(cache_root, records, vocabulary, validation=True, limit=settings.limit)
    cached_prior = cache_root.parent / 'metric_prior.npy'
    if cached_prior.is_file():
        prior = np.load(cached_prior)
        log.info('Loaded the metric prior from `%s`.', cached_prior)
    else:
        log.info('Computing the metric prior from %d training charts.', len(train_set))
        prior = measure_prior(train_set.examples)
        np.save(cached_prior, prior)
    model = ChartModel(len(vocabulary), shape, prior).to(device)
    log.info('Model has %.1f M parameters.', sum(p.numel() for p in model.parameters()) / 1e6)
    train_loader = DataLoader(
        train_set,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=settings.workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=settings.workers > 0,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=settings.batch_size,
        num_workers=settings.workers,
        persistent_workers=settings.workers > 0,
    )
    placement_loss = nn.BCEWithLogitsLoss()
    pattern_loss = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=0.01)
    steps = max(len(train_loader) * settings.epochs, 1)

    def rate(step: int) -> float:
        if step < settings.warmup:
            return step / max(settings.warmup, 1)
        progress = (step - settings.warmup) / max(steps - settings.warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    best: dict[str, float] = {'quarter_auc': 0.0}
    for epoch in range(settings.epochs):
        model.train()
        running = 0.0
        seen = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                encoded = model.encode(
                    batch['features'].to(device),
                    batch['difficulty'].to(device),
                    batch['meter'].to(device),
                    batch['scale'].to(device),
                    batch['style'].to(device),
                    batch['rate'].to(device),
                )
                logits = model.placement(encoded, batch['position'].to(device))
                patterns = model.selection(encoded, _selection_batch(batch, device))
                loss = placement_loss(logits.float(), batch['placement'].to(device))
                loss += settings.selection_weight * pattern_loss(
                    patterns.float().reshape(-1, patterns.shape[-1]),
                    batch['pattern_target'].to(device).reshape(-1),
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            running += float(loss.item())
            seen += 1
        measured = _evaluate(model, valid_loader, device)
        log.info(
            'Epoch %d: train %.4f, placement %.4f, pattern %.4f, quarter AUC %.4f, '
            'eighth AUC %.4f.',
            epoch + 1,
            running / max(seen, 1),
            measured['placement'],
            measured['pattern'],
            measured['quarter_auc'],
            measured['eighth_auc'],
        )
        if measured['quarter_auc'] > best['quarter_auc']:
            best = measured
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {'model': model.state_dict(), 'prior': prior, 'vocabulary': len(vocabulary)}, output
            )
    return best
