"""
Placement and selection heads reading the shared audio encoder.

The placement head emits a *correction* to a fixed metric prior rather than a
probability. The empirical odds of a step given its position in the bar are
computed once from the corpus and added as a bias the network cannot change, so
rediscovering "notes fall on beats" earns it nothing. Whatever accuracy it
gains above that bias must come from the audio, which is the property the first
model lacked: it reached 0.986 overall AUC where a 48-entry lookup table
reached 0.978.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from torch import nn
import numpy as np
import torch

from .encoder import MEASURE_SLOTS, AudioEncoder, EncoderConfig

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = (
    'MAX_DELTA',
    'ChartModel',
    'PlacementHead',
    'SelectionBatch',
    'SelectionHead',
    'metric_prior_logits',
)

MAX_DELTA = 96
"""Largest gap between consecutive steps represented distinctly, in slots.

:meta hide-value:
"""
_PRIOR_FLOOR = 1e-4
_PRIOR_CEILING = 1.0 - 1e-4


def metric_prior_logits(
    counts: NDArray[np.float64], totals: NDArray[np.float64]
) -> NDArray[np.float32]:
    """
    Turn observed step frequencies into log-odds per metric position.

    Parameters
    ----------
    counts : :py:class:`~numpy.ndarray`
        Steps observed at each position in the bar, shaped ``(positions,)``.
    totals : :py:class:`~numpy.ndarray`
        Slots observed at each position, shaped ``(positions,)``.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        Log-odds per position, clipped away from certainty.
    """
    rate = np.divide(counts, np.maximum(totals, 1.0))
    rate = np.clip(rate, _PRIOR_FLOOR, _PRIOR_CEILING)
    return np.log(rate / (1.0 - rate)).astype(np.float32)


class PlacementHead(nn.Module):
    """Scores each note slot as a correction to the metric prior."""

    prior: torch.Tensor
    """Fixed log-odds of a step at each position in the bar."""

    def __init__(self, width: int, prior: NDArray[np.float32] | None = None) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width // 2), nn.GELU(), nn.Linear(width // 2, 1)
        )
        empty = np.zeros(MEASURE_SLOTS, dtype=np.float32)
        self.register_buffer('prior', torch.from_numpy(prior if prior is not None else empty))

    def forward(
        self, encoded: torch.Tensor, position: torch.Tensor, weight: float = 1.0
    ) -> torch.Tensor:
        """
        Score every slot.

        Parameters
        ----------
        encoded : :py:class:`~torch.Tensor`
            Encoder output shaped ``(batch, slots, width)``.
        position : :py:class:`~torch.Tensor`
            Position within the bar per slot, shaped ``(batch, slots)``.
        weight : float
            How much of the metric prior to add. One reproduces training. Less
            than one leans on the audio, which matters when the scores are used
            for ranking rather than as probabilities.

        Returns
        -------
        :py:class:`~torch.Tensor`
            Step logits shaped ``(batch, slots)``.
        """
        correction: torch.Tensor = self.project(encoded).squeeze(-1)
        return correction + weight * self.prior[position]


class SelectionBatch(NamedTuple):
    """One batch of steps presented to :py:class:`SelectionHead`."""

    delta: torch.Tensor
    """Slots since the previous step, shaped ``(batch, steps)``."""
    position: torch.Tensor
    """Position within the bar per step, shaped ``(batch, steps)``."""
    previous: torch.Tensor
    """Previous pattern token per step, shaped ``(batch, steps)``."""
    slots: torch.Tensor
    """Encoder slot index of each step, shaped ``(batch, steps)``."""


class SelectionHead(nn.Module):
    """Chooses the panels of each step, reading the audio at that moment."""

    def __init__(
        self, width: int, vocabulary: int, layers: int = 6, heads: int = 6, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.pattern = nn.Embedding(vocabulary + 1, width)
        self.delta = nn.Embedding(MAX_DELTA + 1, width)
        self.position = nn.Embedding(MEASURE_SLOTS, width)
        layer = nn.TransformerEncoderLayer(
            width,
            heads,
            4 * width,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation='gelu',
        )
        self.decoder = nn.TransformerEncoder(layer, layers)
        self.output = nn.Linear(width, vocabulary)

    def forward(self, encoded: torch.Tensor, batch: SelectionBatch) -> torch.Tensor:
        """
        Score the pattern of each step.

        Parameters
        ----------
        encoded : :py:class:`~torch.Tensor`
            Encoder output shaped ``(batch, slots, width)``.
        batch : SelectionBatch
            The steps to score.

        Returns
        -------
        :py:class:`~torch.Tensor`
            Pattern logits shaped ``(batch, steps, vocabulary)``.
        """
        gathered = torch.gather(
            encoded, 1, batch.slots.unsqueeze(-1).expand(-1, -1, encoded.shape[-1])
        )
        tokens = (
            gathered
            + self.pattern(batch.previous)
            + self.delta(batch.delta.clamp(0, MAX_DELTA))
            + self.position(batch.position)
        )
        causal = nn.Transformer.generate_square_subsequent_mask(
            tokens.shape[1], device=tokens.device
        )
        decoded: torch.Tensor = self.decoder(tokens, mask=causal, is_causal=True)
        logits: torch.Tensor = self.output(decoded)
        return logits


class ChartModel(nn.Module):
    """The shared encoder with both heads attached."""

    def __init__(
        self,
        vocabulary: int,
        config: EncoderConfig | None = None,
        prior: NDArray[np.float32] | None = None,
    ) -> None:
        super().__init__()
        settings = config if config is not None else EncoderConfig()
        self.encoder = AudioEncoder(settings)
        self.placement = PlacementHead(settings.model_dimension, prior)
        self.selection = SelectionHead(settings.model_dimension, vocabulary)

    def encode(
        self,
        features: torch.Tensor,
        difficulty: torch.Tensor,
        meter: torch.Tensor,
        scale: torch.Tensor,
        style: torch.Tensor,
        rate: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode a window of audio.

        Parameters
        ----------
        features : :py:class:`~torch.Tensor`
            Fine-grid features shaped ``(batch, fine_slots, channels)``.
        difficulty : :py:class:`~torch.Tensor`
            Difficulty index per example, shaped ``(batch,)``.
        meter : :py:class:`~torch.Tensor`
            Rating per example, shaped ``(batch,)``.
        scale : :py:class:`~torch.Tensor`
            Rating scale index per example, shaped ``(batch,)``.
        style : :py:class:`~torch.Tensor`
            Performance style index per example, shaped ``(batch,)``.
        rate : :py:class:`~torch.Tensor`
            Note-rate bucket per example, shaped ``(batch,)``.

        Returns
        -------
        :py:class:`~torch.Tensor`
            One vector per note slot.
        """
        encoded: torch.Tensor = self.encoder(features, difficulty, meter, scale, style, rate)
        return encoded
