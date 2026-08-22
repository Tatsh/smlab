"""
Audio encoder shared by placement and selection.

*Where* a step goes and *which panels* it uses are different questions asked of
the same music, so both heads read one encoder rather than each learning its
own view of the audio.

Three ideas shape it:

* **Difficulty modulates every layer.** A rating does not merely scale how many
  notes appear; it decides which layer of the music to follow. Beginner charts
  track the kick, Challenge charts track a drum fill or a vocal line. Feature
  modulation applies a learnt scale and shift at each block, so the rating can
  gate whole stems, which concatenating one embedding at the input cannot.
* **Attention is local at the slot level and global at the measure level.** A
  song is a few thousand slots, too many for full attention, but only a hundred
  or so measures. Structure is a measure-scale phenomenon anyway: charters
  reuse a pattern when the music repeats, so a chorus should be able to attend
  to the previous chorus.
* **Placement predicts a correction, not a probability.** The empirical odds of
  a step given its position in the bar are supplied as a fixed bias, so the
  network cannot score well by rediscovering that notes fall on beats. It can
  only earn its parameters by reading the audio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import override
import math

from torch import nn
from torch.nn import functional as F  # noqa: N812
import torch

from .chart import DIFFICULTIES
from .features import TOTAL_CHANNELS

__all__ = (
    'CONDITION_DIMENSION',
    'MEASURE_SLOTS',
    'STYLES',
    'AudioEncoder',
    'EncoderConfig',
    'FiLMConditioner',
)

MEASURE_SLOTS = 48
"""Note-grid slots in one measure.

:meta hide-value:
"""
CONDITION_DIMENSION = 128
"""Width of the conditioning vector difficulty and rating produce.

:meta hide-value:
"""
MAX_METER = 21
MAX_RATE = 16
"""
Number of note-rate buckets the conditioner knows, one per whole note/second.

The rating alone is a lossy description of how dense a chart is. The classic
ten-point scale saturates at the top, where charts labelled ten run anywhere
from 3.5 to 6.9 notes per second, and a keyboard chart runs roughly twice as
dense as a pad chart carrying the same number. Conditioning on the measured
rate lets the model learn density directly instead of inferring it from an
unreliable integer.

:meta hide-value:
"""
"""Highest rating represented, with larger values clamped.

:meta hide-value:
"""
SCALES = (10, 20)
"""Rating scales the corpus uses.

:meta hide-value:
"""
STYLES = ('feet', 'hands', 'keyboard')
"""
How a chart may be performed, in conditioning-index order.

A pad chart is not a keyboard chart with its illegal rows removed; it is
composed differently, alternating feet and placing jumps on accents. Telling
the network which idiom it is writing in lets it learn each one, rather than
having a decode-time filter mangle patterns it never knew were disallowed.

:meta hide-value:
"""


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    """Shape of the audio encoder."""

    channels: int = 256
    """Width of the convolutional front end."""
    dropout: float = 0.1
    """Dropout applied inside transformer blocks."""
    heads: int = 6
    """Attention heads per transformer layer."""
    local_blocks: int = 4
    """Residual dilated convolution blocks before downsampling."""
    measure_layers: int = 6
    """Transformer layers over measures, with full attention."""
    model_dimension: int = 384
    """Width of the transformer stacks."""
    slot_layers: int = 6
    """Transformer layers over note slots, with local attention."""
    window: int = 192
    """Slots each position may attend to, centred on itself."""


class FiLMConditioner(nn.Module):
    """Produces a per-layer scale and shift from difficulty and rating."""

    def __init__(self, layers: int, width: int) -> None:
        super().__init__()
        self.difficulty = nn.Embedding(len(DIFFICULTIES), CONDITION_DIMENSION)
        self.meter = nn.Embedding(MAX_METER, CONDITION_DIMENSION)
        self.scale = nn.Embedding(len(SCALES), CONDITION_DIMENSION)
        self.style = nn.Embedding(len(STYLES), CONDITION_DIMENSION)
        self.rate = nn.Embedding(MAX_RATE, CONDITION_DIMENSION)
        self.project = nn.Sequential(
            nn.Linear(5 * CONDITION_DIMENSION, 2 * CONDITION_DIMENSION),
            nn.GELU(),
            nn.Linear(2 * CONDITION_DIMENSION, layers * 2 * width),
        )
        self.layers = layers
        """Number of blocks this conditions."""
        self.width = width
        """Channel width each block expects."""

    @override
    def forward(
        self,
        difficulty: torch.Tensor,
        meter: torch.Tensor,
        scale: torch.Tensor,
        style: torch.Tensor,
        rate: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return modulation parameters for every conditioned block.

        Parameters
        ----------
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
            Modulation shaped ``(batch, layers, 2, width)``, holding a scale and
            a shift for each block.
        """
        combined = torch.cat(
            [
                self.difficulty(difficulty),
                self.meter(meter),
                self.scale(scale),
                self.style(style),
                self.rate(rate),
            ],
            dim=-1,
        )
        projected: torch.Tensor = self.project(combined)
        return projected.view(-1, self.layers, 2, self.width)


def _modulate(features: torch.Tensor, modulation: torch.Tensor) -> torch.Tensor:
    """
    Apply a feature-wise scale and shift.

    Parameters
    ----------
    features : :py:class:`~torch.Tensor`
        Activations shaped ``(batch, length, width)``.
    modulation : :py:class:`~torch.Tensor`
        Scale and shift shaped ``(batch, 2, width)``.

    Returns
    -------
    :py:class:`~torch.Tensor`
        Modulated activations of the same shape.
    """
    gamma = modulation[:, 0].unsqueeze(1)
    beta = modulation[:, 1].unsqueeze(1)
    return features * (1.0 + gamma) + beta


class _ConvBlock(nn.Module):
    """A dilated convolution over the fine grid with a residual path."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=5, padding=2 * dilation, dilation=dilation
        )
        self.norm = nn.GroupNorm(8, channels)

    @override
    def forward(self, inputs: torch.Tensor, modulation: torch.Tensor) -> torch.Tensor:
        """
        Apply the block.

        Parameters
        ----------
        inputs : :py:class:`~torch.Tensor`
            Activations shaped ``(batch, channels, length)``.
        modulation : :py:class:`~torch.Tensor`
            Scale and shift shaped ``(batch, 2, channels)``.

        Returns
        -------
        :py:class:`~torch.Tensor`
            Activations of the same shape.
        """
        hidden = F.gelu(self.norm(self.conv(inputs)))
        hidden = _modulate(hidden.transpose(1, 2), modulation).transpose(1, 2)
        return inputs + hidden


class _AttentionBlock(nn.Module):
    """A transformer block with optional local masking."""

    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm_attention = nn.LayerNorm(width)
        self.norm_feed = nn.LayerNorm(width)
        self.feed = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * width, width)
        )

    @override
    def forward(
        self, inputs: torch.Tensor, modulation: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        """
        Apply the block.

        Parameters
        ----------
        inputs : :py:class:`~torch.Tensor`
            Activations shaped ``(batch, length, width)``.
        modulation : :py:class:`~torch.Tensor`
            Scale and shift shaped ``(batch, 2, width)``.
        mask : :py:class:`~torch.Tensor` | None
            Additive attention mask, or ``None`` for full attention.

        Returns
        -------
        :py:class:`~torch.Tensor`
            Activations of the same shape.
        """
        normed = self.norm_attention(inputs)
        attended, _ = self.attention(normed, normed, normed, attn_mask=mask, need_weights=False)
        hidden: torch.Tensor = inputs + attended
        projected: torch.Tensor = self.feed(self.norm_feed(hidden))
        return hidden + _modulate(projected, modulation)


def local_mask(length: int, window: int, device: torch.device) -> torch.Tensor:
    """
    Build an additive mask restricting attention to a neighbourhood.

    Parameters
    ----------
    length : int
        Sequence length.
    window : int
        Positions either side that remain visible.
    device : :py:class:`~torch.device`
        Device the mask is built on.

    Returns
    -------
    :py:class:`~torch.Tensor`
        Mask shaped ``(length, length)``, zero where attention is permitted.
    """
    positions = torch.arange(length, device=device)
    distance = (positions[:, None] - positions[None, :]).abs()
    return torch.where(distance <= window // 2, 0.0, -math.inf)


class AudioEncoder(nn.Module):
    """Turns stem features into one vector per note slot."""

    def __init__(
        self, config: EncoderConfig | None = None, channels_in: int = TOTAL_CHANNELS
    ) -> None:
        super().__init__()
        settings = config if config is not None else EncoderConfig()
        self.config = settings
        """The shape this encoder was built with."""
        self.stem = nn.Conv1d(channels_in, settings.channels, kernel_size=5, padding=2)
        self.blocks = nn.ModuleList(
            _ConvBlock(settings.channels, 2**index) for index in range(settings.local_blocks)
        )
        self.downsample = nn.Conv1d(
            settings.channels, settings.model_dimension, kernel_size=2, stride=2
        )
        self.slot_layers = nn.ModuleList(
            _AttentionBlock(settings.model_dimension, settings.heads, settings.dropout)
            for _ in range(settings.slot_layers)
        )
        self.measure_project = nn.Linear(2 * settings.model_dimension, settings.model_dimension)
        self.measure_layers = nn.ModuleList(
            _AttentionBlock(settings.model_dimension, settings.heads, settings.dropout)
            for _ in range(settings.measure_layers)
        )
        total = settings.local_blocks + settings.slot_layers + settings.measure_layers
        self.film = FiLMConditioner(total, settings.model_dimension)
        self.conv_film = nn.Linear(settings.model_dimension, settings.channels)
        self.output_norm = nn.LayerNorm(settings.model_dimension)

    @override
    def forward(
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
            One vector per note slot, shaped ``(batch, note_slots, width)``.
        """
        modulation = self.film(difficulty, meter, scale, style, rate)
        index = 0
        hidden = self.stem(features.transpose(1, 2))
        for block in self.blocks:
            hidden = block(hidden, self._conv_modulation(modulation[:, index]))
            index += 1
        slots = self.downsample(hidden).transpose(1, 2)
        mask = local_mask(slots.shape[1], self.config.window, slots.device)
        for layer in self.slot_layers:
            slots = layer(slots, modulation[:, index], mask)
            index += 1
        measures = self._pool_measures(slots)
        for layer in self.measure_layers:
            measures = layer(measures, modulation[:, index], None)
            index += 1
        encoded: torch.Tensor = self.output_norm(slots + self._broadcast(measures, slots.shape[1]))
        return encoded

    @staticmethod
    def _broadcast(measures: torch.Tensor, length: int) -> torch.Tensor:
        """
        Spread measure context back across the slots it covers.

        A song rarely ends on a bar line, so the final measure is usually
        partial and pooling discards it. Its slots still need context, which the
        preceding measure supplies. Training never exercises this because every
        window is a whole number of measures.

        Parameters
        ----------
        measures : :py:class:`~torch.Tensor`
            Measure activations shaped ``(batch, measures, width)``.
        length : int
            Slots the result must cover.

        Returns
        -------
        :py:class:`~torch.Tensor`
            Activations shaped ``(batch, length, width)``.
        """
        if measures.shape[1] == 0:
            return measures.new_zeros((measures.shape[0], length, measures.shape[2]))
        spread = measures.repeat_interleave(MEASURE_SLOTS, dim=1)
        if spread.shape[1] < length:
            tail = spread[:, -1:].expand(-1, length - spread.shape[1], -1)
            spread = torch.cat([spread, tail], dim=1)
        return spread[:, :length]

    def _conv_modulation(self, modulation: torch.Tensor) -> torch.Tensor:
        """
        Narrow transformer-width modulation to the convolution width.

        Parameters
        ----------
        modulation : :py:class:`~torch.Tensor`
            Scale and shift shaped ``(batch, 2, model_dimension)``.

        Returns
        -------
        :py:class:`~torch.Tensor`
            Scale and shift shaped ``(batch, 2, channels)``.
        """
        narrowed: torch.Tensor = self.conv_film(modulation)
        return narrowed

    def _pool_measures(self, slots: torch.Tensor) -> torch.Tensor:
        """
        Summarise note slots into measures.

        Parameters
        ----------
        slots : :py:class:`~torch.Tensor`
            Slot activations shaped ``(batch, note_slots, width)``.

        Returns
        -------
        :py:class:`~torch.Tensor`
            Measure activations shaped ``(batch, measures, width)``.
        """
        batch, length, width = slots.shape
        usable = length - length % MEASURE_SLOTS
        blocks = slots[:, :usable].view(batch, -1, MEASURE_SLOTS, width)
        pooled = torch.cat([blocks.mean(dim=2), blocks.amax(dim=2)], dim=-1)
        projected: torch.Tensor = self.measure_project(pooled)
        return projected
