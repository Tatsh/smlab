"""
Choosing where a song's preview should start.

``#SAMPLELENGTH`` needs no model: 83% of the corpus uses exactly fifteen
seconds and a further 6% uses twelve, so it is a convention rather than a
per-song judgement.

``#SAMPLESTART`` is a real decision, and the corpus supplies 3,719 human
examples of it. The task is framed as choosing one measure out of the whole
song, so the model emits a score per measure and a softmax over the song picks
the winner. That framing suits a problem with exactly one label per example and
keeps the answer on a musical boundary, which is where charters put it about
thirteen times more often than chance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import logging

from torch import nn
import numpy as np
import torch

from .dataset import FEATURE_DIMENSION, SUBDIVISIONS_PER_BEAT
from .timing import BEATS_PER_MEASURE

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .timing import TimingData

__all__ = (
    'DEFAULT_SAMPLE_LENGTH',
    'EARLIEST_FRACTION',
    'LATEST_FRACTION',
    'MAX_MEASURES',
    'PreviewModel',
    'measure_features',
    'predict_sample_start',
)

log = logging.getLogger(__name__)

DEFAULT_SAMPLE_LENGTH = 15.0
"""
Preview length in seconds.

Eighty-three per cent of the corpus uses this exact value, so it is a constant
rather than something to predict.

:meta hide-value:
"""
MAX_MEASURES = 512
"""Longest song the model considers, in measures.

:meta hide-value:
"""
SLOTS_PER_MEASURE = int(BEATS_PER_MEASURE) * SUBDIVISIONS_PER_BEAT
"""Grid slots in one measure.

:meta hide-value:
"""
POOLED_DIMENSION = FEATURE_DIMENSION * 2 + 1
"""Width per measure: mean and peak of each feature, plus relative position.

:meta hide-value:
"""
EARLIEST_FRACTION = 0.10
"""
Earliest point of a song a preview may start, as a fraction of its length.

Clamping to the span charters actually use trims the model's worst mistakes: on
held-out songs it moves the ninetieth percentile error from 63 to 56 seconds
without costing central accuracy.

:meta hide-value:
"""
LATEST_FRACTION = 0.65
"""Latest point of a song a preview may start, as a fraction of its length.

:meta hide-value:
"""
_FEATURE_SCALE = 80.0
_MIN_MEASURES = 4


def measure_features(features: NDArray[np.float16]) -> NDArray[np.float32]:
    """
    Pool beat-grid features down to one row per measure.

    A relative position channel is appended because charters favour the middle
    of a song, and the model should be able to use that prior alongside the
    audio evidence rather than having to infer it from length alone.

    Parameters
    ----------
    features : :py:class:`~numpy.ndarray`
        Beat-grid features shaped ``(slots, FEATURE_DIMENSION)``.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        Features shaped ``(measures, POOLED_DIMENSION)``.
    """
    count = min(features.shape[0] // SLOTS_PER_MEASURE, MAX_MEASURES)
    if count < 1:
        return np.zeros((0, POOLED_DIMENSION), dtype=np.float32)
    usable = features[: count * SLOTS_PER_MEASURE].astype(np.float32) / _FEATURE_SCALE
    blocks = usable.reshape(count, SLOTS_PER_MEASURE, FEATURE_DIMENSION)
    position = np.linspace(0.0, 1.0, count, dtype=np.float32).reshape(count, 1)
    return np.concatenate([blocks.mean(axis=1), blocks.max(axis=1), position], axis=1)


class PreviewModel(nn.Module):
    """Scores every measure of a song as a candidate preview start."""

    def __init__(self, channels: int = 128, hidden: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(POOLED_DIMENSION, channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(channels),
            nn.Conv1d(channels, channels, kernel_size=5, padding=4, dilation=2),
            nn.GELU(),
            nn.BatchNorm1d(channels),
        )
        self.recurrent = nn.GRU(channels, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Score each measure.

        Parameters
        ----------
        features : :py:class:`~torch.Tensor`
            Pooled features shaped ``(batch, measures, POOLED_DIMENSION)``.
        mask : :py:class:`~torch.Tensor`
            True where a measure exists, shaped ``(batch, measures)``.

        Returns
        -------
        :py:class:`~torch.Tensor`
            Logits shaped ``(batch, measures)``, with padding set to negative
            infinity so that a softmax ignores it.
        """
        encoded = self.encoder(features.transpose(1, 2)).transpose(1, 2)
        recurrent, _ = self.recurrent(encoded)
        logits: torch.Tensor = self.head(recurrent).squeeze(-1)
        return logits.masked_fill(~mask, float('-inf'))


def predict_sample_start(
    model: PreviewModel, features: NDArray[np.float16], timing: TimingData
) -> float:
    """
    Choose where a song's preview should start.

    Parameters
    ----------
    model : PreviewModel
        The trained preview model.
    features : :py:class:`~numpy.ndarray`
        Beat-grid features for the song.
    timing : TimingData
        Timing used to convert the chosen measure back into seconds.

    Returns
    -------
    float
        Preview start in seconds, never negative.
    """
    pooled = measure_features(features)
    if pooled.shape[0] < _MIN_MEASURES:
        return 0.0
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(pooled).unsqueeze(0).to(device)
        mask = torch.ones(1, pooled.shape[0], dtype=torch.bool, device=device)
        measure = int(torch.argmax(model(tensor, mask)[0]).item())
    # Clamp in measure space rather than seconds, so the guardrail cannot land
    # the preview part-way through a bar.
    total = pooled.shape[0]
    lowest = int(EARLIEST_FRACTION * total)
    highest = max(int(LATEST_FRACTION * total), lowest)
    measure = min(max(measure, lowest), highest)
    return float(max(timing.time_at_beat(measure * BEATS_PER_MEASURE), 0.0))
