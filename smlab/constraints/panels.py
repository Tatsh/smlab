"""Keeping the four panels evenly used."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .codes import PANELS

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from smlab.vocab import Vocabulary

__all__ = ('panel_bias', 'panel_membership')

USAGE_DECAY = 0.957
"""
How much the record of panel use fades with each note.

Balancing cumulative totals keeps the four panels even over a whole song while letting any single
measure sit on one arrow, since the totals barely move. A half-life of about sixteen notes, roughly
a measure of eighths, makes the same correction answer for the passage being written rather than for
the song.

This correction is not cosmetic. The model reaches for the two middle panels about twice as often as
the outer two whatever it is trained on, and mirroring the training data cannot reach that bias —
see :py:data:`~smlab.chart.data.MIRRORS`. Correcting it here is the only thing that works, so
:py:attr:`~smlab.generate.GenerationConfig.balance` defaults high.
"""


def panel_membership(vocabulary: Vocabulary) -> NDArray[np.float32]:
    """
    Build the table of which panels each pattern steps on.

    Parameters
    ----------
    vocabulary : Vocabulary
        Pattern vocabulary.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        A ``(patterns, panels)`` indicator matrix.
    """
    table = np.zeros((len(vocabulary), PANELS), dtype=np.float32)
    for token in range(len(vocabulary)):
        for panel in vocabulary.stepped_panels(token):
            table[token, panel] = 1.0
    return table


def panel_bias(membership: NDArray[np.float32], usage: NDArray[np.float64]) -> NDArray[np.float32]:
    """
    Score each pattern by how far its panels are from their fair share.

    Real charts spread their notes evenly over the four panels: across the corpus at ratings eight
    to fifteen the median shares are 24.6, 26.5, 24.5 and 24.5 per cent. The model does not. Left to
    itself it produces 20.6, 32.5, 28.7 and 18.1, favouring the two middle panels in every chart
    measured, across two songs and three difficulties. Nudging each pattern by how over-used its
    panels are so far pulls the totals back towards even without dictating any individual step.

    Parameters
    ----------
    membership : :py:class:`~numpy.ndarray`
        Which panels each pattern steps on.
    usage : :py:class:`~numpy.ndarray`
        Times each panel has been stepped on so far.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        A score adjustment per vocabulary entry, in logits.
    """
    share = (usage + 1.0) / (usage.sum() + PANELS)
    return np.asarray(membership @ -np.log(share * PANELS), dtype=np.float32)
