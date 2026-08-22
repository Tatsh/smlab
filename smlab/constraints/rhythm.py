"""Where a note may land: how fine the grid is, and how the notes are spread across it."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from smlab.dataset import SUBDIVISIONS_PER_BEAT
from smlab.encoder import MEASURE_SLOTS

from .codes import HOLD_SLOTS

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ('crowded', 'on_grid', 'subdivision_quota', 'thin_measures')

_SPARSE_FRACTION = 0.34
_MAX_EMPTY_SHARE = 0.10
_SIXTEENTH_STRIDE = 3
_TWELFTH_STRIDE = 4
_MIN_FREEZE_SLOTS = 6
"""
Gap a freeze needs before it may start, in grid slots.

Half a beat. A freeze pins one foot, so every note under it falls to the other, and real charts
hardly ever ask for that: over 8811 corpus freezes the other foot plays nothing at all during three
quarters of them, the ninetieth percentile is 2.04 notes per second and only 1.7 per cent exceed
four. Holding a panel while the other foot runs a sixteenth stream counts as two limbs and is
unplayable in practice.

Requiring room before the next note, and ending the freeze when that note arrives, makes the other
foot silent for its whole span by construction.
"""
_MIX_BY_NPS = (
    (100.0, 0.0, 0.0),
    (100.0, 0.0, 0.0),
    (79.0, 19.5, 0.0),
    (59.1, 34.0, 3.1),
    (50.0, 35.6, 9.3),
    (43.6, 33.8, 18.8),
    (38.0, 31.0, 23.1),
    (34.8, 31.2, 24.7),
    (32.8, 30.8, 31.8),
    (30.7, 30.3, 38.8),
)
"""
Median share of quarters, eighths and sixteenths at each note rate.

Indexed by whole notes per second. How fine a chart's rhythm is follows from how dense it is rather
than from its rating, so this is read against the target rate. The eight-to-nine row is
interpolated; the rest are measured. Shares do not total a hundred because the remainder is the
handful of notes real charts put off the sixteenth grid.

:meta hide-value:
"""


def subdivision_quota(rate: float, wanted: int) -> tuple[int, int, int]:
    """
    Split a note budget across quarters, eighths and sixteenths.

    How fine a chart's rhythm is follows from how dense it is, not from the rating directly.
    Measured across the corpus, a chart running three to four notes per second is 59 per cent
    quarters and 3 per cent sixteenths, while one running nine to ten is 31 and 39. Reproducing that
    split is what keeps a chart from reading as a wall of off-colour arrows.

    Parameters
    ----------
    rate : float
        Target notes per second.
    wanted : int
        Total notes to place.

    Returns
    -------
    tuple[int, int, int]
        How many quarters, eighths and sixteenths to aim for.
    """
    mix = _MIX_BY_NPS[min(int(max(rate, 0.0)), len(_MIX_BY_NPS) - 1)]
    total = sum(mix)
    quarters = int(wanted * mix[0] / total)
    eighths = int(wanted * mix[1] / total)
    return quarters, eighths, wanted - quarters - eighths


def thin_measures(chosen: list[int]) -> set[int]:
    """
    Find the measures that were barely used.

    Parameters
    ----------
    chosen : list[int]
        Slots picked by score.

    Returns
    -------
    set[int]
        Measures holding far fewer notes than the chart's typical measure.
    """
    counts = Counter(slot // MEASURE_SLOTS for slot in chosen)
    busy = sorted(counts.values())
    floor = _SPARSE_FRACTION * busy[len(busy) // 2]
    thin = sorted(
        (measure for measure, count in counts.items() if count < floor),
        key=lambda measure: counts[measure],
    )
    # Resting a measure is only worth doing for the emptiest of them. Charts at ratings twelve to
    # sixteen leave 7 per cent of their measures silent at the median and 12.6 at the ninetieth
    # percentile, so clearing every thin measure a low note budget produces would strand the chart
    # in silence.
    span = max(counts) + 1
    return set(thin[: int(_MAX_EMPTY_SHARE * span)])


def on_grid(logits: NDArray[np.float32], *, triplets: bool) -> NDArray[np.float32]:
    """
    Put scores for slots off the working grid out of reach.

    The note grid runs in twelfths of a beat so that both sixteenths and triplets can be written,
    but no chart uses both freely. Ranking every slot equally scatters notes onto twenty-fourths and
    forty-eighths, which is what a chart full of stray off-colour arrows is. Across 260 thousand
    corpus notes at ratings eight to fifteen, 97.3 per cent are quarters, eighths or sixteenths, and
    74 per cent of charts never leave that grid at all.

    Parameters
    ----------
    logits : :py:class:`~numpy.ndarray`
        Placement logits per slot.
    triplets : bool
        Whether to work on the twelfth grid as well as the sixteenth one.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        The logits, with off-grid slots pushed below every on-grid one.
    """
    within = np.arange(len(logits)) % SUBDIVISIONS_PER_BEAT
    usable = within % _SIXTEENTH_STRIDE == 0
    if triplets:
        usable |= within % _TWELFTH_STRIDE == 0
    return np.where(usable, logits, -np.inf).astype(np.float32)


def crowded(slots: list[int]) -> NDArray[np.bool_]:
    """
    Mark the steps with no room to pin a foot before the next note.

    Parameters
    ----------
    slots : list[int]
        Chosen slot indices, in ascending order.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        One flag per step, true where a freeze would still be held when the next note lands and so
        would leave the other foot playing alone.
    """
    positions = np.asarray(slots, dtype=np.int64)
    gaps = np.diff(positions, append=positions[-1] + HOLD_SLOTS)
    return np.asarray(gaps < _MIN_FREEZE_SLOTS, dtype=np.bool_)
