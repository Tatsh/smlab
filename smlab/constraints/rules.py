"""The gate every candidate pattern passes through before it may be written."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from smlab.playability import MAX_FEET, MAX_LIMBS

from .codes import HOLD_CODE, MINE_CODE, ROLL_CODE, TAIL_CODE

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from smlab.generate import GenerationConfig
    from smlab.vocab import Vocabulary

__all__ = ('allowed', 'permitted')

_RELAX_CROSSOVERS = 1
_RELAX_RATIONS = 2
_RELAX_JUMPS = 3
_MIN_JACK_SECONDS = 0.15
"""
Shortest gap allowed before stepping the same panel again.

:py:data:`~smlab.playability.FAST_JACK_SECONDS` answers a different question: whether a chart is
physically possible with two feet, which is how the corpus is labelled. What a chart may
idiomatically ask for is slower than what a foot can technically do. Measured over 1931 feet-style
charts and 550 thousand same-panel intervals, the first percentile is 150 milliseconds and only 0.4
per cent fall under 130. A sixteenth at 142.5 BPM is 105 milliseconds, so the classifier's threshold
of 90 lets through a jack that human charts essentially never write.
"""
_MIN_JUMP_SECONDS = 0.13
"""
Shortest gap allowed before a jump, when the chart must be danceable.

From the same charts, over 44 thousand jumps: 3.5 per cent follow a gap under 130 milliseconds and
the fifth percentile is 150. Jumps land on quarters and eighths, and a jump arriving a sixteenth
after the previous note asks both feet to move in the time one of them normally gets.
"""


def _rationed(codes: tuple[int, ...], config: GenerationConfig, *, spent: bool) -> bool:
    """
    Report whether a pattern uses a note kind the chart may not spend.

    Parameters
    ----------
    codes : tuple[int, ...]
        The four panel codes.
    config : GenerationConfig
        Generation settings.
    spent : bool
        Whether the chart has used its allowance of freeze notes.

    Returns
    -------
    bool
        Whether the pattern is out of bounds.
    """
    # More than half the vocabulary carries a mine, so leaving them unconstrained buries the chart
    # in them.
    if not config.mines and MINE_CODE in codes:
        return True
    # Rolls are all but extinct: 98 per cent of real charts have none.
    if not config.rolls and ROLL_CODE in codes:
        return True
    return spent and HOLD_CODE in codes


def allowed(
    vocabulary: Vocabulary,
    config: GenerationConfig,
    held: dict[int, int],
    previous: frozenset[int],
    gap_seconds: float,
    room_seconds: float,
    crossed: frozenset[int],
    *,
    busy: bool = False,
    crowded_jumps: bool = False,
    overrun: frozenset[int] = frozenset(),
    relax: int = 0,
    spent: bool = False,
) -> NDArray[np.bool_]:
    """
    Build the mask of patterns usable at one step.

    Parameters
    ----------
    vocabulary : Vocabulary
        Pattern vocabulary.
    config : GenerationConfig
        Generation settings.
    held : dict[int, int]
        Slot at which each open hold began, keyed by panel.
    previous : frozenset[int]
        Panels stepped on at the previous row.
    gap_seconds : float
        Seconds since the previous row.
    room_seconds : float
        The shorter of the gaps either side of this row. A jump is hard to leave as well as to
        reach, so the tighter side is what decides.
    crossed : frozenset[int]
        Panels that would land on a crossed foot, and are out of budget.
    busy : bool
        Whether the passage around this row is too dense to pin a foot.
    crowded_jumps : bool
        Whether the chart has used its allowance of two-note rows.
    overrun : frozenset[int]
        Panels that would carry a stretch of crossed steps past its cap. Unlike ``crossed`` this is
        never given up.
    relax : int
        How many preferences to give up, from crossover balance upwards.
    spent : bool
        Whether the chart has already used its allowance of freeze notes.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        One flag per vocabulary entry.
    """
    limit = MAX_FEET if config.style == 'feet' else MAX_LIMBS
    mask = np.zeros(len(vocabulary), dtype=np.bool_)
    for position in range(len(vocabulary)):
        codes = vocabulary.panels_of(position)
        stepped = vocabulary.stepped_panels(position)
        # The empty pattern is in the vocabulary, and nothing else here rules it out: it steps on
        # nothing, so it conflicts with no held panel and trips no timing rule. Left in, the
        # selection head takes it at slots the placement head chose for a note, and the chart
        # quietly loses them — 28 per cent of a rating sixteen chart on one measurement, scattered
        # as holes the placement head never asked for. Where the rests go is placement's decision;
        # selection only picks panels.
        if not any(codes):
            continue
        if any(codes[panel] not in {0, TAIL_CODE} for panel in held):
            continue
        if any(code == TAIL_CODE and panel not in held for panel, code in enumerate(codes)):
            continue
        if len(stepped) + len(held) > limit:
            continue
        if config.style != 'keyboard' and gap_seconds < _MIN_JACK_SECONDS and stepped & previous:
            continue
        if overrun and stepped & overrun:
            continue
        if relax < _RELAX_JUMPS and (
            config.style == 'feet' and room_seconds < _MIN_JUMP_SECONDS and len(stepped) > 1
        ):
            continue
        if relax < _RELAX_RATIONS and (
            _rationed(codes, config, spent=spent or busy) or (crowded_jumps and len(stepped) > 1)
        ):
            continue
        if relax < _RELAX_CROSSOVERS and crossed and stepped & crossed:
            continue
        mask[position] = True
    return mask


def permitted(
    vocabulary: Vocabulary,
    config: GenerationConfig,
    held: dict[int, int],
    previous: frozenset[int],
    gap_seconds: float,
    room_seconds: float,
    crossed: frozenset[int],
    *,
    busy: bool,
    crowded_jumps: bool,
    overrun: frozenset[int],
    spent: bool,
) -> NDArray[np.bool_]:
    """
    Build the usable-pattern mask, giving up preferences before rules.

    All the constraints can bite at once, and when they do something still has to be placed.
    Dropping them all together is what lets a chart meant for two feet come out with a panel
    repeating every 89 milliseconds. They are given up in order of how much they matter: crossover
    balance first, then the freeze and mine rations, then the rule against fast jumps. Hold
    consistency and the jack limit are never given up, because a chart that breaks the first will
    not load and one that breaks the second cannot be danced.

    Parameters
    ----------
    vocabulary : Vocabulary
        Pattern vocabulary.
    config : GenerationConfig
        Generation settings.
    held : dict[int, int]
        Slot at which each open hold began, keyed by panel.
    previous : frozenset[int]
        Panels stepped on at the previous row.
    gap_seconds : float
        Seconds since the previous row.
    room_seconds : float
        The shorter of the gaps either side of this row.
    crossed : frozenset[int]
        Panels that would land on a crossed foot.
    busy : bool
        Whether the passage around this row is too dense to pin a foot.
    crowded_jumps : bool
        Whether the chart has used its allowance of two-note rows.
    overrun : frozenset[int]
        Panels that would carry a stretch of crossed steps past its cap.
    spent : bool
        Whether the chart has used its allowance of freeze notes.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        One flag per vocabulary entry, from the strictest level that leaves anything usable.
    """
    for relax in range(_RELAX_JUMPS + 1):
        mask = allowed(
            vocabulary,
            config,
            held,
            previous,
            gap_seconds,
            room_seconds,
            crossed,
            busy=busy,
            crowded_jumps=crowded_jumps,
            overrun=overrun,
            relax=relax,
            spent=spent,
        )
        if mask.any():
            return mask
    # The cap on consecutive crossed steps outlasts every preference above and is surrendered only
    # when nothing else is left, because a wall of them is worse to play than anything the other
    # rules were protecting.
    return allowed(
        vocabulary,
        config,
        held,
        previous,
        gap_seconds,
        room_seconds,
        crossed,
        busy=busy,
        crowded_jumps=crowded_jumps,
        relax=_RELAX_JUMPS,
        spent=spent,
    )
