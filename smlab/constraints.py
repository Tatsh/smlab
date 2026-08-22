"""
The rulebook a generated chart is decoded under.

Neither head knows anything about feet. The placement head ranks slots and the
selection head ranks patterns, and left alone they will pin a foot for two bars,
retap a panel in 89 milliseconds, or spend a whole chart on the two middle
panels. Everything here is a rule measured off the corpus and applied at decode
time, separately from the model that proposes the notes.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from .dataset import SUBDIVISIONS_PER_BEAT
from .encoder import MEASURE_SLOTS
from .playability import MAX_FEET, MAX_LIMBS

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .generate import GenerationConfig
    from .vocab import Vocabulary

__all__ = (
    'HOLD_CODE',
    'HOLD_CODES',
    'HOLD_SLOTS',
    'ROLL_CODE',
    'TAIL_CODE',
    'Budget',
    'allowed',
    'crowded',
    'on_grid',
    'panel_bias',
    'panel_membership',
    'permitted',
    'subdivision_quota',
    'thin_measures',
)

TAIL_CODE = 3
"""Panel code closing a freeze."""
HOLD_CODE = 2
"""Panel code opening a freeze."""
ROLL_CODE = 4
"""Panel code opening a roll."""
HOLD_CODES = frozenset({HOLD_CODE, ROLL_CODE})
"""Panel codes that open something the foot must stay on."""
HOLD_SLOTS = 12
"""
How long a freeze runs before its tail is written, in grid slots.

One beat. Across 8811 corpus freezes at ratings ten to eighteen the median is
exactly that, three quarters end within two beats and only 4.6 per cent last
longer than four. The selection head almost never picks a tail pattern of its
own accord, so without a short timeout every freeze runs to the cap: measured
at eight beats, a foot stays pinned for two whole bars.
"""

_MINE_CODE = 5
_PANELS = 4
_LEFT_PANEL = 0
_RIGHT_PANEL = 3
_SPARSE_FRACTION = 0.34
_MAX_EMPTY_SHARE = 0.10
_SIXTEENTH_STRIDE = 3
_TWELFTH_STRIDE = 4
_RELAX_CROSSOVERS = 1
_RELAX_RATIONS = 2
_RELAX_JUMPS = 3
_STREAM_SECONDS = 0.16
_MIN_FREEZE_SLOTS = 6
"""
Gap a freeze needs before it may start, in grid slots.

Half a beat. A freeze pins one foot, so every note under it falls to the other,
and real charts hardly ever ask for that: over 8811 corpus freezes the other
foot plays nothing at all during three quarters of them, the ninetieth
percentile is 2.04 notes per second and only 1.7 per cent exceed four. Holding
a panel while the other foot runs a sixteenth stream counts as two limbs and is
unplayable in practice.

Requiring room before the next note, and ending the freeze when that note
arrives, makes the other foot silent for its whole span by construction.
"""
_MAX_SAME_ARROW = 6
"""
Longest run of the identical single arrow before that panel is barred.

For feet the jack limit breaks a run on its own, but a keyboard has no such
limit and nothing else stops one. Across 221 keyboard-marked corpus charts
rated ten and above the longest such run is 4 at the median, 5 at the
seventy-fifth percentile and 8 at the ninetieth, against a measured 19 with no
cap at all. Six sits between the corpus p75 and p90.
"""
_USAGE_DECAY = 0.957
"""
How much the record of panel use fades with each note.

Balancing cumulative totals keeps the four panels even over a whole song while
letting any single measure sit on one arrow, since the totals barely move. A
half-life of about sixteen notes, roughly a measure of eighths, makes the same
correction answer for the passage being written rather than for the song.

This correction is not cosmetic. The model reaches for the two middle panels
about twice as often as the outer two whatever it is trained on, and mirroring
the training data cannot reach that bias — see
:py:data:`~smlab.chart_data.MIRRORS`. Correcting it here is the only thing that
works, so :py:attr:`~smlab.generate.GenerationConfig.balance` defaults high.
"""
_CROSSOVER_SHARE = 0.15
"""
Share of streamed notes that may land on a crossed foot.

Inside a run the feet alternate, so the panel sequence decides whether a step
lands crossed over. Charts rated twelve to sixteen cross on 12 per cent of
their streamed notes at the median, 15 at the seventy-fifth percentile and 18.9
at the ninetieth.

Set at the seventy-fifth percentile rather than the median, because barring a
crossover can only ever bar the left or right panel: up and down are never
crossovers. Held at the median the rule fires on a quarter of all steps and
starves the outer two panels, costing the right panel about four points of its
share. At the seventy-fifth it caps the tail without shaping the whole chart.
"""
_MIN_JACK_SECONDS = 0.15
"""
Shortest gap allowed before stepping the same panel again.

:py:data:`~smlab.playability.FAST_JACK_SECONDS` answers a different question:
whether a chart is physically possible with two feet, which is how the corpus
is labelled. What a chart may idiomatically ask for is slower than what a foot
can technically do. Measured over 1931 feet-style charts and 550 thousand
same-panel intervals, the first percentile is 150 milliseconds and only 0.4 per
cent fall under 130. A sixteenth at 142.5 BPM is 105 milliseconds, so the
classifier's threshold of 90 lets through a jack that human charts essentially
never write.
"""
_MIN_JUMP_SECONDS = 0.13
"""
Shortest gap allowed before a jump, when the chart must be danceable.

From the same charts, over 44 thousand jumps: 3.5 per cent follow a gap under
130 milliseconds and the fifth percentile is 150. Jumps land on quarters and
eighths, and a jump arriving a sixteenth after the previous note asks both feet
to move in the time one of them normally gets.
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

Indexed by whole notes per second. How fine a chart's rhythm is follows from
how dense it is rather than from its rating, so this is read against the target
rate. The eight-to-nine row is interpolated; the rest are measured. Shares do
not total a hundred because the remainder is the handful of notes real charts
put off the sixteenth grid.

:meta hide-value:
"""


def subdivision_quota(rate: float, wanted: int) -> tuple[int, int, int]:
    """
    Split a note budget across quarters, eighths and sixteenths.

    How fine a chart's rhythm is follows from how dense it is, not from the
    rating directly. Measured across the corpus, a chart running three to four
    notes per second is 59 per cent quarters and 3 per cent sixteenths, while
    one running nine to ten is 31 and 39. Reproducing that split is what keeps
    a chart from reading as a wall of off-colour arrows.

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
    # Resting a measure is only worth doing for the emptiest of them. Charts at
    # ratings twelve to sixteen leave 7 per cent of their measures silent at the
    # median and 12.6 at the ninetieth percentile, so clearing every thin
    # measure a low note budget produces would strand the chart in silence.
    span = max(counts) + 1
    return set(thin[: int(_MAX_EMPTY_SHARE * span)])


def on_grid(logits: NDArray[np.float32], *, triplets: bool) -> NDArray[np.float32]:
    """
    Put scores for slots off the working grid out of reach.

    The note grid runs in twelfths of a beat so that both sixteenths and
    triplets can be written, but no chart uses both freely. Ranking every slot
    equally scatters notes onto twenty-fourths and forty-eighths, which is what
    a chart full of stray off-colour arrows is. Across 260 thousand corpus
    notes at ratings eight to fifteen, 97.3 per cent are quarters, eighths or
    sixteenths, and 74 per cent of charts never leave that grid at all.

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
        One flag per step, true where a freeze would still be held when the
        next note lands and so would leave the other foot playing alone.
    """
    positions = np.asarray(slots, dtype=np.int64)
    gaps = np.diff(positions, append=positions[-1] + HOLD_SLOTS)
    return np.asarray(gaps < _MIN_FREEZE_SLOTS, dtype=np.bool_)


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
    table = np.zeros((len(vocabulary), _PANELS), dtype=np.float32)
    for token in range(len(vocabulary)):
        for panel in vocabulary.stepped_panels(token):
            table[token, panel] = 1.0
    return table


def panel_bias(membership: NDArray[np.float32], usage: NDArray[np.float64]) -> NDArray[np.float32]:
    """
    Score each pattern by how far its panels are from their fair share.

    Real charts spread their notes evenly over the four panels: across the
    corpus at ratings eight to fifteen the median shares are 24.6, 26.5, 24.5
    and 24.5 per cent. The model does not. Left to itself it produces 20.6,
    32.5, 28.7 and 18.1, favouring the two middle panels in every chart
    measured, across two songs and three difficulties. Nudging each pattern by
    how over-used its panels are so far pulls the totals back towards even
    without dictating any individual step.

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
    share = (usage + 1.0) / (usage.sum() + _PANELS)
    return np.asarray(membership @ -np.log(share * _PANELS), dtype=np.float32)


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
    # More than half the vocabulary carries a mine, so leaving them
    # unconstrained buries the chart in them.
    if not config.mines and _MINE_CODE in codes:
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
    spent: bool = False,
    busy: bool = False,
    crowded_jumps: bool = False,
    relax: int = 0,
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
        The shorter of the gaps either side of this row. A jump is hard to
        leave as well as to reach, so the tighter side is what decides.
    crossed : frozenset[int]
        Panels that would land on a crossed foot, and are out of budget.
    spent : bool
        Whether the chart has already used its allowance of freeze notes.
    busy : bool
        Whether the passage around this row is too dense to pin a foot.
    crowded_jumps : bool
        Whether the chart has used its allowance of two-note rows.
    relax : int
        How many preferences to give up, from crossover balance upwards.

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
        # The empty pattern is in the vocabulary, and nothing else here rules
        # it out: it steps on nothing, so it conflicts with no held panel and
        # trips no timing rule. Left in, the selection head takes it at slots
        # the placement head chose for a note, and the chart quietly loses
        # them — 28 per cent of a rating sixteen chart on one measurement,
        # scattered as holes the placement head never asked for. Where the
        # rests go is placement's decision; selection only picks panels.
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
    spent: bool,
    busy: bool,
    crowded_jumps: bool,
) -> NDArray[np.bool_]:
    """
    Build the usable-pattern mask, giving up preferences before rules.

    All the constraints can bite at once, and when they do something still has
    to be placed. Dropping them all together is what lets a chart meant for two
    feet come out with a panel repeating every 89 milliseconds. They are given
    up in order of how much they matter: crossover balance first, then the
    freeze and mine rations, then the rule against fast jumps. Hold consistency
    and the jack limit are never given up, because a chart that breaks the
    first will not load and one that breaks the second cannot be danced.

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
    spent : bool
        Whether the chart has used its allowance of freeze notes.
    busy : bool
        Whether the passage around this row is too dense to pin a foot.
    crowded_jumps : bool
        Whether the chart has used its allowance of two-note rows.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        One flag per vocabulary entry, from the strictest level that leaves
        anything usable.
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
            spent=spent,
            busy=busy,
            crowded_jumps=crowded_jumps,
            relax=relax,
        )
        if mask.any():
            return mask
    return mask


class Budget:
    """
    What the chart has spent so far, and what that forbids next.

    Freezes, crossovers and panel use are all things the model reaches for more
    often than charts do. None of them can simply be banned, because a chart
    with none of any of them is also wrong, so each is rationed against a rate
    measured from the corpus.
    """

    def __init__(self) -> None:
        self.usage = np.zeros(_PANELS, dtype=np.float64)
        """Decayed count of how often each panel has been stepped on."""
        self._notes = 0
        self._freezes = 0
        self._foot = 0
        self._streamed = 0
        self._crossings = 0
        self._last: frozenset[int] = frozenset()
        self._repeats = 0
        self._rows = 0
        self._jumps = 0

    def crossed(self) -> frozenset[int]:
        """
        Return the panels that would cross the feet and are out of budget.

        Returns
        -------
        frozenset[int]
            Panels to keep off, which is empty while budget remains.
        """
        if self._crossings <= _CROSSOVER_SHARE * self._streamed:
            return frozenset()
        return frozenset({_RIGHT_PANEL if self._foot == 0 else _LEFT_PANEL})

    def enter_run(self, gap_seconds: float, *, style: str) -> bool:
        """
        Note whether this step falls inside a run, and swap feet if it does.

        Parameters
        ----------
        gap_seconds : float
            Seconds since the previous row.
        style : str
            Which physical constraints apply.

        Returns
        -------
        bool
            Whether the step is part of a run.
        """
        if gap_seconds > _STREAM_SECONDS or style != 'feet':
            return False
        self._streamed += 1
        self._foot ^= 1
        return True

    def freezes_spent(self, allowance: float) -> bool:
        """
        Report whether the chart has used up its freezes.

        Parameters
        ----------
        allowance : float
            Share of notes that may be freeze heads.

        Returns
        -------
        bool
            Whether no more freezes may be placed.
        """
        return self._freezes > allowance * max(self._notes, 1)

    def jumps_spent(self, allowance: float) -> bool:
        """
        Report whether the chart has used up its jumps.

        Parameters
        ----------
        allowance : float
            Share of rows that may carry more than one note.

        Returns
        -------
        bool
            Whether the next row must be a single note.
        """
        return self._jumps > allowance * max(self._rows, 1)

    def record(self, stepped: frozenset[int], codes: list[int], *, in_run: bool) -> None:
        """
        Account for the pattern just chosen.

        Parameters
        ----------
        stepped : frozenset[int]
            Panels the pattern steps on.
        codes : list[int]
            The four panel codes.
        in_run : bool
            Whether this step fell inside a run.
        """
        self.usage *= _USAGE_DECAY
        self._rows += 1
        if len(stepped) > 1:
            self._jumps += 1
        if stepped == self._last and len(stepped) == 1:
            self._repeats += 1
        else:
            self._last = stepped
            self._repeats = 1
        for panel in stepped:
            self.usage[panel] += 1
        self._notes += len(stepped)
        self._freezes += sum(1 for code in codes if code in HOLD_CODES)
        if len(stepped) != 1:
            return
        landed = next(iter(stepped))
        if not in_run:
            # Outside a run the feet reset to whichever one the step suits.
            self._foot = 1 if landed == _RIGHT_PANEL else 0
        elif (self._foot == 0 and landed == _RIGHT_PANEL) or (
            self._foot == 1 and landed == _LEFT_PANEL
        ):
            self._crossings += 1

    def stale(self) -> frozenset[int]:
        """
        Return the panel that has repeated too long to be used again.

        Returns
        -------
        frozenset[int]
            The over-used panel, or nothing while the run is short enough.
        """
        if self._repeats < _MAX_SAME_ARROW:
            return frozenset()
        return self._last
