"""
Settings that control generation, and the rating scales they are read against.

Difficulty reaches the model twice: as conditioning, and as a target note rate
that decides how many slots to keep. The rate is the part that has to be right,
because a rating number means nothing on its own — it is only meaningful once
you know which scale it was written on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from .playability import Style

__all__ = (
    'CLASSIC_SCALE',
    'DEFAULT_BALANCE',
    'DEFAULT_SCALE',
    'NPS_BY_METER_10',
    'NPS_BY_METER_15',
    'NPS_BY_METER_20',
    'NPS_BY_METER_KEYBOARD',
    'SCALES',
    'GenerationConfig',
    'target_nps',
)

log = logging.getLogger(__name__)

NPS_BY_METER_10 = (
    0.60,
    0.88,
    1.07,
    1.43,
    1.79,
    2.19,
    2.57,
    3.00,
    3.48,
    4.10,
    4.96,
)
"""
Notes per second per rating on the classic ten-point scale.

Median over every chart in a classic-rated pack, from 829 charts at rating one
down to 37 at rating ten.

The top rung is a bucket rather than a rating. Charts labelled ten run from 3.5
to 6.9 notes per second, because everything harder than a nine had nowhere else
to go. Ask for a ten and you get the middle of that crowd; use
:py:attr:`GenerationConfig.nps` to say what you actually want.

:meta hide-value:
"""
NPS_BY_METER_15 = (
    0.60,
    0.66,
    1.21,
    1.50,
    1.87,
    2.25,
    2.64,
    3.04,
    3.63,
    4.33,
    5.18,
    5.92,
    6.67,
    7.57,
    8.47,
    9.37,
)
"""
Notes per second per rating on the In The Groove scale.

Measured from the four In The Groove packs, which stop at thirteen, with 297
charts at rating one falling to 10 at rating thirteen. Fourteen and fifteen
continue the 0.9 notes per second per level slope seen across eleven to
thirteen, so they are extrapolated rather than observed.

This scale is far steeper at the top than either DDR scale: an ITG nine is 4.33
notes per second, where a twenty-scale nine is 3.12.

:meta hide-value:
"""
NPS_BY_METER_20 = (
    0.60,
    0.70,
    0.93,
    1.19,
    1.45,
    1.82,
    2.17,
    2.49,
    2.75,
    3.12,
    3.34,
    3.61,
    3.94,
    4.35,
    4.83,
    5.30,
    5.88,
    6.23,
    6.85,
    8.07,
    9.07,
)
"""
Notes per second per rating on the twenty-point scale.

Median over every chart in a pack that rates above ten, from 691 charts at
rating one down to 11 at rating nineteen. Twenty is extrapolated; no chart in
the corpus carries it.

The previous version of this table flattened out near the top, giving eighteen,
nineteen and twenty the same rate. Real charts do the opposite and accelerate:
a nineteen runs 8.07 notes per second against a sixteen's 5.88.

:meta hide-value:
"""
NPS_BY_METER_KEYBOARD = (
    0.60,
    0.90,
    1.40,
    2.01,
    2.28,
    2.56,
    2.92,
    3.21,
    3.64,
    4.65,
    5.73,
    6.40,
    6.92,
    7.63,
    7.70,
    7.89,
    8.03,
    8.16,
    8.30,
    8.43,
    8.57,
)
"""
Rows per second per rating for charts played on a keyboard.

Hands are not feet: a keyboard chart at a given rating runs roughly twice as
dense as a pad chart at the same number. Measured over the keyboard-marked
packs on the twenty-point scale, a fifteen averages 7.89 rows per second where
a pad fifteen is 5.30. Ratings above fifteen are extrapolated along the 0.13
per level slope seen across thirteen to fifteen; ratings below three had too
few charts to measure and are interpolated.

Counted in rows rather than notes because that is what the note budget picks.
Keyboard rows carry 1.14 notes at the median, so the note rate runs above these
figures.

:meta hide-value:
"""
SCALES = (10, 15, 20)
"""
Rating scales that can be asked for.

Classic DDR rates out of ten, In The Groove out of fifteen, and X-era DDR out
of twenty. The same number means three different things across them: a nine is
4.45 notes per second in ITG, 3.81 on the classic scale and 2.64 on the modern
one.

:meta hide-value:
"""
DEFAULT_SCALE = 20
"""
Rating scale assumed when none is given.

Modern mixes rate out of twenty, so that is what a bare number most likely
means.

:meta hide-value:
"""
CLASSIC_SCALE = 10
"""
The scale the classic ten-point ratings are on.

:meta hide-value:
"""
_TABLES = {10: NPS_BY_METER_10, 15: NPS_BY_METER_15, 20: NPS_BY_METER_20}
DEFAULT_BALANCE = 4.0
"""
How hard to pull the four panels towards equal use, by default.

:meta hide-value:
"""
_JUMP_SHARE = 0.08
"""Share of rows a pad chart puts two notes on, which barely moves with rating."""
_KEYBOARD_JUMP_SHARE = 0.14
"""Share for a keyboard, whose rows carry 1.14 notes at the corpus median."""
_CROSSOVER_SHARE = 0.04
"""
Share of streamed notes that may land on a crossed foot.

Inside a run the limbs alternate, so the panel sequence decides whether a step
lands crossed: left, down, right, up puts the left foot on the right panel, and
its reverse, up, left, down, right, puts the right foot on the left one.

Real charts cross far more freely than this. Measured over 400 corpus charts
rated twelve to eighteen, eighth speed crosses on 8.5 per cent of its streamed
notes at the tenth percentile and 14.1 at the median. This sits at half the
tenth percentile deliberately: a crossover reads as a flourish once in a while
and as a nuisance when it keeps arriving, and the generator streams almost
entirely in sixteenths where it is worst. A sixteenth stream is held to a
fraction of this again, and zero bars crossing outright.
"""


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Settings controlling how a chart is generated."""

    balance: float = DEFAULT_BALANCE
    """
    How hard to pull the four panels towards equal use.

    Zero leaves the model alone, which is not a usable setting: the model puts
    roughly a third of its notes on each of the two middle panels and a sixth
    on each outer one, where the corpus is even to within two points. One
    applies the plain correction for how over-used each panel is so far, in
    logits, and leaves 13 to 15 points of spread.

    Four brings that to 4 to 8 points and costs nothing measurable — the jump
    rate, the sixteenth share and the repetition are unchanged, and crossovers
    stay inside the corpus interquartile range. Six is slightly closer again
    with no further cost, but the returns have flattened by four.
    """
    crossovers: float | None = None
    """
    Share of streamed notes that may land on a crossed foot.

    ``None`` uses the measured default and zero bars crossing outright. A
    crossover is a step the alternation puts on the far panel: left, down,
    right, up reaches the left foot across to the right panel, and its reverse,
    up, left, down, right, reaches the right foot across to the left. They read
    as a flourish at an eighth and as a scramble at a sixteenth, so a sixteenth
    stream is held to a fraction of whatever this allows and may never carry
    two of them in a row.
    """
    density: float = 1.0
    """
    Multiplier on how many steps to keep.

    Values above one lower the placement threshold and produce a busier chart.
    """
    difficulty: str = 'Hard'
    """Difficulty slot to write and to condition the model on."""
    holds: float = 0.04
    """
    Largest share of notes that may be freeze heads.

    Real charts rated eight to fourteen put freezes on 3.9 per cent of their
    notes at the median and 11.8 per cent at the ninetieth percentile. Left
    unrationed the model reaches for them roughly eight times that often, which
    both misreads the music and makes the chart look far more repetitive than
    its note patterns actually are.
    """
    jumps: float = 0.0
    """
    Share of rows that may carry two or more notes at once.

    Zero picks the rate for the style. What makes a chart hard is how fast the
    feet have to move, not how often both land together: across the corpus the
    jump rate is flat at 7 to 8 per cent of rows from rating five through
    seventeen, reaching only 9.6 at eighteen. BREAKING THE FUTURE, rated
    nineteen, jumps on 10.2 per cent of its rows while its own rating fourteen
    chart jumps on 5.2.
    """
    meter: int = 8
    """Difficulty rating to condition on."""
    mines: bool = False
    """
    Whether mines may be placed.

    Off by default. Mines appear on about one per cent of rows in real charts,
    but they occur in so many distinct combinations that they crowd the pattern
    vocabulary, and a chart peppered with them is worse than one with none.
    """
    nps: float = 0.0
    """
    Note rate to aim for, overriding what the rating implies.

    Zero defers to the rating. Set it when the label is not trustworthy, which
    on the classic scale it often is not above nine.
    """
    rolls: bool = False
    """
    Whether roll heads may be placed.

    Off by default. Ninety-eight per cent of charts in the corpus contain no
    rolls at all, so the median chart of any rating has exactly zero.
    """
    scale: int = DEFAULT_SCALE
    """Rating scale the meter is on. One of :py:data:`SCALES`."""
    seed: int = 0
    """Seed for the sampling generator."""
    style: Style = 'feet'
    """
    Which physical constraints to respect.

    ``feet`` keeps the chart danceable with two feet, ``hands`` additionally
    allows three and four panel chords, and ``keyboard`` removes the limits
    entirely.
    """
    temperature: float = 0.9
    """Softmax temperature used when sampling patterns."""
    triplets: bool = False
    """
    Whether to place notes on the twelfth-of-a-beat grid as well.

    Off by default. The note grid is fine enough to write both sixteenths and
    triplets, but charts pick one: 97.3 per cent of corpus notes are quarters,
    eighths or sixteenths and 74 per cent of charts never leave that grid.
    Ranking both grids together scatters notes onto twenty-fourths, which reads
    as a chart full of stray off-colour arrows.
    """

    @property
    def crossover_share(self) -> float:
        """
        Share of streamed notes that may land on a crossed foot.

        Returns
        -------
        float
            The override if one was given, otherwise the measured default. A
            sixteenth stream is held to a fraction of whichever applies.
        """
        return _CROSSOVER_SHARE if self.crossovers is None else self.crossovers

    @property
    def jump_share(self) -> float:
        """
        Share of rows that may carry more than one note.

        Returns
        -------
        float
            The override if one was given, otherwise what the style implies.
        """
        if self.jumps > 0:
            return self.jumps
        return _KEYBOARD_JUMP_SHARE if self.style == 'keyboard' else _JUMP_SHARE

    @property
    def rate(self) -> float:
        """
        Note rate this configuration asks for, in notes per second.

        Returns
        -------
        float
            The override if one was given, the keyboard rate when the chart is
            for a keyboard, otherwise the rate the rating implies on its scale.
        """
        if self.nps > 0:
            return self.nps
        if self.style == 'keyboard':
            table = NPS_BY_METER_KEYBOARD
            return table[min(max(self.meter, 0), len(table) - 1)]
        return target_nps(self.meter, self.scale)


def target_nps(meter: int, scale: int = DEFAULT_SCALE) -> float:
    """
    Return the note rate a chart of this rating should carry.

    Difficulty is a physical quantity: what makes a chart hard is how fast the
    feet have to move, not how many grid slots carry a note. Counting in notes
    per second therefore transfers across tempo, where a rate per slot would
    make a 250 BPM song twice as dense as a 125 BPM song of the same rating.

    Parameters
    ----------
    meter : int
        Difficulty rating.
    scale : int
        Which rating scale the meter is on. One of :py:data:`SCALES`; anything
        else is rounded to the nearest of them.

    Returns
    -------
    float
        Notes per second.
    """
    nearest = min(SCALES, key=lambda candidate: abs(candidate - scale))
    table = _TABLES[nearest]
    return table[min(max(meter, 0), len(table) - 1)]
