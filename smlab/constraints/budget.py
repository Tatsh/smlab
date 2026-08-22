"""What the chart has spent so far, and what that forbids next."""

from __future__ import annotations

import numpy as np

from smlab.playability import STREAM_SECONDS

from .codes import HOLD_CODES, LEFT_PANEL, PANELS, RIGHT_PANEL
from .panels import USAGE_DECAY

__all__ = ('Budget',)

_MAX_CROSSED_RUN = 2
"""
Longest stretch of consecutive crossed steps allowed at eighth speed.

A share bounds how many crossovers a chart holds but says nothing about how they clump, and the
clump is what hurts: one crossed step is a step, four in a row is a wall. Across 7285 crossed
stretches in 500 corpus charts rated twelve to eighteen, 83 per cent are a single step and 97 per
cent are one or two, leaving 2.7 per cent at three or more.
"""
_MAX_CROSSED_RUN_FAST = 1
"""
Longest stretch of consecutive crossed steps allowed at sixteenth speed.

One. Two crossovers arriving 100 milliseconds apart is where a stream stops being danceable, so the
second is always barred however few the chart has spent so far.
"""
_FAST_CROSSOVER_FRACTION = 1.0 / 6.0
"""
How much of the crossover allowance a sixteenth stream may spend.

A crossover at an eighth is a flourish and at a sixteenth it is a scramble, so sixteenths get a
sixth of what the chart allows generally, which is about two per cent of their notes against the
11.6 per cent real charts write at the median. That is rare rather than absent; an allowance of zero
bars them outright, at any speed.
"""
_SIXTEENTH_SLOTS = 3
"""Grid slots one sixteenth note spans, used to tell a fast run from a slow one."""
_MAX_SAME_ARROW = 6
"""
Longest run of the identical single arrow before that panel is barred.

For feet the jack limit breaks a run on its own, but a keyboard has no such limit and nothing else
stops one. Across 221 keyboard-marked corpus charts rated ten and above the longest such run is 4 at
the median, 5 at the seventy-fifth percentile and 8 at the ninetieth, against a measured 19 with no
cap at all. Six sits between the corpus p75 and p90.
"""


class Budget:
    """
    What the chart has spent so far, and what that forbids next.

    Freezes, crossovers and panel use are all things the model reaches for more often than charts
    do. None of them can simply be banned, because a chart with none of any of them is also wrong,
    so each is rationed against a rate measured from the corpus.
    """

    def __init__(self) -> None:
        self.usage = np.zeros(PANELS, dtype=np.float64)
        """Decayed count of how often each panel has been stepped on."""
        self._notes = 0
        self._freezes = 0
        self._foot = 0
        self._streamed = 0
        self._crossings = 0
        self._crossed_run = 0
        self._fast = False
        self._fast_crossings = 0
        self._fast_streamed = 0
        self._last: frozenset[int] = frozenset()
        self._repeats = 0
        self._rows = 0
        self._jumps = 0

    def crossed(self, allowance: float) -> frozenset[int]:
        """
        Return the panels that would cross the feet and are out of budget.

        Two things are rationed, and a sixteenth stream is held to a tighter version of both: the
        share of the chart that crosses at all, and how many crossed steps may follow one another.
        The share alone bounds the total while letting them arrive in a clump, which is the part
        that is unpleasant to play.

        Parameters
        ----------
        allowance : float
            Share of streamed notes that may land on a crossed foot.

        Returns
        -------
        frozenset[int]
            Panels to keep off, which is empty while budget remains.
        """
        spent = self._crossings > allowance * self._streamed
        if self._fast:
            spent = spent or self._fast_crossings > (
                _FAST_CROSSOVER_FRACTION * allowance * self._fast_streamed
            )
        if not spent:
            return frozenset()
        return self._crossing_panel()

    def overrun(self, allowance: float) -> frozenset[int]:
        """
        Return the panel that would cross when crossing is no longer allowed.

        This is a floor rather than a preference, so unlike :py:meth:`crossed` it survives every
        level of relaxation and is given up only when nothing else can be placed at all. A share can
        be surrendered and the chart still reads; a wall of crossovers in a sixteenth stream cannot.

        Parameters
        ----------
        allowance : float
            Share of streamed notes that may land on a crossed foot. Zero bars crossing outright, at
            any speed.

        Returns
        -------
        frozenset[int]
            The panel to keep off, which is empty while crossing is permitted.
        """
        if allowance <= 0:
            return self._crossing_panel()
        cap = _MAX_CROSSED_RUN_FAST if self._fast else _MAX_CROSSED_RUN
        if self._crossed_run < cap:
            return frozenset()
        return self._crossing_panel()

    def _crossing_panel(self) -> frozenset[int]:
        """
        Return the panel the limb whose turn it is would have to cross to.

        Returns
        -------
        frozenset[int]
            The single panel that would cross.
        """
        return frozenset({RIGHT_PANEL if self._foot == 0 else LEFT_PANEL})

    def enter_run(self, gap_seconds: float, seconds_per_slot: float) -> bool:
        """
        Note whether this step falls inside a run, and swap limbs if it does.

        This applies whatever the chart is for. A keyboard has no legs to cross, but the shape a
        crossover makes — the pattern reaching past itself between two notes a sixteenth apart — is
        as awkward under four fingers as under two feet.

        Parameters
        ----------
        gap_seconds : float
            Seconds since the previous row.
        seconds_per_slot : float
            How long one grid slot lasts, which fixes what counts as a sixteenth at this tempo.

        Returns
        -------
        bool
            Whether the step is part of a run.
        """
        self._fast = gap_seconds <= _SIXTEENTH_SLOTS * seconds_per_slot
        if gap_seconds > STREAM_SECONDS:
            return False
        self._streamed += 1
        self._fast_streamed += self._fast
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
        self.usage *= USAGE_DECAY
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
            self._foot = 1 if landed == RIGHT_PANEL else 0
            self._crossed_run = 0
        elif (self._foot == 0 and landed == RIGHT_PANEL) or (
            self._foot == 1 and landed == LEFT_PANEL
        ):
            self._crossings += 1
            self._fast_crossings += self._fast
            self._crossed_run += 1
        else:
            self._crossed_run = 0

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
