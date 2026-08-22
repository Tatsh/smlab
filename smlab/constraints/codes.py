"""The panel codes a decoded row is written in, and how long a freeze runs."""

from __future__ import annotations

__all__ = (
    'HOLD_CODE',
    'HOLD_CODES',
    'HOLD_SLOTS',
    'LEFT_PANEL',
    'MINE_CODE',
    'PANELS',
    'RIGHT_PANEL',
    'ROLL_CODE',
    'TAIL_CODE',
)

TAIL_CODE = 3
"""Panel code closing a freeze."""
HOLD_CODE = 2
"""Panel code opening a freeze."""
ROLL_CODE = 4
"""Panel code opening a roll."""
MINE_CODE = 5
"""Panel code for a mine."""
HOLD_CODES = frozenset({HOLD_CODE, ROLL_CODE})
"""Panel codes that open something the foot must stay on."""
PANELS = 4
"""Panels a ``dance-single`` chart steps on."""
LEFT_PANEL = 0
"""Index of the leftmost panel."""
RIGHT_PANEL = 3
"""Index of the rightmost panel."""
HOLD_SLOTS = 12
"""
How long a freeze runs before its tail is written, in grid slots.

One beat. Across 8811 corpus freezes at ratings ten to eighteen the median is exactly that, three
quarters end within two beats and only 4.6 per cent last longer than four. The selection head almost
never picks a tail pattern of its own accord, so without a short timeout every freeze runs to the
cap: measured at eight beats, a foot stays pinned for two whole bars.
"""
