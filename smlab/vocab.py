"""
Vocabulary of note-row patterns.

A row is four panel codes, so the space has ``PANEL_CODES ** 4`` members, but real charts use only a
small, heavily skewed subset. Treating each observed row as one token lets the selection model learn
jumps and holds as single decisions rather than four independent ones, which is how charters think
about them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
import collections
import json
import logging

from .cache import iter_cached, load_cached
from .chart import HOLD_HEAD, LIFT, ROLL_HEAD, TAP
from .dataset import CODE_BY_CHAR, PANEL_CODES

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    import numpy as np

__all__ = ('EMPTY_ROW', 'Vocabulary', 'build_vocabulary', 'decode_row', 'encode_row')

log = logging.getLogger(__name__)

_STEP_CODES = frozenset(CODE_BY_CHAR[character] for character in (TAP, HOLD_HEAD, ROLL_HEAD, LIFT))
_DEFAULT_LIMIT = 512
EMPTY_ROW = 0
"""
Token for a row with nothing on it.

:meta hide-value:
"""


def encode_row(codes: Sequence[int]) -> int:
    """
    Pack four panel codes into a single token value.

    Parameters
    ----------
    codes : :py:class:`~collections.abc.Sequence`
        Four panel codes.

    Returns
    -------
    int
        The packed value.
    """
    token = 0
    for code in codes:
        token = token * PANEL_CODES + int(code)
    return token


def decode_row(token: int) -> tuple[int, int, int, int]:
    """
    Unpack a token value into four panel codes.

    Parameters
    ----------
    token : int
        A packed row value.

    Returns
    -------
    tuple[int, int, int, int]
        The four panel codes.
    """
    codes: list[int] = []
    for _ in range(4):
        codes.append(token % PANEL_CODES)
        token //= PANEL_CODES
    return codes[3], codes[2], codes[1], codes[0]


class Vocabulary:
    """A frequency-ordered set of note-row patterns."""

    def __init__(self, patterns: Sequence[int]) -> None:
        self.patterns = tuple(patterns)
        """Packed row values, ordered by descending frequency."""
        self.index = {pattern: position for position, pattern in enumerate(self.patterns)}
        """Position of each pattern within :attr:`patterns`."""

    def __len__(self) -> int:
        """
        Return the number of patterns, including the unknown slot.

        Returns
        -------
        int
            The number of patterns, including the unknown slot.
        """
        return len(self.patterns)

    @classmethod
    def load(cls, path: Path) -> Vocabulary:
        """
        Read a vocabulary from disk.

        Parameters
        ----------
        path : :py:class:`~pathlib.Path`
            The file to read.

        Returns
        -------
        Vocabulary
            The stored vocabulary.
        """
        return cls(json.loads(path.read_text(encoding='utf-8')))

    def panels_of(self, position: int) -> tuple[int, int, int, int]:
        """
        Return the panel codes of a vocabulary entry.

        Parameters
        ----------
        position : int
            Index within :attr:`patterns`.

        Returns
        -------
        tuple[int, int, int, int]
            The four panel codes.
        """
        return decode_row(self.patterns[position])

    def save(self, path: Path) -> None:
        """
        Write the vocabulary to disk.

        Parameters
        ----------
        path : :py:class:`~pathlib.Path`
            Destination file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(list(self.patterns)), encoding='utf-8')

    def stepped_panels(self, position: int) -> frozenset[int]:
        """
        Return which panels an entry requires the player to step on.

        Parameters
        ----------
        position : int
            Index within :attr:`patterns`.

        Returns
        -------
        frozenset[int]
            Panel indices that carry a tap, hold, roll, or lift.
        """
        return frozenset(
            panel for panel, code in enumerate(self.panels_of(position)) if code in _STEP_CODES
        )

    def token_for(self, codes: Sequence[int]) -> int:
        """
        Return the vocabulary index for a row, falling back to the commonest.

        Parameters
        ----------
        codes : :py:class:`~collections.abc.Sequence`
            Four panel codes.

        Returns
        -------
        int
            Index within :attr:`patterns`.
        """
        return self.index.get(encode_row(codes), 0)


def build_vocabulary(cache_root: Path, *, limit: int = _DEFAULT_LIMIT) -> Vocabulary:
    """
    Count note-row patterns across the cache and keep the commonest.

    Parameters
    ----------
    cache_root : :py:class:`~pathlib.Path`
        Cache directory to scan.
    limit : int
        Largest vocabulary to build.

    Returns
    -------
    Vocabulary
        The frequency-ordered vocabulary.
    """
    counter: collections.Counter[int] = collections.Counter()
    for path in iter_cached(cache_root):
        if (song := load_cached(path)) is None:
            continue
        for chart in song.charts:
            counter.update(int(encode_row(row)) for row in cast('np.ndarray', chart['panels']))
    # The empty row is the commonest pattern in any chart and the one the selection head must never
    # choose: where the rests go is the placement head's decision, and a vocabulary entry for
    # silence lets selection overrule it. Left in, it was taken at 28 per cent of the slots
    # placement had picked for a note, scattering holes through the chart.
    del counter[EMPTY_ROW]
    log.info('Found %d distinct note-row patterns, excluding the empty row.', len(counter))
    return Vocabulary([pattern for pattern, _ in counter.most_common(limit)])


def coverage(counts: Iterable[tuple[int, int]], limit: int) -> float:
    """
    Return the share of rows a truncated vocabulary would represent.

    Parameters
    ----------
    counts : :py:class:`~collections.abc.Iterable`
        Pattern and count pairs, ordered by descending count.
    limit : int
        Vocabulary size to evaluate.

    Returns
    -------
    float
        Fraction of all rows covered, between zero and one.
    """
    values = [count for _, count in counts]
    total = sum(values)
    return sum(values[:limit]) / total if total else 0.0
