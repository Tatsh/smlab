"""
Training examples drawn from the stem feature cache.

Entries are read from disk per window rather than held in memory, because the
cache runs to fourteen gigabytes. The cache is written uncompressed for this
reason: reading one window out of a compressed archive costs 20 ms because the
whole feature array has to be inflated first, against 1.5 ms uncompressed.
Passing ``mmap_mode`` does not rescue the compressed case, since numpy ignores
it there rather than reporting that it cannot comply.

The rating scale a chart uses is not recorded in the simfile, so it is inferred
from its pack: a pack whose charts never exceed ten is using the classic scale,
and one that reaches thirteen or more is using the twenty-point scale. Without
this the same number means two different difficulties and the conditioning is
ambiguous by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
import hashlib
import json
import logging

from torch.utils.data import Dataset
import numpy as np
import torch

from .cache import cache_path_for
from .dataset import SUBDIVISIONS_PER_BEAT, difficulty_index
from .encoder import MAX_METER, MAX_RATE, MEASURE_SLOTS, STYLES
from .features import TOTAL_CHANNELS
from .heads import MAX_DELTA, metric_prior_logits

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from .typing import SongRecord
    from .vocab import Vocabulary

__all__ = (
    'MIRRORS',
    'WINDOW_MEASURES',
    'WINDOW_STEPS',
    'ChartExample',
    'ChartWindows',
    'measure_prior',
    'pack_scales',
)

log = logging.getLogger(__name__)

WINDOW_MEASURES = 64
"""Measures per training example.

:meta hide-value:
"""
WINDOW_SLOTS = WINDOW_MEASURES * MEASURE_SLOTS
"""Note-grid slots per training example.

:meta hide-value:
"""
WINDOW_STEPS = 384
"""Largest number of steps a window's selection target holds.

:meta hide-value:
"""
_CLASSIC_MAX = 10
_MODERN_MIN = 13
_VALIDATION_SHARE = 2
_HASH_BUCKETS = 20
_DECIBEL_SCALE = 40.0


def _style_index(name: str) -> int:
    """
    Return the conditioning index for a performance style.

    Parameters
    ----------
    name : str
        Style name recorded in the manifest.

    Returns
    -------
    int
        Position within :data:`~smlab.encoder.STYLES`, defaulting to feet.
    """
    return STYLES.index(name) if name in STYLES else 0


def pack_scales(records: list[SongRecord]) -> dict[str, int]:
    """
    Infer which rating scale each pack uses.

    Parameters
    ----------
    records : list[SongRecord]
        Manifest records covering the corpus.

    Returns
    -------
    dict[str, int]
        Scale index per pack name, zero for the classic ten-point scale.
    """
    highest: dict[str, int] = {}
    for record in records:
        for chart in record['charts']:
            highest[record['pack']] = max(highest.get(record['pack'], 0), chart['meter'])
    return {
        pack: (1 if top >= _MODERN_MIN else 0)
        for pack, top in highest.items()
        if top <= _CLASSIC_MAX or top >= _MODERN_MIN
    }


_MIN_RATE_NOTES = 2
MIRRORS = (
    (0, 1, 2, 3),
    (3, 1, 2, 0),
    (0, 2, 1, 3),
    (3, 2, 1, 0),
)
"""
The four reflections of a dance pad, as panel permutations.

Left and right may be swapped, up and down may be swapped, and either may be
done independently: all four leave a chart exactly as playable as it started,
because they are symmetries of the pad itself.

These are the Klein four-group, and it is worth being clear about what they
cannot do. Every one of them maps the outer pair onto the outer pair and the
middle pair onto the middle pair, so no amount of training or averaging over
them can move weight between those pairs. The bias the model actually has is
exactly that split: about a third of its notes on each of down and up against a
sixth on each of left and right. Two retrains and an inference-time average
over these reflections all failed to shift it, for that reason.

Reaching it would need a quarter turn, and a quarter turn is not a symmetry of
*play*: left and right are naturally one foot each while up and down are
shared, so turning a left-right alternation into an up-down one changes the
difficulty. The bias is corrected at decode time instead, by
:py:attr:`~smlab.generate.GenerationConfig.balance`.

:meta hide-value:
"""


@dataclass(frozen=True, slots=True)
class ChartExample:
    """One chart within one cached song."""

    bpm: float
    """Tempo, used to turn a note count into a rate."""
    difficulty: int
    """Conditioning index of the difficulty."""
    meter: int
    """Rating, clamped to the embedding range."""
    path: Path
    """Cache entry the features live in."""
    scale: int
    """Rating scale index, zero for the classic ten-point scale."""
    style: int
    """Performance style index, zero for a chart danceable with two feet."""


def _note_rate(slots: NDArray[np.int64], bpm: float) -> int:
    """
    Bucket a chart's note rate, in whole notes per second.

    Parameters
    ----------
    slots : :py:class:`~numpy.ndarray`
        Grid slot of every note in the chart.
    bpm : float
        Tempo of the song.

    Returns
    -------
    int
        Bucket index, clamped to the conditioner's range.
    """
    if len(slots) < _MIN_RATE_NOTES or bpm <= 0:
        return 0
    seconds = (int(slots[-1]) - int(slots[0])) * 60.0 / bpm / SUBDIVISIONS_PER_BEAT
    if seconds <= 0:
        return 0
    return min(int(len(slots) / seconds), MAX_RATE - 1)


def measure_prior(examples: list[ChartExample]) -> NDArray[np.float32]:
    """
    Compute the log-odds of a step at each position in the bar.

    This becomes a fixed bias on the placement head, so the network cannot earn
    accuracy by rediscovering that notes fall on beats.

    Parameters
    ----------
    examples : list[ChartExample]
        Charts to count over, normally the training split.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        Log-odds per position, shaped ``(MEASURE_SLOTS,)``.
    """
    counts = np.zeros(MEASURE_SLOTS, dtype=np.float64)
    totals = np.zeros(MEASURE_SLOTS, dtype=np.float64)
    for example in examples:
        with np.load(example.path, allow_pickle=False, mmap_mode='r') as data:
            meta = json.loads(str(data['meta']))
            slots = data['features'].shape[0] // 2
            totals += np.bincount(np.arange(slots) % MEASURE_SLOTS, minlength=MEASURE_SLOTS)
            for entry in meta:
                if difficulty_index(entry['difficulty']) != example.difficulty:
                    continue
                rows = np.asarray(data[f'slots_{entry["index"]}'])
                counts += np.bincount(rows % MEASURE_SLOTS, minlength=MEASURE_SLOTS)
    return metric_prior_logits(counts, totals)


class ChartWindows(Dataset[dict[str, torch.Tensor]]):
    """Random windows of a chart, with placement and selection targets."""

    def __init__(
        self,
        cache_root: Path,
        records: list[SongRecord],
        vocabulary: Vocabulary,
        *,
        validation: bool = False,
        limit: int = 0,
    ) -> None:
        self.validation = validation
        """Whether this split is held out, in which case nothing is mirrored."""
        self.vocabulary = vocabulary
        """Pattern vocabulary used to tokenise note rows."""
        self.examples: list[ChartExample] = []
        """Every chart this dataset can draw a window from."""
        self._rng: np.random.Generator | None = None
        scales = pack_scales(records)
        for record in records:
            path = cache_path_for(cache_root, record['simfile'])
            if not path.is_file() or record['pack'] not in scales:
                continue
            bucket = int(hashlib.sha1(path.stem.encode(), usedforsecurity=False).hexdigest(), 16)
            if (bucket % _HASH_BUCKETS < _VALIDATION_SHARE) != validation:
                continue
            for chart in record['charts']:
                self.examples.append(
                    ChartExample(
                        bpm=float(record['primary_bpm']),
                        difficulty=difficulty_index(chart['difficulty']),
                        meter=min(max(chart['meter'], 0), MAX_METER - 1),
                        path=path,
                        scale=scales[record['pack']],
                        style=_style_index(chart.get('style', 'feet')),
                    )
                )
            if limit and len(self.examples) >= limit:
                break
        log.info('Prepared %d chart windows (validation=%s).', len(self.examples), validation)

    @property
    def rng(self) -> np.random.Generator:
        """
        Source of window starts and mirrors, seeded per loader worker.

        A generator made in ``__init__`` is copied into every worker process
        along with its state, so all of them draw the same window starts and
        the same mirrors in lockstep. That is not a small waste: with six
        workers the model sees a sixth of the window variety it should, and the
        run cannot be reproduced from the seed alone. PyTorch gives each worker
        a distinct seed per epoch, which is what this defers to.

        Returns
        -------
        :py:class:`~numpy.random.Generator`
            This worker's generator.
        """
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            self._rng = np.random.default_rng(0 if info is None else info.seed % (2**32))
        return self._rng

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # noqa: PLR0914
        """
        Return one training window.

        Parameters
        ----------
        index : int
            Chart index.

        Returns
        -------
        dict[str, :py:class:`~torch.Tensor`]
            Features, conditioning, and both heads' targets.
        """
        example = self.examples[index]
        with np.load(example.path, allow_pickle=False, mmap_mode='r') as data:
            meta = json.loads(str(data['meta']))
            chosen = next(
                (
                    entry
                    for entry in meta
                    if difficulty_index(entry['difficulty']) == example.difficulty
                ),
                None,
            )
            fine = data['features']
            note_slots = fine.shape[0] // 2
            start = int(self.rng.integers(0, max(note_slots - WINDOW_SLOTS, 1)))
            start -= start % MEASURE_SLOTS
            window = np.zeros((2 * WINDOW_SLOTS, TOTAL_CHANNELS), dtype=np.float32)
            available = np.asarray(fine[2 * start : 2 * (start + WINDOW_SLOTS)], dtype=np.float32)
            window[: available.shape[0]] = available / _DECIBEL_SCALE
            slots = (
                np.asarray(data[f'slots_{chosen["index"]}'], dtype=np.int64)
                if chosen
                else np.zeros(0, dtype=np.int64)
            )
            panels = (
                np.asarray(data[f'panels_{chosen["index"]}'], dtype=np.uint8)
                if chosen
                else np.zeros((0, 4), dtype=np.uint8)
            )
        rate = _note_rate(slots, example.bpm)
        # Every window is shown under one of the pad's four reflections, chosen
        # at random. A held-out window is never mirrored, so validation figures
        # stay comparable across runs.
        mirror = MIRRORS[0 if self.validation else int(self.rng.integers(len(MIRRORS)))]
        inside = (slots >= start) & (slots < start + WINDOW_SLOTS)
        local = slots[inside] - start
        rows = panels[inside][:, mirror]
        placement = np.zeros(WINDOW_SLOTS, dtype=np.float32)
        placement[local] = 1.0
        count = min(len(local), WINDOW_STEPS)
        tokens = np.array([self.vocabulary.token_for(row) for row in rows[:count]], dtype=np.int64)
        previous = np.full(WINDOW_STEPS, len(self.vocabulary), dtype=np.int64)
        previous[1:count] = tokens[: count - 1]
        target = np.full(WINDOW_STEPS, -100, dtype=np.int64)
        target[:count] = tokens
        step_slots = np.zeros(WINDOW_STEPS, dtype=np.int64)
        step_slots[:count] = local[:count]
        delta = np.zeros(WINDOW_STEPS, dtype=np.int64)
        if count > 1:
            delta[1:count] = np.clip(np.diff(local[:count]), 0, MAX_DELTA)
        return {
            'delta': torch.from_numpy(delta),
            'difficulty': torch.tensor(example.difficulty, dtype=torch.long),
            'features': torch.from_numpy(window),
            'meter': torch.tensor(example.meter, dtype=torch.long),
            'pattern_target': torch.from_numpy(target),
            'placement': torch.from_numpy(placement),
            'position': torch.from_numpy(
                (np.arange(WINDOW_SLOTS, dtype=np.int64) + start) % MEASURE_SLOTS
            ),
            'previous': torch.from_numpy(previous),
            'rate': torch.tensor(rate, dtype=torch.long),
            'scale': torch.tensor(example.scale, dtype=torch.long),
            'style': torch.tensor(example.style, dtype=torch.long),
            'step_position': torch.from_numpy((step_slots + start) % MEASURE_SLOTS),
            'step_slots': torch.from_numpy(step_slots),
        }

    def __len__(self) -> int:
        """
        Return the number of charts.

        Returns
        -------
        int
            The number of charts.
        """
        return len(self.examples)
