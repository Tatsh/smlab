"""
Training for the downbeat phase model.

Labels are free. A song with a declared constant tempo and a declared offset
tells us exactly where its downbeats fall, so every excerpt of it is a labelled
example and the label is whatever phase that excerpt starts at. Folding a
different excerpt, or folding from a different starting point, produces another
example of the same song at another phase, which is what makes 794 usable songs
enough to train on.

Only songs whose tempo is constant, whose offset is declared rather than
guessed, and which carry no stops are used. Anything else has a beat grid that
either moves or was never authored, and would teach the model the wrong answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
import hashlib
import logging

from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch

from .audio import OnsetParams, load_audio
from .offset import (
    BEATS_PER_MEASURE,
    EXCERPT_SECONDS,
    PHASE_BINS,
    OffsetModel,
    band_envelopes,
    fold_profile,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

__all__ = (
    'OffsetTrainingConfig',
    'build_envelope_cache',
    'cyclic_error',
    'train_offset_model',
    'usable',
)

log = logging.getLogger(__name__)

_VALIDATION_SHARE = 1
_HASH_BUCKETS = 8
_HALF_BEAT_BINS = PHASE_BINS // 8
"""Bins spanning half a beat, which is the classic wrong answer."""
_PROGRESS_EVERY = 100
_NEAR_BINS = 2
_MIN_SECONDS = 30.0
"""Shortest song worth folding, since a fold needs several bars to mean much."""


@dataclass(frozen=True, slots=True)
class OffsetTrainingConfig:
    """Hyperparameters for the phase model."""

    batch_size: int = 64
    """Profiles per optimiser step."""
    epochs: int = 60
    """Passes over the song list."""
    excerpt_seconds: float = EXCERPT_SECONDS
    """
    How much audio each folded profile covers.

    Long enough that a fold averages over many bars, short enough that a tempo
    estimate a fraction of a beat per minute out has not yet drifted a whole
    bin across the excerpt.
    """
    learning_rate: float = 2e-3
    """Peak learning rate."""
    windows: int = 8
    """Excerpts drawn from each song per epoch."""


def usable(record: dict[str, object]) -> bool:
    """
    Report whether a song can supply a trustworthy phase label.

    Parameters
    ----------
    record : dict[str, object]
        A manifest record.

    Returns
    -------
    bool
        Whether the song is usable.
    """
    return bool(
        record.get('constant_bpm')
        and record.get('offset_declared')
        and not record.get('stops')
        and record.get('primary_bpm')
    )


def build_envelope_cache(records: Sequence[dict[str, object]], destination: Path) -> int:
    """
    Compute and store band onset envelopes for every usable song.

    Parameters
    ----------
    records : :py:class:`~collections.abc.Sequence`
        Manifest records.
    destination : :py:class:`~pathlib.Path`
        Directory to fill.

    Returns
    -------
    int
        How many songs were cached.
    """
    destination.mkdir(parents=True, exist_ok=True)
    params = OnsetParams()
    written = 0
    keepers = [r for r in records if usable(r)]
    log.info('Computing band envelopes for %d songs.', len(keepers))
    for position, record in enumerate(keepers):
        if position and position % _PROGRESS_EVERY == 0:
            log.info('  %d of %d.', position, len(keepers))
        audio = Path(str(record['audio']))
        # A stable digest, not the built-in hash: that is salted per process,
        # so a resumed run would miss every file it had already written.
        digest = hashlib.sha1(str(audio).encode(), usedforsecurity=False).hexdigest()[:16]
        target = destination / f'{digest}.npz'
        if target.is_file():
            written += 1
            continue
        try:
            samples = load_audio(audio)
        except Exception:  # noqa: BLE001
            log.info('Could not read `%s`.', audio)
            continue
        if len(samples) < _MIN_SECONDS * params.sample_rate:
            continue
        np.savez(
            target,
            envelopes=band_envelopes(samples, params),
            bpm=np.float64(str(record['primary_bpm'])),
            offset=np.float64(str(record['offset'])),
            rate=np.float64(params.frame_rate),
        )
        written += 1
    return written


class FoldedProfiles(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Folded excerpts of cached songs, labelled with their downbeat phase."""

    def __init__(
        self, cache: Path, config: OffsetTrainingConfig, *, validation: bool = False
    ) -> None:
        self.config = config
        """Training settings, which fix how long an excerpt is."""
        self.paths: list[Path] = []
        """Cached songs this split may draw from."""
        self._rng: np.random.Generator | None = None
        for path in sorted(cache.glob('*.npz')):
            bucket = int(path.stem[:8], 16) % _HASH_BUCKETS
            if (bucket < _VALIDATION_SHARE) == validation:
                self.paths.append(path)
        self.validation = validation
        """Whether this split is held out, in which case excerpts are fixed."""
        log.info('Prepared %d songs (validation=%s).', len(self.paths), validation)

    @property
    def rng(self) -> np.random.Generator:
        """
        Source of excerpt positions, seeded per loader worker.

        Returns
        -------
        :py:class:`~numpy.random.Generator`
            This worker's generator.
        """
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            self._rng = np.random.default_rng(0 if info is None else info.seed % (2**32))
        return self._rng

    def __len__(self) -> int:
        """
        Return how many examples one pass draws.

        Returns
        -------
        int
            Songs times excerpts per song.
        """
        return len(self.paths) * (1 if self.validation else self.config.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fold one excerpt and label it.

        Parameters
        ----------
        index : int
            Example index.

        Returns
        -------
        tuple[:py:class:`~torch.Tensor`, :py:class:`~torch.Tensor`]
            The folded profile and the bar position of its downbeat.

        Raises
        ------
        IndexError
            If the split holds no songs.
        """
        if not self.paths:
            message = 'This split holds no songs; the cache may be empty or all in one bucket.'
            raise IndexError(message)
        path = self.paths[index % len(self.paths)]
        with np.load(path, allow_pickle=False) as data:
            envelopes = np.asarray(data['envelopes'], dtype=np.float32)
            bpm = float(data['bpm'])
            offset = float(data['offset'])
            rate = float(data['rate'])
        span = int(self.config.excerpt_seconds * rate)
        room = max(envelopes.shape[1] - span, 1)
        # A held-out song is always folded from the same place, so its score
        # does not wander between runs. A training song is folded from a fresh
        # place each time, which is where the extra examples come from.
        start = (index * 7919) % room if self.validation else int(self.rng.integers(room))
        excerpt = envelopes[:, start : start + span]
        seconds = start / rate
        profile = fold_profile(excerpt, rate, bpm, start=-seconds)
        # Folding an excerpt beginning `seconds` into the song is the same as
        # folding from zero, so the label is just where the downbeat sits in the
        # bar. The excerpt's own start does not move it.
        period = BEATS_PER_MEASURE * 60.0 / bpm
        phase = int(np.floor(((-offset) % period) / period * PHASE_BINS)) % PHASE_BINS
        return torch.from_numpy(profile), torch.tensor(phase, dtype=torch.long)


def cyclic_error(predicted: NDArray[np.int64], actual: NDArray[np.int64]) -> NDArray[np.int64]:
    """
    Return how many bins apart two phases are, the short way round.

    Parameters
    ----------
    predicted : :py:class:`~numpy.ndarray`
        Chosen bins.
    actual : :py:class:`~numpy.ndarray`
        True bins.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        Distance in bins, never more than half a bar.
    """
    raw = np.abs(predicted - actual)
    return np.asarray(np.minimum(raw, PHASE_BINS - raw), dtype=np.int64)


def train_offset_model(
    cache: Path, output: Path, config: OffsetTrainingConfig | None = None
) -> dict[str, float]:
    """
    Train the phase model and keep the best weights.

    Parameters
    ----------
    cache : :py:class:`~pathlib.Path`
        Envelope cache directory.
    output : :py:class:`~pathlib.Path`
        Where to write the checkpoint.
    config : OffsetTrainingConfig | None
        Hyperparameters, or ``None`` for the defaults.

    Returns
    -------
    dict[str, float]
        The best validation measurements reached.
    """
    settings = config if config is not None else OffsetTrainingConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_set = FoldedProfiles(cache, settings)
    valid_set = FoldedProfiles(cache, settings, validation=True)
    loaders = {
        'train': DataLoader(train_set, batch_size=settings.batch_size, shuffle=True, num_workers=4),
        'valid': DataLoader(valid_set, batch_size=settings.batch_size, num_workers=2),
    }
    model = OffsetModel().to(device)
    log.info('Model has %.2f M parameters.', sum(p.numel() for p in model.parameters()) / 1e6)
    optimiser = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, settings.epochs)
    best: dict[str, float] = {'within_one_bin': 0.0}
    for epoch in range(settings.epochs):
        model.train()
        total = 0.0
        for profile, phase in loaders['train']:
            optimiser.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(
                model(profile.to(device)), phase.to(device), label_smoothing=0.05
            )
            loss.backward()  # type: ignore[no-untyped-call]
            optimiser.step()
            total += float(loss)
        schedule.step()
        model.eval()
        chosen: list[int] = []
        truth: list[int] = []
        with torch.no_grad():
            for profile, phase in loaders['valid']:
                chosen.extend(model(profile.to(device)).argmax(-1).cpu().tolist())
                truth.extend(phase.tolist())
        gap = cyclic_error(np.asarray(chosen), np.asarray(truth))
        measured = {
            'exact': float(np.mean(gap == 0)),
            'within_one_bin': float(np.mean(gap <= 1)),
            'within_two_bins': float(np.mean(gap <= _NEAR_BINS)),
            'half_beat_out': float(np.mean(np.abs(gap - _HALF_BEAT_BINS) <= 1)),
        }
        log.info(
            'Epoch %d: train %.4f, exact %.3f, within one bin %.3f, within two %.3f.',
            epoch + 1,
            total / max(len(loaders['train']), 1),
            measured['exact'],
            measured['within_one_bin'],
            measured['within_two_bins'],
        )
        if measured['within_one_bin'] > best['within_one_bin']:
            best = measured
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'model': model.state_dict()}, output)
    return best
