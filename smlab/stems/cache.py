"""
Building the stem-based feature cache.

Separation runs on the GPU and feature extraction runs on the processor, so the
two are pipelined: while one song is being turned into mel bands on worker
threads, the next is already being separated. Doing them in sequence would
leave whichever device is idle waiting for the other.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING
import json
import logging

import librosa
import numpy as np
import torch

from smlab.audio import DEFAULT_SAMPLE_RATE
from smlab.cache import cache_path_for
from smlab.chart import DIFFICULTIES
from smlab.dataset import chart_targets
from smlab.features import TOTAL_CHANNELS, fine_features
from smlab.simfile import SimfileError, load_simfile

from .separate import SeparationError, load_separator, separate

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from numpy.typing import NDArray

    from smlab.typing import SongRecord

    from .separate import Separator

__all__ = ('build_stem_cache', 'cache_channels', 'stem_cache_entry')

log = logging.getLogger(__name__)

_MIN_SLOTS = 128
_MIN_ROWS = 16
_FEATURE_WORKERS = 4


def _resampled_stems(
    model: Separator, path: Path, device: torch.device
) -> tuple[dict[str, NDArray[np.float32]], NDArray[np.float32]]:
    """
    Separate a song and bring every layer to the analysis sample rate.

    Parameters
    ----------
    model : Separator
        The separation model.
    path : :py:class:`~pathlib.Path`
        Audio file to process.
    device : :py:class:`~torch.device`
        Device separation runs on.

    Returns
    -------
    tuple[dict[str, :py:class:`~numpy.ndarray`], :py:class:`~numpy.ndarray`]
        The stems and the mixture, all mono at the analysis rate.
    """
    stems = separate(model, path, device)
    rate = model.samplerate
    resampled = {
        name: librosa.resample(samples, orig_sr=rate, target_sr=DEFAULT_SAMPLE_RATE)
        for name, samples in stems.items()
    }
    mixture, _ = librosa.load(str(path), sr=DEFAULT_SAMPLE_RATE, mono=True)
    return resampled, np.asarray(mixture, dtype=np.float32)


def stem_cache_entry(
    record: SongRecord, model: Separator, device: torch.device
) -> dict[str, NDArray[np.generic]] | None:
    """
    Build every array one song's cache entry holds.

    Parameters
    ----------
    record : SongRecord
        Manifest record for the song.
    model : Separator
        The separation model.
    device : :py:class:`~torch.device`
        Device separation runs on.

    Returns
    -------
    dict[str, :py:class:`~numpy.ndarray`] | None
        Arrays keyed by cache entry name, or ``None`` when the song is unusable.
    """
    try:
        simfile = load_simfile(Path(record['simfile']))
        if (timing := simfile.timing) is None:
            return None
        stems, mixture = _resampled_stems(model, Path(record['audio']), device)
        features = fine_features(stems, mixture, timing)
    except (SimfileError, SeparationError, OSError, ValueError, RuntimeError) as error:
        log.debug('Skipping `%s`: %s', record['simfile'], error)
        return None
    if features.shape[0] < _MIN_SLOTS:
        return None
    arrays: dict[str, NDArray[np.generic]] = {'features': features}
    meta: list[dict[str, object]] = []
    # Note targets stay on the note grid, which is half the feature grid.
    note_slots = features.shape[0] // 2
    for index, chart in enumerate(simfile.singles()):
        targets = chart_targets(chart, note_slots)
        if chart.difficulty not in DIFFICULTIES or len(targets) < _MIN_ROWS:
            continue
        arrays[f'slots_{index}'] = targets.slots
        arrays[f'panels_{index}'] = targets.panels
        meta.append({'difficulty': chart.difficulty, 'index': index, 'meter': chart.meter})
    if not meta:
        return None
    arrays['meta'] = np.asarray(json.dumps(meta, sort_keys=True))
    return arrays


def build_stem_cache(
    records: Iterable[SongRecord], root: Path, *, device: torch.device | None = None
) -> Iterator[tuple[str, bool]]:
    """
    Build stem-based cache entries for a corpus.

    Parameters
    ----------
    records : :py:class:`~collections.abc.Iterable`
        Manifest records to process.
    root : :py:class:`~pathlib.Path`
        Cache directory to fill.
    device : :py:class:`~torch.device` | None
        Device separation runs on, or ``None`` to pick automatically.

    Yields
    ------
    tuple[str, bool]
        The simfile path and whether an entry was written for it.
    """
    root.mkdir(parents=True, exist_ok=True)
    chosen = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_separator(chosen)
    log.info('Separating into %s on %s.', ', '.join(model.sources), chosen)

    def write(record: SongRecord) -> tuple[str, bool]:
        destination = cache_path_for(root, record['simfile'])
        if destination.exists():
            return record['simfile'], True
        if (arrays := stem_cache_entry(record, model, chosen)) is None:
            return record['simfile'], False
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Stored uncompressed deliberately. numpy silently ignores mmap_mode on
        # a compressed archive, so every read would decompress the whole feature
        # array in order to slice one window out of it. Uncompressed costs twice
        # the disk and reads a window six times faster.
        # numpy's stub declares keyword parameters alongside the array keywords,
        # so an unpacked mapping of arrays cannot be expressed.
        np.savez(destination, **arrays)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        return record['simfile'], True

    with ThreadPoolExecutor(max_workers=_FEATURE_WORKERS) as pool:
        yield from pool.map(write, records)


def cache_channels() -> int:
    """
    Return the feature width entries in this cache carry.

    Returns
    -------
    int
        Channels per fine slot.
    """
    return TOTAL_CHANNELS
