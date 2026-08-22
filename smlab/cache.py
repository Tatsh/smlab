"""Building and reading the beat-grid feature cache."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import hashlib
import json
import logging

import numpy as np

from .audio import load_audio
from .chart import DIFFICULTIES
from .dataset import beat_features, chart_targets
from .simfile import SimfileError, load_simfile

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

    from .typing import SongRecord

__all__ = (
    'CachedSong',
    'cache_path_for',
    'iter_cached',
    'load_cached',
    'write_song_cache',
)

log = logging.getLogger(__name__)

_MIN_SLOTS = 64
_MIN_ROWS = 16


class CachedSong:
    """Features and chart targets for one cached song."""

    def __init__(
        self, features: NDArray[np.float16], charts: list[dict[str, object]], title: str
    ) -> None:
        self.charts = charts
        """Per-chart dictionaries holding slots, panels, difficulty, and meter."""
        self.features = features
        """Beat-grid features shaped ``(slots, FEATURE_DIMENSION)``."""
        self.title = title
        """The song title, kept for logging."""

    def __len__(self) -> int:
        """
        Return the number of grid slots.

        Returns
        -------
        int
            The number of grid slots.
        """
        return int(self.features.shape[0])


def cache_path_for(root: Path, simfile: str) -> Path:
    """
    Return the cache file belonging to a simfile.

    Parameters
    ----------
    root : :py:class:`~pathlib.Path`
        Cache directory.
    simfile : str
        Absolute path of the source simfile.

    Returns
    -------
    :py:class:`~pathlib.Path`
        Location of the cache entry.
    """
    digest = hashlib.sha1(simfile.encode(), usedforsecurity=False).hexdigest()
    return root / digest[:2] / f'{digest}.npz'


def _song_arrays(record: SongRecord) -> dict[str, NDArray[np.generic]] | None:
    """
    Build every array that a song's cache entry holds.

    Parameters
    ----------
    record : SongRecord
        The manifest record to process.

    Returns
    -------
    dict[str, :py:class:`~numpy.ndarray`] | None
        Arrays keyed by cache entry name, or ``None`` when the song is unusable.
    """
    try:
        simfile = load_simfile(Path(record['simfile']))
        if (timing := simfile.timing) is None:
            return None
        features = beat_features(load_audio(Path(record['audio'])), timing)
    except (SimfileError, OSError, ValueError, RuntimeError) as error:
        log.debug('Could not build features for `%s`: %s', record['simfile'], error)
        return None
    if features.shape[0] < _MIN_SLOTS:
        return None
    arrays: dict[str, NDArray[np.generic]] = {'features': features}
    meta: list[dict[str, object]] = []
    for index, chart in enumerate(simfile.singles()):
        targets = chart_targets(chart, features.shape[0])
        if chart.difficulty not in DIFFICULTIES or len(targets) < _MIN_ROWS:
            continue
        arrays[f'slots_{index}'] = targets.slots
        arrays[f'panels_{index}'] = targets.panels
        meta.append({'difficulty': chart.difficulty, 'index': index, 'meter': chart.meter})
    if not meta:
        return None
    arrays['meta'] = np.asarray(json.dumps(meta, sort_keys=True))
    return arrays


def write_song_cache(item: tuple[SongRecord, str]) -> str | None:
    """
    Build and write the cache entry for one song.

    Parameters
    ----------
    item : tuple[SongRecord, str]
        The manifest record and the cache root, passed as one argument so that the function can be
        mapped across a process pool.

    Returns
    -------
    str | None
        The simfile path on success, or ``None`` when the song was unusable.
    """
    record, root = item
    destination = cache_path_for(Path(root), record['simfile'])
    if destination.exists():
        return record['simfile']
    if (arrays := _song_arrays(record)) is None:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    # numpy's stub declares keyword parameters of its own alongside the array keywords, so an
    # unpacked mapping of arrays cannot be expressed.
    np.savez_compressed(destination, **arrays)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    return record['simfile']


def load_cached(path: Path) -> CachedSong | None:
    """
    Read one cache entry.

    Parameters
    ----------
    path : :py:class:`~pathlib.Path`
        The cache file to read.

    Returns
    -------
    CachedSong | None
        The cached song, or ``None`` when the file is unreadable.
    """
    try:
        with np.load(path, allow_pickle=False) as data:
            features = data['features']
            meta = json.loads(str(data['meta']))
            charts = [
                {
                    'difficulty': entry['difficulty'],
                    'meter': entry['meter'],
                    'panels': data[f'panels_{entry["index"]}'],
                    'slots': data[f'slots_{entry["index"]}'],
                }
                for entry in meta
            ]
    except (OSError, KeyError, ValueError) as error:
        log.debug('Could not read cache entry `%s`: %s', path, error)
        return None
    return CachedSong(features, charts, path.stem)


def iter_cached(root: Path) -> Iterator[Path]:
    """
    Yield every cache entry beneath a cache directory.

    Parameters
    ----------
    root : :py:class:`~pathlib.Path`
        Cache directory.

    Yields
    ------
    :py:class:`~pathlib.Path`
        Each cache file, in sorted order.
    """
    yield from sorted(root.glob('*/*.npz'))
