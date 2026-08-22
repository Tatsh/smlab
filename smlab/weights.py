"""
Fetching trained weights that are too large to ship inside the wheel.

The chart model is 152 MB as float32, which does not belong in a package, so it is downloaded once
and cached. Where it is downloaded *from* is deliberately configurable: the weights are derived from
a personal simfile library, so whoever trains a set decides where, or whether, to publish it.

Resolution order is local then remote. A directory of locally trained checkpoints always wins, so
retraining never has to fight the cache. Small models that still ship inside the wheel are read
through :py:mod:`smlab.resources` instead; this module covers only the ones too large to bundle.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import logging
import os

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = (
    'CHART_WEIGHTS',
    'DEFAULT_REPOSITORY',
    'OFFSET_WEIGHTS',
    'REPOSITORY_VARIABLE',
    'REVISION_VARIABLE',
    'WeightsError',
    'resolve_weights',
    'weights_repository',
    'weights_revision',
)

log = logging.getLogger(__name__)

CHART_WEIGHTS = 'chart.pt'
"""File name of the chart model checkpoint.

:meta hide-value:
"""
OFFSET_WEIGHTS = 'offset.pt'
"""
File the downbeat phase model is published under.

:meta hide-value:
"""
DEFAULT_REPOSITORY = 'tatsh/smlab'
"""
Hugging Face repository consulted when nothing else supplies the weights.

Override it with :data:`REPOSITORY_VARIABLE` rather than editing this, so a different corpus can
publish its own without patching the package.

:meta hide-value:
"""
REPOSITORY_VARIABLE = 'SMLAB_WEIGHTS_REPO'
"""Environment variable naming the repository to download from.

:meta hide-value:
"""
REVISION_VARIABLE = 'SMLAB_WEIGHTS_REVISION'
"""
Environment variable pinning a revision.

Pinning matters because retraining changes the weights while the file name stays the same; a tag is
what makes a result reproducible.

:meta hide-value:
"""
CACHE_VARIABLE = 'SMLAB_WEIGHTS_DIR'
"""
Environment variable relocating the download cache.

Worth setting when the home filesystem is small, since the checkpoint is well
over a hundred megabytes.

:meta hide-value:
"""


class WeightsError(Exception):
    """Raised when trained weights cannot be found anywhere."""


def weights_repository() -> str:
    """
    Return the repository weights are downloaded from.

    Returns
    -------
    str
        Repository identifier, from the environment when set.
    """
    return os.environ.get(REPOSITORY_VARIABLE, DEFAULT_REPOSITORY)


def weights_revision() -> str | None:
    """
    Return the pinned revision, if any.

    Returns
    -------
    str | None
        Revision from the environment, or ``None`` for the default branch.
    """
    return os.environ.get(REVISION_VARIABLE) or None


def _local_candidates(name: str, override: Path | None) -> Iterator[Path]:
    """
    Yield local paths that may hold the weights.

    Parameters
    ----------
    name : str
        Checkpoint file name.
    override : :py:class:`~pathlib.Path` | None
        Directory given on the command line, if any.

    Yields
    ------
    :py:class:`~pathlib.Path`
        Candidate paths, most specific first.
    """
    if override is not None:
        yield override / name
        yield override
    yield Path('checkpoints') / name


def _download(name: str) -> Path:
    """
    Fetch the weights from the configured repository.

    Parameters
    ----------
    name : str
        Checkpoint file name.

    Returns
    -------
    :py:class:`~pathlib.Path`
        Path to the cached download.

    Raises
    ------
    WeightsError
        If the hub client is missing or the download fails.
    """
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
    except ImportError as error:
        msg = 'huggingface-hub is needed to download weights; install it or pass --checkpoints.'
        raise WeightsError(msg) from error
    repository = weights_repository()
    log.info('Fetching `%s` from `%s`.', name, repository)
    try:
        return Path(
            hf_hub_download(
                repo_id=repository,
                filename=name,
                revision=weights_revision(),
                cache_dir=os.environ.get(CACHE_VARIABLE),
            )
        )
    except Exception as error:
        msg = (
            f'Could not download `{name}` from `{repository}`: {error}. '
            f'Set {REPOSITORY_VARIABLE} to another repository, or pass --checkpoints '
            f'to use locally trained weights.'
        )
        raise WeightsError(msg) from error


def resolve_weights(
    name: str = CHART_WEIGHTS, override: Path | None = None, *, allow_download: bool = True
) -> Path:
    """
    Find trained weights, preferring local copies over a download.

    Parameters
    ----------
    name : str
        Checkpoint file name.
    override : :py:class:`~pathlib.Path` | None
        Directory of locally trained checkpoints, which wins over everything else.
    allow_download : bool
        Whether a missing checkpoint may be fetched from the configured repository.

    Returns
    -------
    :py:class:`~pathlib.Path`
        Path to a checkpoint that exists.

    Raises
    ------
    WeightsError
        If no local copy exists and downloading is refused or fails.
    """
    for candidate in _local_candidates(name, override):
        if candidate.is_file():
            log.debug('Using local weights at `%s`.', candidate)
            return candidate
    if not allow_download:
        msg = f'No local `{name}` and downloading was not permitted.'
        raise WeightsError(msg)
    return _download(name)
