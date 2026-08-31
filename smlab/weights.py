"""
Finding the trained weights that are too large to ship inside the wheel.

The chart model is 152 MB as float32, which does not belong in a package, so it is looked for on the
machine first and fetched once if it is not there. The search covers the directories a distribution
installs data files into as well as the per-user cache, so a package that ships the weights
alongside the program never downloads anything.

Resolution is local before remote, and most specific before least. A directory of locally trained
checkpoints always wins, so retraining never has to fight the cache, and a user's own copy is
preferred over a system-wide one. Small models that still ship inside the wheel are read through
:py:mod:`smlab.resources` instead; this module covers only the ones too large to bundle.

Where the weights are downloaded *from* is deliberately configurable: they are derived from a
personal simfile library, so whoever trains a set decides where, or whether, to publish it. What is
downloaded is checked against digests bundled with the package, so a truncated transfer or a
substituted file is refused rather than loaded.
"""

from __future__ import annotations

from http.client import HTTPException
from pathlib import Path
from typing import TYPE_CHECKING
import hashlib
import logging
import os
import sys
import tempfile
import urllib.request

from platformdirs import PlatformDirs

from . import __version__
from .resources import load_checksums

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .typing import ProgressCallback

__all__ = (
    'CHART_WEIGHTS',
    'DEFAULT_RELEASE',
    'DEFAULT_REPOSITORY',
    'DIRECTORY_VARIABLE',
    'OFFSET_WEIGHTS',
    'RELEASE_VARIABLE',
    'REPOSITORY_VARIABLE',
    'URL_VARIABLE',
    'WeightsError',
    'download_directory',
    'file_digest',
    'resolve_weights',
    'weights_directories',
    'weights_release',
    'weights_repository',
    'weights_url',
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
DEFAULT_REPOSITORY = 'Tatsh/smlab'
"""
GitHub repository whose releases carry the weights.

Override it with :data:`REPOSITORY_VARIABLE` rather than editing this, so a different corpus can
publish its own without patching the package.

:meta hide-value:
"""
DEFAULT_RELEASE = f'v{__version__}'
"""
Release tag the weights are attached to.

Every release carries them, so this follows the version rather than being kept up to date by hand.
Pin it with :data:`RELEASE_VARIABLE` to fetch the weights of another release.

:meta hide-value:
"""
DIRECTORY_VARIABLE = 'SMLAB_WEIGHTS_DIR'
"""
Environment variable naming a directory to search before any of the standard ones.

This is what a system package sets when it installs the weights somewhere unusual.

:meta hide-value:
"""
REPOSITORY_VARIABLE = 'SMLAB_WEIGHTS_REPO'
"""Environment variable naming the ``owner/name`` repository to download from.

:meta hide-value:
"""
RELEASE_VARIABLE = 'SMLAB_WEIGHTS_RELEASE'
"""
Environment variable pinning the release tag.

Pinning matters because retraining changes the weights while the file name stays the same; a tag is
what makes a result reproducible.

:meta hide-value:
"""
URL_VARIABLE = 'SMLAB_WEIGHTS_URL'
"""
Environment variable giving a base URL to fetch from, for a mirror.

The file name is appended to it, so ``https://example.org/smlab`` becomes
``https://example.org/smlab/chart.pt``.

:meta hide-value:
"""

_CHUNK = 1 << 20
_PROGRESS_STEP = 1 << 23
_TIMEOUT = 30.0


class WeightsError(Exception):
    """Raised when trained weights cannot be found anywhere."""


def _directories(*, ensure_exists: bool = False) -> PlatformDirs:
    """
    Return the platform's directory layout for this program.

    Constructed on each call rather than once at import, so that a change to the environment is
    picked up.

    Parameters
    ----------
    ensure_exists : bool
        Whether reading a directory out of the result also creates it. Only ever set when asking
        for a directory to write into: the system-wide ones belong to a package manager, and
        creating them is not the program's business, nor usually permitted.

    Returns
    -------
    :py:class:`~platformdirs.PlatformDirs`
        Layout covering both the per-user and the system-wide directories.
    """
    return PlatformDirs(
        appname='smlab', appauthor='Tatsh', ensure_exists=ensure_exists, multipath=True
    )


def download_directory(*, ensure_exists: bool = False) -> Path:
    """
    Return the directory downloaded weights are written to.

    This is the user's data directory rather than the cache, mirroring where a system package
    installs them: they are expensive to fetch and are not rebuilt from anything on the machine, so
    a cache cleaner should leave them alone.

    Parameters
    ----------
    ensure_exists : bool
        Whether to create the directory.

    Returns
    -------
    :py:class:`~pathlib.Path`
        Directory under the user's data directory.
    """
    return _directories(ensure_exists=ensure_exists).user_data_path


def weights_repository() -> str:
    """
    Return the repository weights are downloaded from.

    Returns
    -------
    str
        Repository identifier, from the environment when set.
    """
    return os.environ.get(REPOSITORY_VARIABLE) or DEFAULT_REPOSITORY


def weights_release() -> str:
    """
    Return the release tag weights are downloaded from.

    Returns
    -------
    str
        Release tag, from the environment when set.
    """
    return os.environ.get(RELEASE_VARIABLE) or DEFAULT_RELEASE


def weights_url(name: str) -> str:
    """
    Return the address one checkpoint is downloaded from.

    Parameters
    ----------
    name : str
        Checkpoint file name.

    Returns
    -------
    str
        A release asset URL, or the file below the base URL when one is configured.
    """
    if base := os.environ.get(URL_VARIABLE):
        return f'{base.rstrip("/")}/{name}'
    return f'https://github.com/{weights_repository()}/releases/download/{weights_release()}/{name}'


def weights_directories(override: Path | None = None) -> tuple[Path, ...]:
    """
    Return every directory searched for weights, most specific first.

    The order is the command line, then the environment, then the working directory a training run
    writes to, then the user's own data and cache directories, then the directories a package
    manager installs into.

    Parameters
    ----------
    override : :py:class:`~pathlib.Path` | None
        Directory given on the command line, if any.

    Returns
    -------
    tuple[:py:class:`~pathlib.Path`, ...]
        Directories to search, without duplicates and in search order.
    """
    directories = _directories()
    candidates = [
        *([override] if override is not None else []),
        *([Path(named)] if (named := os.environ.get(DIRECTORY_VARIABLE)) else []),
        Path('checkpoints'),
        download_directory(),
        Path(sys.prefix, 'share', 'smlab'),
        *(Path(part) for part in directories.site_data_dir.split(os.pathsep)),
    ]
    return tuple(dict.fromkeys(candidates))


def file_digest(path: Path) -> str:
    """
    Return the SHA-256 of a file, read in chunks so that a large checkpoint fits in memory.

    Parameters
    ----------
    path : :py:class:`~pathlib.Path`
        File to read.

    Returns
    -------
    str
        Hexadecimal digest.
    """
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def _candidates(name: str, override: Path | None) -> Iterator[Path]:
    """
    Yield local paths that may hold the weights.

    Parameters
    ----------
    name : str
        Checkpoint file name.
    override : :py:class:`~pathlib.Path` | None
        Directory or file given on the command line, if any.

    Yields
    ------
    :py:class:`~pathlib.Path`
        Candidate paths, most specific first.
    """
    if override is not None and override.is_file():
        yield override
    for directory in weights_directories(override):
        yield directory / name


def _verify(path: Path, name: str, url: str) -> None:
    """
    Check a downloaded file against the digest bundled with the package.

    A checkpoint the package knows no digest for is accepted, since a mirror may carry files this
    release predates.

    Parameters
    ----------
    path : :py:class:`~pathlib.Path`
        The downloaded file.
    name : str
        Checkpoint file name, which is how the digest is looked up.
    url : str
        Where the file came from, for the error message.

    Raises
    ------
    WeightsError
        If the digest is known and does not match.
    """
    if (expected := load_checksums().get(name)) is None:
        log.debug('No bundled checksum for `%s`; accepting the download unverified.', name)
        return
    if (digest := file_digest(path)) != expected:
        msg = (
            f'`{name}` fetched from `{url}` is not the file this release expects: its SHA-256 is '
            f'{digest} rather than {expected}. The download was discarded.'
        )
        raise WeightsError(msg)


def _stream(url: str, handle: int, name: str, progress: ProgressCallback | None) -> None:
    """
    Copy what one address serves into an open file, reporting progress as it goes.

    Parameters
    ----------
    url : str
        Address to read.
    handle : int
        File descriptor to write to, which is closed on the way out.
    name : str
        Checkpoint file name, which is what progress is reported against.
    progress : ProgressCallback | None
        Called every few megabytes, and once more when the transfer ends.
    """
    with (
        os.fdopen(handle, 'wb') as file,
        urllib.request.urlopen(url, timeout=_TIMEOUT) as response,  # noqa: S310
    ):
        total = int(response.headers.get('Content-Length') or 0)
        received = reported = 0
        while chunk := response.read(_CHUNK):
            file.write(chunk)
            received += len(chunk)
            if progress is not None and received - reported >= _PROGRESS_STEP:
                reported = received
                progress(name, received, total)
        if progress is not None:
            progress(name, received, total or received)


def _download(name: str, progress: ProgressCallback | None = None) -> Path:
    """
    Fetch one checkpoint into the directory downloads are kept in.

    The file is written under a temporary name and moved into place only once it is complete and
    verified, so an interrupted download never leaves something loadable behind.

    Parameters
    ----------
    name : str
        Checkpoint file name.
    progress : ProgressCallback | None
        Called with the file name, the bytes received, and the total expected.

    Returns
    -------
    :py:class:`~pathlib.Path`
        Path to the cached download.

    Raises
    ------
    WeightsError
        If the address is not one that may be fetched, or the download fails or does not verify.
    """
    url = weights_url(name)
    if not url.startswith('https://'):
        msg = f'Refusing to fetch `{name}` over an insecure address: {url}.'
        raise WeightsError(msg)
    destination = download_directory(ensure_exists=True) / name
    log.info('Fetching `%s` from `%s`.', name, url)
    handle, temporary_name = tempfile.mkstemp(dir=destination.parent, suffix='.part')
    temporary = Path(temporary_name)
    try:
        _stream(url, handle, name, progress)
        _verify(temporary, name, url)
    except (HTTPException, OSError, ValueError) as error:
        temporary.unlink(missing_ok=True)
        msg = (
            f'Could not fetch `{name}` from `{url}`: {error}. Set {URL_VARIABLE} to a mirror, or '
            f'pass --checkpoints to use locally trained weights.'
        )
        raise WeightsError(msg) from error
    except WeightsError:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(destination)
    log.info('Cached `%s` at `%s`.', name, destination)
    return destination


def resolve_weights(
    name: str = CHART_WEIGHTS,
    override: Path | None = None,
    *,
    allow_download: bool = True,
    progress: ProgressCallback | None = None,
) -> Path:
    """
    Find trained weights, preferring a copy already on the machine over a download.

    Parameters
    ----------
    name : str
        Checkpoint file name.
    override : :py:class:`~pathlib.Path` | None
        Directory of locally trained checkpoints, or the checkpoint itself, which wins over
        everything else.
    allow_download : bool
        Whether a missing checkpoint may be fetched from the configured release.
    progress : ProgressCallback | None
        Called with the file name, the bytes received, and the total expected while downloading.

    Returns
    -------
    :py:class:`~pathlib.Path`
        Path to a checkpoint that exists.

    Raises
    ------
    WeightsError
        If no local copy exists and downloading is refused or fails.
    """
    for candidate in _candidates(name, override):
        if candidate.is_file():
            log.debug('Using weights at `%s`.', candidate)
            return candidate
    if not allow_download:
        searched = ', '.join(str(directory) for directory in weights_directories(override))
        msg = f'No `{name}` in {searched} and downloading was not permitted.'
        raise WeightsError(msg)
    return _download(name, progress)
