"""
Access to the trained models shipped inside the package.

The checkpoints total about seven megabytes, which is small enough to bundle so that an installed
copy generates charts without first needing a corpus, a preprocessing pass, and a training run.
Assets are read through :py:mod:`importlib.resources` rather than by path arithmetic, so they remain
reachable when the package is installed as a zip.
"""

from __future__ import annotations

from importlib import resources
from typing import TYPE_CHECKING
import io
import json
import logging

import torch

from .vocab import Vocabulary

if TYPE_CHECKING:
    from pathlib import Path

__all__ = (
    'ASSET_DIRECTORY',
    'PREVIEW_ASSET',
    'VOCABULARY_ASSET',
    'asset_bytes',
    'has_asset',
    'load_state_dict',
    'load_vocabulary',
)

log = logging.getLogger(__name__)

_PACKAGE = 'smlab'
ASSET_DIRECTORY = 'assets'
"""Directory inside the package holding the bundled models.

:meta hide-value:
"""
PREVIEW_ASSET = 'preview.pt'
"""Bundled preview model file name.

:meta hide-value:
"""
VOCABULARY_ASSET = 'vocabulary.json'
"""Bundled note-row vocabulary file name.

:meta hide-value:
"""


def has_asset(name: str) -> bool:
    """
    Return whether an asset is present in the installed package.

    Parameters
    ----------
    name : str
        Asset file name.

    Returns
    -------
    bool
        True when the asset can be read.
    """
    try:
        return resources.files(_PACKAGE).joinpath(ASSET_DIRECTORY, name).is_file()
    except (ModuleNotFoundError, FileNotFoundError):
        return False


def asset_bytes(name: str) -> bytes:
    """
    Read a bundled asset.

    Parameters
    ----------
    name : str
        Asset file name.

    Returns
    -------
    bytes
        The asset's contents.

    Raises
    ------
    FileNotFoundError
        If the package was built without the asset.
    """
    try:
        return (resources.files(_PACKAGE) / ASSET_DIRECTORY / name).read_bytes()
    except (ModuleNotFoundError, FileNotFoundError) as error:
        msg = f'{name} is not bundled with this installation of smlab.'
        raise FileNotFoundError(msg) from error


def load_state_dict(
    name: str, override: Path | None, device: torch.device
) -> dict[str, torch.Tensor]:
    """
    Load model weights, preferring a local file over the bundled copy.

    Parameters
    ----------
    name : str
        Asset file name, also the file name looked for inside ``override``.
    override : :py:class:`~pathlib.Path` | None
        Directory holding locally trained checkpoints, or ``None`` to use the bundled models.
    device : :py:class:`~torch.device`
        Device the tensors are mapped onto.

    Returns
    -------
    dict[str, :py:class:`~torch.Tensor`]
        The state dictionary.
    """
    source: Path | io.BytesIO
    if override is not None and (local := override / name).is_file():
        log.debug('Loading `%s` from `%s`.', name, local)
        source = local
    else:
        source = io.BytesIO(asset_bytes(name))
    state: dict[str, torch.Tensor] = torch.load(source, map_location=device, weights_only=True)
    return state


def load_vocabulary(override: Path | None) -> Vocabulary:
    """
    Load the note-row vocabulary, preferring a local file over the bundled copy.

    Parameters
    ----------
    override : :py:class:`~pathlib.Path` | None
        A vocabulary file to use instead of the bundled one.

    Returns
    -------
    Vocabulary
        The loaded vocabulary.
    """
    if override is not None and override.is_file():
        return Vocabulary.load(override)
    return Vocabulary(json.loads(asset_bytes(VOCABULARY_ASSET)))
