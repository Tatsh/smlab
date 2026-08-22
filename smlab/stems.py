"""
Separating a song into the layers a chart can follow.

Steps follow whatever stands out: a bassline, a vocal line, a guitar solo, a
drum fill. A mel spectrogram of the mixture forces a network to disentangle
overlapping instruments from frequency alone, which is exactly what fails when
a guitar and a vocal share a band. Separating first turns "which layer is
prominent right now" into something a model can read rather than infer.

``htdemucs`` splits into drums, bass, other, and vocals, which is close to a
one-to-one match with the layers charts follow, and runs at roughly seventy
times realtime on a consumer GPU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast
import logging

import librosa
import numpy as np
import torch

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

__all__ = ('STEM_NAMES', 'SeparationError', 'Separator', 'load_separator', 'separate')

log = logging.getLogger(__name__)

STEM_NAMES = ('drums', 'bass', 'other', 'vocals')
"""
Layers a chart can follow, in the order the features concatenate them.

``other`` carries guitars, keys, and anything not caught by the named stems.

:meta hide-value:
"""
_MODEL_NAME = 'htdemucs'
_OVERLAP = 0.10


class SeparationError(Exception):
    """Raised when a song cannot be separated into stems."""


class Separator(Protocol):
    """
    The part of a demucs model this package uses.

    Demucs ships no stubs describing these attributes on the module it returns,
    so the surface actually relied upon is declared here instead.
    """

    @property
    def samplerate(self) -> int:
        """Sample rate the model expects."""

    @property
    def sources(self) -> list[str]:
        """Stem names, in the order the model emits them."""


def load_separator(device: torch.device) -> Separator:
    """
    Load the separation model.

    Parameters
    ----------
    device : :py:class:`~torch.device`
        Device to place the model on.

    Returns
    -------
    Separator
        The separation model, in evaluation mode.

    Raises
    ------
    SeparationError
        If demucs is not installed.
    """
    try:
        from demucs.pretrained import get_model  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - depends on the extra
        msg = 'Install the stems extra to separate audio: pip install smlab[stems]'
        raise SeparationError(msg) from error
    model = get_model(_MODEL_NAME)
    return cast('Separator', model.to(device).eval())


def separate(model: Separator, path: Path, device: torch.device) -> dict[str, NDArray[np.float32]]:
    """
    Split a song into its stems.

    Parameters
    ----------
    model : Separator
        A model from :func:`load_separator`.
    path : :py:class:`~pathlib.Path`
        Audio file to separate.
    device : :py:class:`~torch.device`
        Device to run on.

    Returns
    -------
    dict[str, :py:class:`~numpy.ndarray`]
        Mono stems keyed by name, at the separator's sample rate.

    Raises
    ------
    SeparationError
        If the audio cannot be read or separated.
    """
    from demucs.apply import apply_model  # noqa: PLC0415

    rate = model.samplerate
    try:
        samples, _ = librosa.load(str(path), sr=rate, mono=False)
    except (OSError, ValueError, RuntimeError) as error:
        msg = f'{path}: could not read audio'
        raise SeparationError(msg) from error
    if samples.ndim == 1:
        samples = np.stack([samples, samples])
    waveform = torch.from_numpy(np.ascontiguousarray(samples)).float()
    try:
        with torch.no_grad():
            separated = apply_model(
                cast('Any', model),
                waveform[None].to(device),
                device=device,
                split=True,
                overlap=_OVERLAP,
            )[0]
    except (RuntimeError, ValueError) as error:
        msg = f'{path}: separation failed'
        raise SeparationError(msg) from error
    order = list(model.sources)
    return {
        name: separated[order.index(name)].mean(dim=0).cpu().numpy().astype(np.float32)
        for name in STEM_NAMES
        if name in order
    }
