"""
Writing a generated song out as a directory.

A song is written as a directory named after its title, holding the simfile and a copy of the audio
under the same name, which is the layout StepMania expects inside a pack.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING
import shutil

from .common import safe_directory_name
from .dwi import render_dwi
from .sm import render_simfile
from .ssc import render_ssc

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from smlab.timing import TimingData

    from .common import Format, SongMetadata

__all__ = ('write_song',)


def write_song(
    metadata: SongMetadata,
    audio: Path,
    timing: TimingData,
    charts: Sequence[tuple[str, int, Sequence[tuple[int, Sequence[int]]]]],
    parent: Path,
    fmt: Format = 'ssc',
    seconds: float = 0.0,
) -> Path:
    """
    Write a complete song directory.

    Parameters
    ----------
    metadata : SongMetadata
        Header fields for the song. Its ``music`` field is replaced with the copied audio's name.
    audio : :py:class:`~pathlib.Path`
        Source audio file, which is copied rather than moved.
    timing : TimingData
        Tempo and offset for the song.
    charts : :py:class:`~collections.abc.Sequence`
        Difficulty name, rating, and rows for each chart to write.
    parent : :py:class:`~pathlib.Path`
        Directory the song directory is created inside, usually a pack.
    fmt : Format
        Simfile format to write.
    seconds : float
        Length of the song, used by the ``.ssc`` radar figures.

    Returns
    -------
    :py:class:`~pathlib.Path`
        The simfile that was written.
    """
    name = safe_directory_name(metadata.title or audio.stem, audio.stem)
    directory = parent / name
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f'{name}{audio.suffix.lower()}'
    if audio.resolve() != destination.resolve():
        shutil.copy2(audio, destination)
    header = replace(metadata, music=destination.name)
    match fmt:
        case 'ssc':
            text = render_ssc(header, timing, charts, seconds)
        case 'dwi':
            text = render_dwi(header, timing, charts)
        case _:
            text = render_simfile(header, timing, charts)
    simfile = directory / f'{name}.{fmt}'
    simfile.write_text(text)
    return simfile
