"""
Reading song metadata from audio file tags.

MP3 carries ID3 frames and Ogg carries Vorbis comments, which name the same fields differently.
Mutagen's "easy" interface maps both onto one vocabulary, so only the field names peculiar to
simfiles need handling here. WAV is not supported, since it has no standard tagging scheme worth
relying on.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING
import logging

import mutagen

if TYPE_CHECKING:
    from pathlib import Path

    from .writer import SongMetadata

__all__ = ('TAGGED_SUFFIXES', 'apply_tags', 'read_tags')

log = logging.getLogger(__name__)

TAGGED_SUFFIXES = frozenset({'.flac', '.m4a', '.mp3', '.ogg', '.opus'})
"""
Audio extensions whose tags are read.

WAV is deliberately absent because it has no standard tag scheme.

:meta hide-value:
"""
# Simfile field name mapped to the tag names that may carry it, in preference order. Subtitle is
# deliberately absent: the tags that might supply it carry release details rather than the
# parenthetical a simfile expects, so it is left blank unless given explicitly.
_FIELD_TAGS = {
    'artist': ('artist', 'albumartist', 'performer'),
    'genre': ('genre',),
    'title': ('title',),
}


def read_tags(path: Path) -> dict[str, str]:
    """
    Read song metadata from an audio file's tags.

    Parameters
    ----------
    path : :py:class:`~pathlib.Path`
        The audio file to read.

    Returns
    -------
    dict[str, str]
        Simfile field names mapped to values, omitting anything absent or empty. An unreadable or
        untagged file yields an empty mapping.
    """
    if path.suffix.lower() not in TAGGED_SUFFIXES:
        return {}
    try:
        audio = mutagen.File(path, easy=True)
    except (OSError, mutagen.MutagenError) as error:
        log.debug('Could not read tags from `%s`: %s', path, error)
        return {}
    if audio is None or not audio.tags:
        return {}
    found: dict[str, str] = {}
    for field, names in _FIELD_TAGS.items():
        for name in names:
            values = audio.tags.get(name)
            if values and (value := str(values[0]).strip()):
                found[field] = value
                break
    return found


def apply_tags(metadata: SongMetadata, path: Path) -> SongMetadata:
    """
    Fill unset metadata fields from an audio file's tags.

    Values already present are preserved, so anything given on the command line wins over what the
    file claims.

    Parameters
    ----------
    metadata : SongMetadata
        Metadata gathered so far.
    path : :py:class:`~pathlib.Path`
        The audio file to read tags from.

    Returns
    -------
    SongMetadata
        A copy with any missing fields filled in.
    """
    if not (tags := read_tags(path)):
        return metadata
    return replace(
        metadata,
        artist=metadata.artist or tags.get('artist', ''),
        genre=metadata.genre or tags.get('genre', ''),
        title=metadata.title or tags.get('title', ''),
    )
