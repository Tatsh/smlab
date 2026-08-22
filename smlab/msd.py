"""
Tokeniser for the MSD tag syntax shared by ``.sm``, ``.ssc``, and ``.dwi``.

All three formats store values as ``#TAG:value;`` structures, optionally with
extra colon-separated fields, and allow ``//`` line comments. Values may span
lines, which is how note data is embedded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple
import re

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

__all__ = ('MsdTag', 'parse_beat_value_list', 'parse_float', 'parse_msd', 'read_simfile_text')

COMMENT_PATTERN = re.compile(r'//[^\r\n]*')
"""Matches an MSD line comment.

:meta hide-value:
"""
_ENCODINGS = ('utf-8', 'cp932')
"""Encodings tried in turn, before falling back to a lossy latin-1 read."""


class MsdTag(NamedTuple):
    """One parsed ``#TAG:value;`` structure."""

    tag: str
    """The tag name, upper-cased."""
    params: tuple[str, ...]
    """Colon-separated values following the tag name, each stripped."""


def read_simfile_text(path: Path) -> str:
    """
    Read a simfile, tolerating the mixed encodings found in real packs.

    Japanese-authored simfiles are frequently Shift-JIS rather than UTF-8, and a
    few are neither, so decoding falls back to Latin-1 with replacement.

    Parameters
    ----------
    path : :py:class:`~pathlib.Path`
        The file to read.

    Returns
    -------
    str
        The decoded text.
    """
    raw = path.read_bytes()
    for encoding in _ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode('latin-1', errors='replace')


def parse_msd(text: str) -> Iterator[MsdTag]:
    """
    Yield each ``#TAG:value;`` structure in an MSD document.

    A missing terminating semicolon is tolerated, because it occurs in the wild,
    by treating the next tag as the terminator.

    Parameters
    ----------
    text : str
        The full simfile text.

    Yields
    ------
    MsdTag
        Each tag encountered, in document order.
    """
    text = COMMENT_PATTERN.sub('', text)
    length = len(text)
    index = 0
    while (hash_at := text.find('#', index)) >= 0:
        semicolon = text.find(';', hash_at + 1)
        next_hash = text.find('#', hash_at + 1)
        if semicolon < 0:
            end = next_hash if next_hash >= 0 else length
        elif 0 <= next_hash < semicolon:
            # The tag was never terminated, so stop at the next one rather than
            # swallowing it into this value.
            end = next_hash
        else:
            end = semicolon
        body = text[hash_at + 1 : end]
        # A semicolon is consumed, but a tag that had to terminate the one
        # before it must be seen again next pass or it is dropped.
        index = end + 1 if end < length and text[end] == ';' else end
        if not body.strip():
            continue
        params = body.split(':')
        yield MsdTag(params[0].strip().upper(), tuple(param.strip() for param in params[1:]))


def parse_float(value: str, default: float = 0.0) -> float:
    """
    Parse a floating point tag value, falling back on malformed input.

    Parameters
    ----------
    value : str
        The raw tag value.
    default : float
        Value returned when parsing fails.

    Returns
    -------
    float
        The parsed number, or ``default``.
    """
    try:
        return float(value.strip())
    except (AttributeError, ValueError):
        return default


def parse_beat_value_list(value: str) -> tuple[tuple[float, float], ...]:
    """
    Parse a ``beat=value,beat=value`` list such as ``#BPMS`` or ``#STOPS``.

    Malformed entries are skipped rather than aborting the whole file, because
    trailing commas and stray whitespace are common.

    Parameters
    ----------
    value : str
        The raw tag value.

    Returns
    -------
    tuple[tuple[float, float], ...]
        Pairs of beat and value, sorted by beat.
    """
    pairs: list[tuple[float, float]] = []
    for chunk in value.replace('\n', '').replace('\r', '').split(','):
        if '=' not in (stripped := chunk.strip()):
            continue
        beat_text, _, value_text = stripped.partition('=')
        try:
            pairs.append((float(beat_text), float(value_text)))
        except ValueError:
            continue
    return tuple(sorted(pairs))
