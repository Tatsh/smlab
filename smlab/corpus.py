"""
Scanning of a StepMania ``Songs`` tree into a ground-truth timing manifest.

Every simfile in the corpus carries a human-verified tempo and offset paired
with its audio, which makes the corpus a labelled benchmark for the timing
estimator rather than merely training data for the step models.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING
import json
import logging

from .dataset import CODE_BY_CHAR
from .playability import analyze_rows
from .simfile import SimfileError, load_simfile

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

    from .typing import ChartRecord, SongRecord

__all__ = (
    'EXCLUDED_PACKS',
    'KEYBOARD_PACKS',
    'KEYBOARD_SONGS',
    'PREFERRED_SUFFIXES',
    'choose_simfile',
    'iter_song_dirs',
    'scan_corpus',
    'summarize_song',
    'write_manifest',
)

log = logging.getLogger(__name__)

EXCLUDED_PACKS = frozenset({
    'smlab output',
    "MetzgerSM's S.E.X. Guitar Pack",
    'Vertex Minipack',
})
"""
Packs left out of the corpus by default.

``smlab output`` holds this tool's own generated charts, and training on those
would feed the model its own mistakes back as ground truth. The others are
excluded because their charting is not representative.

:meta hide-value:
"""
KEYBOARD_PACKS = frozenset({
    'BMR Originals-Exclusive',
    'BeMaNiRuler Originals',
    'Community Keyboard Megapack - Volume 1',
    'Community Keyboard Megapack - Volume 2',
    'DragonForce',
    'DragonForce - Ultra Beatdown',
})
"""
Packs authored for keyboard, whatever a feasibility check concludes.

Whether two feet *could* reach the panels is a different question from whether
the chart was written for them. A keyboard chart that happens to be danceable
still carries keyboard phrasing, and labelling it otherwise teaches the model
that the two idioms are interchangeable.

:meta hide-value:
"""
KEYBOARD_SONGS = frozenset({
    '(Reach) Air',
    'BRING THE PAIN',
    'Crimson King',
    'NUMBER ONE',
    'RAMP! (the logical song)',
    'Salieri Strikes Back',
    'Strangeprogam',
    'Stratofortress',
    'Stuntin-Like-My-Daddy',
})
"""
Individual song directories authored for keyboard.

Matched on the directory name, so copies of the same song in several packs are
all covered.

:meta hide-value:
"""
PREFERRED_SUFFIXES = ('.ssc', '.sm', '.dwi')
"""Simfile extensions in descending order of preference.

``.ssc`` supersedes ``.sm`` and carries finer timing data, while ``.dwi`` is the
oldest format and quantises its gap to whole milliseconds.

:meta hide-value:
"""


def iter_song_dirs(
    root: Path, excluded: frozenset[str] = EXCLUDED_PACKS
) -> Iterator[tuple[str, Path]]:
    """
    Yield every song directory beneath a ``Songs`` tree.

    The tree is two levels deep: a pack directory contains song directories.

    Parameters
    ----------
    root : :py:class:`~pathlib.Path`
        The ``Songs`` directory.
    excluded : frozenset[str]
        Pack names to skip.

    Yields
    ------
    tuple[str, :py:class:`~pathlib.Path`]
        The pack name and the song directory.
    """
    for pack in sorted(root.iterdir()):
        if not pack.is_dir() or pack.name in excluded:
            continue
        try:
            songs = sorted(pack.iterdir())
        except OSError:
            log.warning('Could not read pack `%s`.', pack)
            continue
        for song in songs:
            if song.is_dir():
                yield pack.name, song


def choose_simfile(song_dir: Path) -> Path | None:
    """
    Pick the best simfile in a song directory.

    Parameters
    ----------
    song_dir : :py:class:`~pathlib.Path`
        The directory to search.

    Returns
    -------
    :py:class:`~pathlib.Path` | None
        The preferred simfile, or ``None`` when the directory contains none.
    """
    try:
        entries = sorted(song_dir.iterdir())
    except OSError:
        return None
    for suffix in PREFERRED_SUFFIXES:
        for entry in entries:
            if entry.is_file() and entry.suffix.lower() == suffix:
                return entry
    return None


def summarize_song(item: tuple[str, Path]) -> SongRecord | None:
    """
    Summarise one song directory for the manifest.

    Parameters
    ----------
    item : tuple[str, :py:class:`~pathlib.Path`]
        The pack name and song directory, passed as one argument so that the
        function can be mapped across a process pool.

    Returns
    -------
    SongRecord | None
        The summary, or ``None`` when the song has no simfile, no audio, or no
        usable timing.
    """
    pack, song_dir = item
    if (simfile_path := choose_simfile(song_dir)) is None:
        return None
    try:
        simfile = load_simfile(simfile_path)
    except (SimfileError, OSError, ValueError) as error:
        log.debug('Could not parse `%s`: %s', simfile_path, error)
        return None
    if (timing := simfile.timing) is None or (audio := simfile.music_path()) is None:
        return None
    keyboard_only = pack in KEYBOARD_PACKS or song_dir.name in KEYBOARD_SONGS
    charts: list[ChartRecord] = []
    for chart in simfile.singles():
        rows = [
            (timing.time_at_beat(beat), [CODE_BY_CHAR.get(character, 0) for character in columns])
            for beat, columns in chart.rows()
        ]
        charts.append({
            'difficulty': chart.difficulty,
            'meter': chart.meter,
            'rows': len(rows),
            'style': 'keyboard' if keyboard_only else analyze_rows(rows).style,
        })
    return {
        'audio': str(audio),
        'bpms': [[segment.beat, segment.bpm] for segment in timing.bpms],
        'charts': charts,
        'constant_bpm': timing.is_constant_bpm,
        'file_format': simfile.file_format,
        'offset': timing.offset,
        'offset_declared': simfile.offset_declared,
        'pack': pack,
        'primary_bpm': timing.primary_bpm,
        'sample_length': simfile.sample_length,
        'sample_start': simfile.sample_start,
        'simfile': str(simfile_path),
        'stops': len(timing.stops),
        'title': simfile.title,
    }


def scan_corpus(
    root: Path, *, workers: int = 8, excluded: frozenset[str] = EXCLUDED_PACKS
) -> Iterator[SongRecord]:
    """
    Scan a ``Songs`` tree, yielding one record per usable song.

    Parameters
    ----------
    root : :py:class:`~pathlib.Path`
        The ``Songs`` directory.
    workers : int
        Number of worker processes to parse with.
    excluded : frozenset[str]
        Pack names to skip.

    Yields
    ------
    SongRecord
        Each song that has a simfile, audio, and usable timing.
    """
    songs = list(iter_song_dirs(root, excluded))
    log.info('Found %d song directories.', len(songs))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for record in pool.map(summarize_song, songs, chunksize=16):
            if record is not None:
                yield record


def write_manifest(records: Iterable[SongRecord], path: Path) -> int:
    """
    Write manifest records to a JSON file.

    Parameters
    ----------
    records : :py:class:`~collections.abc.Iterable`
        The records to write.
    path : :py:class:`~pathlib.Path`
        Destination file.

    Returns
    -------
    int
        Number of records written.
    """
    materialized = list(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(materialized, handle, indent=2, sort_keys=True)
    return len(materialized)
