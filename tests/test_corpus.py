"""Tests for scanning a Songs tree into a manifest."""

from __future__ import annotations

from typing import TYPE_CHECKING
import json
import pathlib

import pytest

from smlab.corpus import (
    EXCLUDED_PACKS,
    KEYBOARD_PACKS,
    KEYBOARD_SONGS,
    choose_simfile,
    iter_song_dirs,
    scan_corpus,
    summarize_song,
    write_manifest,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from pytest_mock import MockerFixture

from tests.conftest import SM_TEXT


def _song(root: Path, pack: str, title: str, *, text: str = SM_TEXT, suffix: str = '.sm') -> Path:
    song_dir = root / pack / title
    song_dir.mkdir(parents=True)
    (song_dir / f'{title}{suffix}').write_text(text)
    (song_dir / 'test.ogg').write_bytes(b'')
    return song_dir


def test_every_song_directory_is_found(tmp_path: Path) -> None:
    _song(tmp_path, 'Pack A', 'One')
    _song(tmp_path, 'Pack B', 'Two')
    assert [(pack, song.name) for pack, song in iter_song_dirs(tmp_path)] == [
        ('Pack A', 'One'),
        ('Pack B', 'Two'),
    ]


def test_excluded_packs_are_skipped(tmp_path: Path) -> None:
    # Training on this tool's own output would feed it its own mistakes back.
    _song(tmp_path, 'smlab output', 'Generated')
    _song(tmp_path, 'Real Pack', 'Human')
    assert [song.name for _, song in iter_song_dirs(tmp_path, EXCLUDED_PACKS)] == ['Human']


def test_loose_files_beside_the_packs_are_ignored(tmp_path: Path) -> None:
    _song(tmp_path, 'Pack', 'Song')
    (tmp_path / 'readme.txt').write_text('not a pack')
    (tmp_path / 'Pack' / 'banner.png').write_bytes(b'')
    assert [song.name for _, song in iter_song_dirs(tmp_path)] == ['Song']


def test_an_unreadable_pack_is_reported_and_skipped(tmp_path: Path, mocker: MockerFixture) -> None:
    _song(tmp_path, 'Locked', 'Song')
    _song(tmp_path, 'Open', 'Song')
    readable = pathlib.Path.iterdir

    def guarded(self: pathlib.Path) -> Iterator[pathlib.Path]:
        if self.name == 'Locked':
            raise OSError
        return readable(self)

    mocker.patch('pathlib.Path.iterdir', guarded)
    assert [pack for pack, _ in iter_song_dirs(tmp_path)] == ['Open']


def test_the_richest_simfile_format_wins(tmp_path: Path) -> None:
    # SSC supersedes SM and carries finer timing data.
    song_dir = _song(tmp_path, 'Pack', 'Song')
    (song_dir / 'Song.ssc').write_text('#TITLE:Song;')
    (song_dir / 'Song.dwi').write_text('#TITLE:Song;')
    chosen = choose_simfile(song_dir)
    assert chosen is not None
    assert chosen.suffix == '.ssc'


def test_a_directory_with_no_simfile_chooses_nothing(tmp_path: Path) -> None:
    empty = tmp_path / 'Empty'
    empty.mkdir()
    assert choose_simfile(empty) is None


def test_an_unreadable_directory_chooses_nothing(tmp_path: Path, mocker: MockerFixture) -> None:
    mocker.patch('pathlib.Path.iterdir', side_effect=OSError)
    assert choose_simfile(tmp_path) is None


def test_a_song_is_summarised_with_its_timing(tmp_path: Path) -> None:
    song_dir = _song(tmp_path, 'Pack', 'Song')
    record = summarize_song(('Pack', song_dir))
    assert record is not None
    assert record['primary_bpm'] == pytest.approx(150.0)
    assert record['offset'] == pytest.approx(-0.048)
    assert record['constant_bpm'] is True
    assert record['pack'] == 'Pack'
    assert record['stops'] == 0
    assert record['charts'][0]['difficulty'] == 'Challenge'


def test_a_song_without_a_simfile_is_not_summarised(tmp_path: Path) -> None:
    empty = tmp_path / 'Empty'
    empty.mkdir()
    assert summarize_song(('Pack', empty)) is None


def test_a_malformed_simfile_is_not_summarised(tmp_path: Path) -> None:
    song_dir = _song(tmp_path, 'Pack', 'Broken', text='this is not a simfile')
    assert summarize_song(('Pack', song_dir)) is None


def test_a_simfile_without_audio_is_not_summarised(tmp_path: Path) -> None:
    song_dir = _song(tmp_path, 'Pack', 'Silent')
    (song_dir / 'test.ogg').unlink()
    assert summarize_song(('Pack', song_dir)) is None


def test_a_known_keyboard_pack_overrides_the_feasibility_check(tmp_path: Path) -> None:
    # A keyboard chart that happens to be danceable still carries keyboard
    # phrasing, so the pack list wins over what two feet could reach.
    pack = next(iter(KEYBOARD_PACKS))
    song_dir = _song(tmp_path, pack, 'Song')
    record = summarize_song((pack, song_dir))
    assert record is not None
    assert record['charts'][0]['style'] == 'keyboard'


def test_a_known_keyboard_song_overrides_the_feasibility_check(tmp_path: Path) -> None:
    title = next(iter(KEYBOARD_SONGS))
    song_dir = _song(tmp_path, 'Ordinary Pack', title)
    record = summarize_song(('Ordinary Pack', song_dir))
    assert record is not None
    assert record['charts'][0]['style'] == 'keyboard'


def test_scanning_yields_only_usable_songs(tmp_path: Path) -> None:
    _song(tmp_path, 'Pack', 'Good')
    _song(tmp_path, 'Pack', 'Broken', text='not a simfile')
    titles = [record['title'] for record in scan_corpus(tmp_path, workers=1)]
    assert titles == ['Test Song']


def test_a_manifest_is_written_as_sorted_json(tmp_path: Path) -> None:
    song_dir = _song(tmp_path, 'Pack', 'Song')
    record = summarize_song(('Pack', song_dir))
    assert record is not None
    destination = tmp_path / 'out' / 'manifest.json'
    assert write_manifest([record], destination) == 1
    written = json.loads(destination.read_text(encoding='utf-8'))
    assert written[0]['title'] == 'Test Song'
    assert list(written[0]) == sorted(written[0])
