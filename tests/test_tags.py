"""Tests for reading song metadata from audio tags."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mutagen.id3 import ID3, TCON, TIT2, TPE1
from mutagen.oggvorbis import OggVorbis
from smlab.tags import apply_tags, read_tags
from smlab.writer import SongMetadata
import pytest

if TYPE_CHECKING:
    from pathlib import Path


def _mp3_with_tags(path: Path, data: bytes, **frames: str) -> Path:
    path.write_bytes(data)
    # Mutagen ships a py.typed marker but leaves its ID3 frame classes untyped.
    tags = ID3()  # type: ignore[no-untyped-call]
    if 'title' in frames:
        tags.add(TIT2(encoding=3, text=frames['title']))  # type: ignore[no-untyped-call]
    if 'artist' in frames:
        tags.add(TPE1(encoding=3, text=frames['artist']))  # type: ignore[no-untyped-call]
    if 'genre' in frames:
        tags.add(TCON(encoding=3, text=frames['genre']))  # type: ignore[no-untyped-call]
    tags.save(path)
    return path


def test_mp3_id3_tags_are_read(tmp_path: Path, mp3_bytes: bytes) -> None:
    path = _mp3_with_tags(
        tmp_path / 'song.mp3', mp3_bytes, artist='dj TAKA', genre='Techno', title='Colors'
    )
    assert read_tags(path) == {'artist': 'dj TAKA', 'genre': 'Techno', 'title': 'Colors'}


def test_ogg_vorbis_comments_are_read(tmp_path: Path, ogg_bytes: bytes) -> None:
    path = tmp_path / 'song.ogg'
    path.write_bytes(ogg_bytes)
    audio = OggVorbis(path)  # type: ignore[no-untyped-call]
    audio['title'] = ['Sakura']
    audio['artist'] = ['RevenG']
    audio['genre'] = ['Trance']
    audio.save()
    assert read_tags(path) == {'artist': 'RevenG', 'genre': 'Trance', 'title': 'Sakura'}


def test_wav_tags_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / 'song.wav'
    path.write_bytes(b'RIFF....WAVE')
    assert read_tags(path) == {}


def test_untagged_file_yields_nothing(tmp_path: Path) -> None:
    path = tmp_path / 'bare.mp3'
    path.write_bytes(b'')
    assert read_tags(path) == {}


def test_unreadable_file_yields_nothing(tmp_path: Path) -> None:
    path = tmp_path / 'missing.mp3'
    assert read_tags(path) == {}


def test_explicit_metadata_wins_over_tags(tmp_path: Path, mp3_bytes: bytes) -> None:
    path = _mp3_with_tags(tmp_path / 'song.mp3', mp3_bytes, artist='Tag Artist', title='Tag Title')
    filled = apply_tags(SongMetadata(artist='Given Artist'), path)
    assert filled.artist == 'Given Artist'
    assert filled.title == 'Tag Title'


def test_missing_metadata_is_filled_from_tags(tmp_path: Path, mp3_bytes: bytes) -> None:
    path = _mp3_with_tags(
        tmp_path / 'song.mp3', mp3_bytes, artist='dj TAKA', genre='Techno', title='Colors'
    )
    filled = apply_tags(SongMetadata(), path)
    assert filled.title == 'Colors'
    assert filled.artist == 'dj TAKA'
    assert filled.genre == 'Techno'


def test_subtitle_is_never_taken_from_tags(tmp_path: Path, mp3_bytes: bytes) -> None:
    path = _mp3_with_tags(tmp_path / 'song.mp3', mp3_bytes, title='Colors')
    assert 'subtitle' not in read_tags(path)
    assert not apply_tags(SongMetadata(), path).subtitle


@pytest.mark.parametrize('suffix', ['.mp3', '.ogg', '.flac', '.m4a', '.opus'])
def test_tagged_suffixes_are_attempted(suffix: str, tmp_path: Path) -> None:
    path = tmp_path / f'song{suffix}'
    path.write_bytes(b'')
    assert read_tags(path) == {}


def test_a_file_with_no_tags_leaves_the_metadata_alone(tmp_path: Path) -> None:
    path = tmp_path / 'song.xyz'
    path.write_bytes(b'')
    original = SongMetadata(title='Kept', artist='Also Kept')
    assert apply_tags(original, path) is original
