"""Tests for rendering charts back to simfile text."""

from __future__ import annotations

from typing import TYPE_CHECKING

from smlab.dataset import SUBDIVISIONS_PER_BEAT
from smlab.simfile import load_simfile
from smlab.timing import TimingData
from smlab.writer import (
    SLOTS_PER_MEASURE,
    SongMetadata,
    measure_text,
    render_simfile,
    safe_directory_name,
    write_song,
)
import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ('stride', 'expected_lines'),
    [
        (SLOTS_PER_MEASURE // 4, 4),
        (SLOTS_PER_MEASURE // 8, 8),
        (SLOTS_PER_MEASURE // 16, 16),
        (SLOTS_PER_MEASURE // 12, 12),
    ],
)
def test_measure_uses_coarsest_subdivision(stride: int, expected_lines: int) -> None:
    rows = {index * stride: [1, 0, 0, 0] for index in range(SLOTS_PER_MEASURE // stride)}
    assert len(measure_text(rows).splitlines()) == expected_lines


def test_empty_measure_is_four_blank_rows() -> None:
    assert measure_text({}).splitlines() == ['0000'] * 4


def test_rendered_simfile_declares_timing() -> None:
    timing = TimingData.constant(150.0, -0.048)
    text = render_simfile(
        SongMetadata(music='song.ogg', title='Song'), timing, [('Hard', 8, [(0, [1, 0, 0, 0])])]
    )
    assert '#OFFSET:-0.048000;' in text
    assert '#BPMS:0.000=150.000;' in text
    assert '#MUSIC:song.ogg;' in text
    assert 'dance-single' in text


def test_rendered_simfile_round_trips(tmp_path: Path) -> None:
    timing = TimingData.constant(150.0, -0.048)
    rows = [
        (0, [1, 0, 0, 0]),
        (SUBDIVISIONS_PER_BEAT, [0, 1, 0, 0]),
        (SUBDIVISIONS_PER_BEAT * 2, [0, 0, 1, 0]),
        (SUBDIVISIONS_PER_BEAT * 3, [0, 0, 0, 1]),
    ]
    path = tmp_path / 'out.sm'
    path.write_text(
        render_simfile(SongMetadata(music='song.ogg', title='Song'), timing, [('Hard', 8, rows)])
    )
    (tmp_path / 'song.ogg').write_bytes(b'')
    parsed = load_simfile(path)
    assert parsed.timing is not None
    assert parsed.timing.offset == pytest.approx(-0.048)
    assert parsed.timing.primary_bpm == pytest.approx(150.0)
    chart = parsed.singles()[0]
    assert chart.difficulty == 'Hard'
    assert chart.meter == 8
    assert [row.beat for row in chart.rows()] == [0.0, 1.0, 2.0, 3.0]
    assert [row.columns for row in chart.rows()] == ['1000', '0100', '0010', '0001']


def test_multiple_charts_are_written(tmp_path: Path) -> None:
    timing = TimingData.constant(120.0, 0.0)
    charts = [('Easy', 3, [(0, [1, 0, 0, 0])]), ('Hard', 9, [(0, [0, 0, 0, 1])])]
    path = tmp_path / 'multi.sm'
    path.write_text(render_simfile(SongMetadata(music='song.ogg', title='Song'), timing, charts))
    (tmp_path / 'song.ogg').write_bytes(b'')
    parsed = load_simfile(path)
    assert [chart.difficulty for chart in parsed.singles()] == ['Easy', 'Hard']


def test_metadata_tags_are_written() -> None:
    metadata = SongMetadata(
        artist='Someone',
        artist_translit='Someone Romanised',
        background='bg.png',
        banner='banner.png',
        cdtitle='cd.png',
        credit='Charter',
        genre='Techno',
        music='song.ogg',
        sample_length=12.5,
        sample_start=42.25,
        subtitle='A Subtitle',
        title='A Title',
    )
    text = render_simfile(metadata, TimingData.constant(150.0, 0.0), [])
    for tag in (
        '#TITLE:A Title;',
        '#SUBTITLE:A Subtitle;',
        '#ARTIST:Someone;',
        '#ARTISTTRANSLIT:Someone Romanised;',
        '#GENRE:Techno;',
        '#CREDIT:Charter;',
        '#BANNER:banner.png;',
        '#BACKGROUND:bg.png;',
        '#CDTITLE:cd.png;',
        '#SAMPLESTART:42.250;',
        '#SAMPLELENGTH:12.500;',
    ):
        assert tag in text


@pytest.mark.parametrize(
    ('title', 'expected'),
    [
        ('Colors', 'Colors'),
        ('AC/DC', 'ACDC'),
        ('What? Really*', 'What Really'),
        ('  Padded  ', 'Padded'),
        ('trailing.', 'trailing'),
        ('', 'Untitled'),
        ('///', 'Untitled'),
    ],
)
def test_directory_names_are_sanitized(title: str, expected: str) -> None:
    assert safe_directory_name(title) == expected


def test_write_song_creates_folder_with_audio_copy(tmp_path: Path) -> None:
    source = tmp_path / 'source' / 'whatever.ogg'
    source.parent.mkdir()
    source.write_bytes(b'audio-bytes')
    pack = tmp_path / 'pack'
    metadata = SongMetadata(artist='Someone', title='My Song')
    simfile = write_song(
        metadata,
        source,
        TimingData.constant(150.0, -0.048),
        [('Hard', 8, [(0, [1, 0, 0, 0])])],
        pack,
    )
    assert simfile == pack / 'My Song' / 'My Song.sm'
    assert simfile.is_file()
    copied = pack / 'My Song' / 'My Song.ogg'
    assert copied.read_bytes() == b'audio-bytes'
    assert source.is_file()
    parsed = load_simfile(simfile)
    assert parsed.title == 'My Song'
    assert parsed.artist == 'Someone'
    assert parsed.music == 'My Song.ogg'
    resolved = parsed.music_path()
    assert resolved is not None
    assert resolved.name == 'My Song.ogg'


def test_write_song_falls_back_to_audio_stem(tmp_path: Path) -> None:
    source = tmp_path / 'Fallback Name.mp3'
    source.write_bytes(b'x')
    simfile = write_song(SongMetadata(), source, TimingData.constant(120.0, 0.0), [], tmp_path)
    assert simfile.parent.name == 'Fallback Name'
    assert (simfile.parent / 'Fallback Name.mp3').is_file()


def test_holds_survive_a_round_trip(tmp_path: Path) -> None:
    timing = TimingData.constant(120.0, 0.0)
    rows = [(0, [2, 0, 0, 0]), (SUBDIVISIONS_PER_BEAT * 2, [3, 0, 0, 0])]
    path = tmp_path / 'hold.sm'
    path.write_text(
        render_simfile(SongMetadata(music='song.ogg', title='Song'), timing, [('Hard', 8, rows)])
    )
    (tmp_path / 'song.ogg').write_bytes(b'')
    columns = [row.columns for row in load_simfile(path).singles()[0].rows()]
    assert columns == ['2000', '3000']
