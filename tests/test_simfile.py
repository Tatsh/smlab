"""Tests for simfile parsing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from smlab.chart import normalize_difficulty
from smlab.simfile import SimfileError, load_simfile
import pytest

if TYPE_CHECKING:
    from pathlib import Path


def test_sm_header_is_parsed(sm_file: Path) -> None:
    simfile = load_simfile(sm_file)
    assert simfile.title == 'Test Song'
    assert simfile.artist == 'Nobody'
    assert simfile.music == 'test.ogg'
    assert simfile.file_format == 'sm'
    assert simfile.offset == pytest.approx(-0.048)
    assert simfile.offset_declared
    assert simfile.timing is not None
    assert simfile.timing.primary_bpm == pytest.approx(150.0)


def test_sm_chart_rows_land_on_expected_beats(sm_file: Path) -> None:
    chart = load_simfile(sm_file).singles()[0]
    assert chart.difficulty == 'Challenge'
    assert chart.meter == 9
    rows = list(chart.rows())
    assert [row.beat for row in rows] == [0.0, 1.0, 2.0, 3.0, 4.0, 6.0]
    assert rows[0].columns == '1000'
    assert rows[4].columns == '1001'


def test_ssc_notedata_block_is_parsed(ssc_file: Path) -> None:
    simfile = load_simfile(ssc_file)
    assert simfile.file_format == 'ssc'
    chart = simfile.singles()[0]
    assert chart.difficulty == 'Hard'
    assert chart.meter == 8
    assert len(list(chart.rows())) == 4


def test_dwi_gap_becomes_negative_offset(dwi_file: Path) -> None:
    simfile = load_simfile(dwi_file)
    assert simfile.file_format == 'dwi'
    assert simfile.offset == pytest.approx(-0.048)
    assert simfile.offset_declared
    assert simfile.timing is not None
    assert simfile.timing.primary_bpm == pytest.approx(150.0)


def test_dwi_difficulty_maps_to_stepmania_name(dwi_file: Path) -> None:
    assert load_simfile(dwi_file).singles()[0].difficulty == 'Hard'


def test_missing_offset_tag_is_reported(tmp_path: Path) -> None:
    path = tmp_path / 'no_offset.sm'
    path.write_text('#TITLE:X;\n#BPMS:0.000=150.000;\n')
    (tmp_path / 'x.ogg').write_bytes(b'')
    simfile = load_simfile(path)
    assert not simfile.offset_declared
    assert simfile.offset == pytest.approx(0.0)


def test_missing_bpms_raises(tmp_path: Path) -> None:
    path = tmp_path / 'broken.sm'
    path.write_text('#TITLE:X;\n')
    with pytest.raises(SimfileError, match='#BPMS'):
        load_simfile(path)


def test_unsupported_extension_raises(tmp_path: Path) -> None:
    path = tmp_path / 'thing.txt'
    path.write_text('#TITLE:X;')
    with pytest.raises(SimfileError, match='unsupported extension'):
        load_simfile(path)


@pytest.mark.parametrize(
    ('raw', 'expected'),
    [
        ('challenge', 'Challenge'),
        ('CHALLENGE', 'Challenge'),
        ('  Hard  ', 'Hard'),
        ('Beginner', 'Beginner'),
        ('Nonsense', 'Nonsense'),
    ],
)
def test_difficulty_names_are_normalized(raw: str, expected: str) -> None:
    assert normalize_difficulty(raw) == expected


def test_music_path_resolves_case_insensitively(tmp_path: Path) -> None:
    path = tmp_path / 'case.sm'
    path.write_text('#TITLE:X;\n#MUSIC:Song.OGG;\n#BPMS:0.000=150.000;\n')
    (tmp_path / 'song.ogg').write_bytes(b'')
    resolved = load_simfile(path).music_path()
    assert resolved is not None
    assert resolved.name == 'song.ogg'
