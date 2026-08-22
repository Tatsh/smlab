"""Tests for reading the MSD tag format simfiles use."""

from __future__ import annotations

from typing import TYPE_CHECKING

from smlab.msd import parse_beat_value_list, parse_float, parse_msd, read_simfile_text
import pytest

if TYPE_CHECKING:
    from pathlib import Path


def test_tags_and_their_parameters_are_split() -> None:
    tags = list(parse_msd('#TITLE:Song;#BPMS:0.000=150.000;'))
    assert tags[0].tag == 'TITLE'
    assert tags[0].params == ('Song',)
    assert tags[1].tag == 'BPMS'


def test_a_tag_name_is_upper_cased() -> None:
    assert next(parse_msd('#title:Song;')).tag == 'TITLE'


def test_a_tag_with_several_parameters_keeps_them_all() -> None:
    assert next(parse_msd('#NOTES:dance-single:desc:Hard:8:0,0,0,0,0;')).params == (
        'dance-single',
        'desc',
        'Hard',
        '8',
        '0,0,0,0,0',
    )


def test_text_outside_a_tag_is_ignored() -> None:
    assert [tag.tag for tag in parse_msd('junk\n#TITLE:Song;\nmore junk')] == ['TITLE']


def test_comments_are_stripped() -> None:
    assert next(parse_msd('// a note about the file\n#TITLE:Song;')).params == ('Song',)


def test_an_unterminated_tag_stops_at_the_next_one() -> None:
    # Without this the unterminated value swallows every tag after it.
    assert [tag.tag for tag in parse_msd('#TITLE:Song\n#ARTIST:Nobody;')] == ['TITLE', 'ARTIST']


def test_a_final_unterminated_tag_runs_to_the_end() -> None:
    assert next(parse_msd('#TITLE:Song')).params == ('Song',)


def test_an_empty_tag_is_skipped() -> None:
    assert [tag.tag for tag in parse_msd('#;#TITLE:Song;')] == ['TITLE']


def test_text_with_no_tags_yields_nothing() -> None:
    assert list(parse_msd('nothing here at all')) == []


@pytest.mark.parametrize(
    ('value', 'wanted'),
    [('1.5', 1.5), ('-0.048', -0.048), ('  2  ', 2.0)],
)
def test_a_number_is_read(value: str, wanted: float) -> None:
    assert parse_float(value) == pytest.approx(wanted)


def test_an_unreadable_number_falls_back() -> None:
    assert parse_float('not a number', 7.0) == pytest.approx(7.0)
    assert parse_float('') == pytest.approx(0.0)


def test_a_beat_value_list_is_read_in_beat_order() -> None:
    assert parse_beat_value_list('4.000=180.000,0.000=150.000') == (
        (0.0, 150.0),
        (4.0, 180.0),
    )


def test_an_entry_that_is_not_a_pair_is_dropped() -> None:
    assert parse_beat_value_list('0.000=150.000,rubbish,8.000=x') == ((0.0, 150.0),)


def test_an_empty_beat_value_list_reads_as_nothing() -> None:
    assert parse_beat_value_list('') == ()


def test_a_file_is_read_as_utf8(tmp_path: Path) -> None:
    path = tmp_path / 'song.sm'
    path.write_text('#TITLE:Sakura さくら;', encoding='utf-8')
    assert 'さくら' in read_simfile_text(path)


def test_a_legacy_encoding_is_read(tmp_path: Path) -> None:
    # Older simfiles predate UTF-8 and would otherwise fail to decode at all.
    path = tmp_path / 'song.sm'
    path.write_bytes('#TITLE:さくら;'.encode('cp932'))
    assert read_simfile_text(path).startswith('#TITLE:')


def test_undecodable_bytes_are_replaced_rather_than_raising(tmp_path: Path) -> None:
    # A lead byte with no valid trail defeats both encodings tried, and a
    # simfile that will not decode at all is worse than one with a mangled
    # title.
    path = tmp_path / 'song.sm'
    path.write_bytes(b'#TITLE:\x80\x81 broken;')
    assert read_simfile_text(path).startswith('#TITLE:')
