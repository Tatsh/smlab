"""Tests for simfile parsing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from smlab.chart import normalize_difficulty
from smlab.simfile import SimfileError, load_simfile
from tests.conftest import SM_TEXT

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

    from smlab.chart import Chart


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


def _simfile_with_audio(tmp_path: Path, music: str = 'test.ogg') -> Path:
    path = tmp_path / 'song.sm'
    path.write_text(SM_TEXT.replace('#MUSIC:test.ogg;', f'#MUSIC:{music};'))
    (tmp_path / 'test.ogg').write_bytes(b'')
    return path


def _dwi_chart(tmp_path: Path, stream: str) -> Chart:
    """Parse one DWI note stream and return the chart it produces."""
    path = tmp_path / 'stream.dwi'
    path.write_text(
        f'#TITLE:Stream;\n#FILE:song.mp3;\n#BPM:150;\n#GAP:0;\n#SINGLE:MANIAC:9:\n{stream};\n'
    )
    (tmp_path / 'song.mp3').write_bytes(b'')
    return next(iter(load_simfile(path).singles()))


def test_a_bare_character_is_an_eighth_note(tmp_path: Path) -> None:
    # DWI's default quantisation is two rows per beat.
    chart = _dwi_chart(tmp_path, '46')
    assert [beat for beat, _ in chart.rows()] == [0.0, 0.5]


def test_a_bracket_group_subdivides_the_beat(tmp_path: Path) -> None:
    # Round brackets are sixteenths, square are twelfths, curly are forty-eighths.
    assert [beat for beat, _ in _dwi_chart(tmp_path, '(46)').rows()] == [0.0, 0.25]
    assert [beat for beat, _ in _dwi_chart(tmp_path, '{46}').rows()] == [0.0, 0.0625]


def test_quantization_returns_to_what_it_was_after_a_group(tmp_path: Path) -> None:
    beats = [beat for beat, _ in _dwi_chart(tmp_path, '(44)6').rows()]
    assert beats == [0.0, 0.25, 0.5]


def test_a_stray_closing_bracket_is_survived(tmp_path: Path) -> None:
    assert [beat for beat, _ in _dwi_chart(tmp_path, ')46').rows()] == [0.0, 0.5]


def test_backticks_toggle_the_finest_quantization(tmp_path: Path) -> None:
    beats = [beat for beat, _ in _dwi_chart(tmp_path, '`44`6').rows()]
    assert beats[1] == pytest.approx(1.0 / 48.0)
    assert beats[2] == pytest.approx(2.0 / 48.0)


def test_an_angle_group_is_one_row_of_several_panels(tmp_path: Path) -> None:
    rows = list(_dwi_chart(tmp_path, '<46>').rows())
    assert len(rows) == 1
    assert rows[0][1] == '1001'


def test_an_angle_group_holding_a_zero_is_the_finest_quantization_instead(tmp_path: Path) -> None:
    # The same bracket spells both a chord and 192nd notes; a zero before the closing bracket is
    # what tells them apart, since a chord never names one.
    beats = [row.beat for row in _dwi_chart(tmp_path, '<40006>4').rows()]
    assert beats == pytest.approx([0.0, 4.0 / 48.0, 5.0 / 48.0])


def test_quantization_returns_to_eighths_after_a_tick_group(tmp_path: Path) -> None:
    beats = [row.beat for row in _dwi_chart(tmp_path, '<40006>44').rows()]
    assert beats[-1] == pytest.approx(5.0 / 48.0 + 0.5)


def test_an_unterminated_angle_group_stops_the_stream(tmp_path: Path) -> None:
    assert list(_dwi_chart(tmp_path, '4<6').rows()) == [(0.0, '1000')]


def test_a_digit_naming_two_panels_writes_both(tmp_path: Path) -> None:
    assert list(_dwi_chart(tmp_path, '1').rows()) == [(0.0, '1100')]


def test_a_bang_opens_a_freeze_that_a_later_step_closes(tmp_path: Path) -> None:
    rows = list(_dwi_chart(tmp_path, '4!40004').rows())
    assert [row.columns for row in rows] == ['2000', '3000']
    assert [row.beat for row in rows] == [0.0, 2.0]


def test_the_freeze_marker_does_not_advance_the_beat(tmp_path: Path) -> None:
    # The panel after the bang names what is held; it is part of the same step rather than the
    # next one, so the steps around it stay where they belong.
    assert [row.beat for row in _dwi_chart(tmp_path, '4!46').rows()] == [0.0, 0.5]


def test_a_freeze_may_be_marked_on_a_panel_the_step_does_not_name(tmp_path: Path) -> None:
    rows = list(_dwi_chart(tmp_path, '0!40004').rows())
    assert [row.columns for row in rows] == ['2000', '3000']


def test_characters_that_mean_nothing_are_skipped(tmp_path: Path) -> None:
    assert [beat for beat, _ in _dwi_chart(tmp_path, '4 \n\t 6').rows()] == [0.0, 0.5]


def test_a_stream_holding_no_steps_yields_one_silent_measure(tmp_path: Path) -> None:
    assert list(_dwi_chart(tmp_path, '   ').rows()) == []


def test_a_zero_names_no_panel_but_still_advances(tmp_path: Path) -> None:
    assert [beat for beat, _ in _dwi_chart(tmp_path, '040').rows()] == [0.5]


def test_a_long_stream_spans_as_many_measures_as_it_needs(tmp_path: Path) -> None:
    beats = [beat for beat, _ in _dwi_chart(tmp_path, '4' * 40).rows()]
    assert len(beats) == 40
    assert beats[-1] == pytest.approx(19.5)


def test_a_dwi_without_a_tempo_is_reported(tmp_path: Path) -> None:
    path = tmp_path / 'no_tempo.dwi'
    path.write_text('#TITLE:X;\n#FILE:song.mp3;\n#BPM:0;\n#GAP:0;\n#SINGLE:MANIAC:9:\n4;\n')
    (tmp_path / 'song.mp3').write_bytes(b'')
    with pytest.raises(SimfileError, match='no usable #BPM'):
        load_simfile(path)


def test_dwi_tempo_changes_are_read(tmp_path: Path) -> None:
    path = tmp_path / 'changes.dwi'
    path.write_text(
        '#TITLE:Changes;\n#FILE:song.mp3;\n#BPM:150;\n#GAP:0;\n'
        '#CHANGEBPM:4.0=180.0;\n#FREEZE:8.0=500;\n'
        '#SINGLE:MANIAC:9:\n4444;\n'
    )
    (tmp_path / 'song.mp3').write_bytes(b'')
    timing = load_simfile(path).timing
    assert timing is not None
    assert not timing.is_constant_bpm
    assert len(timing.stops) == 1


def test_a_doubles_chart_uses_eight_panels(tmp_path: Path) -> None:
    text = SM_TEXT.replace('dance-single', 'dance-double')
    path = tmp_path / 'double.sm'
    path.write_text(text)
    (tmp_path / 'test.ogg').write_bytes(b'')
    charts = load_simfile(path).charts
    assert not charts[0].is_single
    assert charts[0].panel_count() == 8


def test_a_measure_holding_only_comments_is_skipped(tmp_path: Path) -> None:
    # A measure whose lines are all comments has no rows to divide the beats between, so it must
    # not be measured at all.
    text = SM_TEXT.replace('1001\n0000\n0110\n0000', '// nothing here\n// nor here')
    path = tmp_path / 'commented.sm'
    path.write_text(text)
    (tmp_path / 'test.ogg').write_bytes(b'')
    chart = next(iter(load_simfile(path).singles()))
    assert [beat for beat, _ in chart.rows()] == [0.0, 1.0, 2.0, 3.0]


def test_an_unreadable_song_directory_finds_no_audio(tmp_path: Path, mocker: MockerFixture) -> None:
    parsed = load_simfile(_simfile_with_audio(tmp_path))
    mocker.patch('pathlib.Path.iterdir', side_effect=OSError)
    assert parsed.music_path() is None


def test_the_music_tag_is_matched_whatever_its_case(tmp_path: Path) -> None:
    # Packs move between filesystems that disagree about case, so a tag naming TEST.OGG has to find
    # test.ogg.
    path = _simfile_with_audio(tmp_path, music='TEST.OGG')
    found = load_simfile(path).music_path()
    assert found is not None
    assert found.name == 'test.ogg'


def test_audio_is_found_by_extension_when_the_tag_is_wrong(tmp_path: Path) -> None:
    path = _simfile_with_audio(tmp_path, music='missing.ogg')
    found = load_simfile(path).music_path()
    assert found is not None
    assert found.suffix == '.ogg'


def test_a_directory_with_no_audio_at_all_finds_none(tmp_path: Path) -> None:
    path = tmp_path / 'song.sm'
    path.write_text(SM_TEXT)
    assert load_simfile(path).music_path() is None


def test_the_music_tag_is_used_directly_when_it_matches(tmp_path: Path) -> None:
    found = load_simfile(_simfile_with_audio(tmp_path)).music_path()
    assert found is not None
    assert found.name == 'test.ogg'


def test_audio_is_found_when_no_music_tag_is_declared(tmp_path: Path) -> None:
    path = tmp_path / 'song.sm'
    path.write_text(SM_TEXT.replace('#MUSIC:test.ogg;\n', ''))
    (tmp_path / 'test.ogg').write_bytes(b'')
    found = load_simfile(path).music_path()
    assert found is not None
    assert found.suffix == '.ogg'
