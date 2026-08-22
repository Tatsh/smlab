"""Tests for rendering charts back to simfile text."""

from __future__ import annotations

from typing import TYPE_CHECKING
import hashlib
import re

from smlab.dataset import SUBDIVISIONS_PER_BEAT
from smlab.simfile import load_simfile
from smlab.timing import TimingData
from smlab.writer import (
    SLOTS_PER_MEASURE,
    STEPFILE_VERSION,
    SongMetadata,
    chart_hash,
    measure_text,
    radar_values,
    render_dwi,
    render_simfile,
    render_ssc,
    safe_directory_name,
    step_stream,
    write_song,
)
import pytest

_TIMING = TimingData.constant(150.0, -0.048)


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
    assert simfile == pack / 'My Song' / 'My Song.ssc'
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


def test_a_chart_with_no_rows_writes_one_silent_measure() -> None:
    text = render_simfile(SongMetadata(title='Empty'), _TIMING, [('Challenge', 9, [])])
    assert '0000\n0000\n0000\n0000' in text


def test_audio_already_in_place_is_not_copied_over_itself(tmp_path: Path) -> None:
    # Generating into the folder the audio already lives in must not truncate
    # the file by copying it onto itself.
    directory = tmp_path / 'Song'
    directory.mkdir()
    audio = directory / 'Song.ogg'
    audio.write_bytes(b'audio data')
    written = write_song(SongMetadata(title='Song'), audio, _TIMING, [('Easy', 3, [])], tmp_path)
    assert written.is_file()
    assert audio.read_bytes() == b'audio data'


def _ssc(rows: list[tuple[int, list[int]]], seconds: float = 60.0) -> str:
    return render_ssc(
        SongMetadata(title='Song', artist='Someone', credit='tatsh', music='Song.mp3'),
        _TIMING,
        [('Challenge', 16, rows)],
        seconds,
    )


def test_the_version_tag_comes_first() -> None:
    # StepMania decides how to read the rest of the file from it.
    assert _ssc([(0, [1, 0, 0, 0])]).splitlines()[0] == f'#VERSION:{STEPFILE_VERSION};'


def test_every_song_tag_is_written_even_when_empty() -> None:
    text = _ssc([(0, [1, 0, 0, 0])])
    for tag in (
        'TITLE',
        'SUBTITLE',
        'ARTIST',
        'TITLETRANSLIT',
        'SUBTITLETRANSLIT',
        'ARTISTTRANSLIT',
        'GENRE',
        'ORIGIN',
        'TAGS',
        'CREDIT',
        'BANNER',
        'BACKGROUND',
        'PREVIEWVID',
        'JACKET',
        'CDIMAGE',
        'DISCIMAGE',
        'LYRICSPATH',
        'CDTITLE',
        'MUSIC',
        'OFFSET',
        'SAMPLESTART',
        'SAMPLELENGTH',
        'SELECTABLE',
        'BPMS',
        'STOPS',
        'DELAYS',
        'WARPS',
        'TIMESIGNATURES',
        'TICKCOUNTS',
        'COMBOS',
        'SPEEDS',
        'SCROLLS',
        'XSCROLLS',
        'FAKES',
        'LABELS',
        'BGCHANGES',
        'ATTACKS',
    ):
        assert f'#{tag}:' in text


def test_a_timing_tag_carrying_a_value_closes_on_the_next_line() -> None:
    # The editor lays them out this way and the loader expects it.
    assert '#BPMS:0.000000=150.000000\n;' in _ssc([(0, [1, 0, 0, 0])])
    assert '#STOPS:;' in _ssc([(0, [1, 0, 0, 0])])


def test_every_chart_tag_is_written() -> None:
    text = _ssc([(0, [1, 0, 0, 0])])
    assert '//---------------dance-single - tatsh----------------' in text
    for tag in (
        'NOTEDATA',
        'CHARTNAME',
        'CHARTHASH',
        'CHARTTYPE',
        'STEPSTYPE',
        'DESCRIPTION',
        'CHARTSTYLE',
        'DIFFICULTY',
        'METER',
        'METERF',
        'LASTSECONDHINT',
        'RADARVALUES',
        'NOTES',
    ):
        assert f'#{tag}:' in text


def _notes_value(text: str) -> str:
    # Reproduce what MsdFile::ReadBuf hands the loader: comments removed, and nothing trimmed.
    body = text[text.index('#NOTES:') + len('#NOTES:') :]
    return re.sub(r'//[^\n]*', '', body[: body.index(';')])


def test_the_chart_hash_digests_the_note_data_the_engine_will_read() -> None:
    text = _ssc([(0, [1, 0, 0, 0]), (SLOTS_PER_MEASURE, [0, 1, 0, 0])])
    expected = hashlib.md5(_notes_value(text).encode(), usedforsecurity=False).hexdigest()
    assert f'#CHARTHASH:{expected};' in text


def test_the_chart_hash_ignores_comments_and_carriage_returns() -> None:
    # The parser strips comments before the engine ever sees the note data, and the file is read
    # as text, so neither can reach the digest.
    # A comment leaves its line behind empty rather than removing it.
    plain = '\n\n0000\n0000\n0000\n0000\n'
    assert chart_hash('\n// measure 0\r\n0000\r\n0000\r\n0000\r\n0000\r\n') == chart_hash(plain)


def test_the_chart_hash_is_sensitive_to_the_notes() -> None:
    assert chart_hash('\n1000\n') != chart_hash('\n0100\n')


def test_the_chart_hash_is_a_plain_md5_of_those_bytes() -> None:
    # Pinned against md5sum over the same bytes, so the digest is verified by something other
    # than the code under test.
    assert chart_hash('\n// measure 0\n0000\n1000\n0000\n0001\n') == (
        '1c720b64958384ee0c29614509c399aa'
    )


def test_measures_are_numbered_in_comments() -> None:
    text = _ssc([(0, [1, 0, 0, 0]), (SLOTS_PER_MEASURE, [0, 1, 0, 0])])
    assert '// measure 0' in text
    assert ',  // measure 1' in text


def test_the_radar_carries_one_set_of_values_for_each_player() -> None:
    line = next(
        text for text in _ssc([(0, [1, 0, 0, 0])]).splitlines() if text.startswith('#RADARVALUES:')
    )
    values = line[len('#RADARVALUES:') : -1].split(',')
    assert len(values) == 28
    assert values[:14] == values[14:]


def test_the_radar_counts_what_the_chart_holds() -> None:
    # Two taps together, a freeze head, its tail beside a mine, then one tap.
    # A tail is not struck, so it counts as neither a note nor a tap.
    rows = [(0, [1, 0, 0, 1]), (12, [2, 0, 0, 0]), (24, [3, 0, 0, 5]), (36, [0, 1, 0, 0])]
    stream, _voltage, air, freeze, _chaos, notes, taps, jumps, holds, mines = radar_values(
        rows, 10.0
    )[:10]
    assert notes == 4
    assert taps == 3
    assert jumps == 1
    assert holds == 1
    assert mines == 1
    assert stream == pytest.approx(4 / 10.0 / 7.0)
    assert air == pytest.approx(0.1)
    assert freeze == pytest.approx(0.1)


def test_a_chord_of_three_counts_one_jump_and_one_hand() -> None:
    # The engine counts a row once as a jump however many panels it carries.
    values = radar_values([(0, [1, 1, 1, 0])], 10.0)
    assert values[7] == 1
    assert values[10] == 1


def test_the_radar_of_nothing_is_nothing() -> None:
    assert radar_values([], 0.0) == (0.0,) * 14


def test_ssc_is_written_by_default(tmp_path: Path) -> None:
    audio = tmp_path / 'Song.ogg'
    audio.write_bytes(b'')
    written = write_song(SongMetadata(title='Song'), audio, _TIMING, [('Easy', 3, [])], tmp_path)
    assert written.suffix == '.ssc'
    assert written.read_text().startswith('#VERSION:')


def test_sm_is_written_when_asked_for(tmp_path: Path) -> None:
    audio = tmp_path / 'Song.ogg'
    audio.write_bytes(b'')
    written = write_song(
        SongMetadata(title='Song'), audio, _TIMING, [('Easy', 3, [])], tmp_path, 'sm'
    )
    assert written.suffix == '.sm'
    assert '#VERSION:' not in written.read_text()


def test_a_difficulty_the_format_does_not_know_becomes_an_edit() -> None:
    text = render_ssc(SongMetadata(title='Song'), _TIMING, [('Ultra', 3, [])], 60.0)
    assert '#DIFFICULTY:Edit;' in text


def test_rolls_and_lifts_are_counted_apart_from_taps() -> None:
    # A roll is its own category, and a lift is a note released rather than
    # struck but still counts among the notes.
    rows = [(0, [4, 0, 0, 0]), (12, [3, 0, 0, 0]), (24, [0, 0, 0, 6])]
    values = radar_values(rows, 10.0)
    assert values[5] == 2
    assert values[8] == 0
    assert values[11] == 1
    assert values[12] == 1


def _dwi(rows: list[tuple[int, list[int]]], difficulty: str = 'Challenge') -> str:
    return render_dwi(
        SongMetadata(artist='Someone', music='Song.mp3', title='Song'),
        _TIMING,
        [(difficulty, 9, rows)],
    )


def test_a_bare_character_covers_an_eighth_of_a_measure() -> None:
    # Four beats to a measure at the default step means eight characters.
    assert step_stream([(0, [1, 0, 0, 0])]) == '40000000'


@pytest.mark.parametrize(
    ('panels', 'expected'),
    [
        ([1, 0, 0, 0], '4'),
        ([0, 1, 0, 0], '2'),
        ([0, 0, 1, 0], '8'),
        ([0, 0, 0, 1], '6'),
        ([1, 1, 0, 0], '1'),
        ([0, 1, 0, 1], '3'),
        ([1, 0, 1, 0], '7'),
        ([0, 0, 1, 1], '9'),
        ([0, 1, 1, 0], 'A'),
        ([1, 0, 0, 1], 'B'),
    ],
)
def test_each_panel_pair_has_its_own_character(panels: list[int], expected: str) -> None:
    assert step_stream([(0, panels)]).startswith(expected)


def test_a_chord_of_three_becomes_an_angle_group() -> None:
    # One character carries two panels at most, so more than that has to be spelled out.
    assert step_stream([(0, [1, 1, 1, 0])]).startswith('<428>')


def test_a_freeze_is_marked_where_it_starts_and_closed_by_a_later_step() -> None:
    # The tail is an ordinary step, which the reader consumes to end the freeze.
    assert step_stream([(0, [2, 0, 0, 0]), (24, [3, 0, 0, 0])]) == '4!40004000'


def test_a_measure_of_sixteenths_is_bracketed() -> None:
    stream = step_stream([(0, [1, 0, 0, 0]), (3, [0, 1, 0, 0])])
    assert stream.startswith('(42')
    assert stream.endswith(')')


def test_a_measure_of_twelfths_is_bracketed() -> None:
    stream = step_stream([(0, [1, 0, 0, 0]), (2, [0, 1, 0, 0])])
    assert stream.startswith('[42')
    assert stream.endswith(']')


def test_a_row_finer_than_a_twelfth_falls_back_to_ticks() -> None:
    # Nothing between a twenty-fourth and a hundred and ninety-second divides the grid.
    stream = step_stream([(0, [1, 0, 0, 0]), (1, [0, 1, 0, 0])])
    assert stream.startswith('`4000200')
    assert stream.endswith("'")


def test_mines_and_lifts_are_dropped() -> None:
    # Neither has a spelling in this format.
    assert step_stream([(0, [5, 0, 6, 0])]) == '00000000'


def test_the_gap_is_whole_milliseconds_of_the_opposite_sign() -> None:
    assert '#GAP:48;' in _dwi([(0, [1, 0, 0, 0])])


def test_the_difficulty_is_written_under_its_own_name() -> None:
    assert '#SINGLE:SMANIAC:9:' in _dwi([(0, [1, 0, 0, 0])])
    assert '#SINGLE:BASIC:9:' in _dwi([(0, [1, 0, 0, 0])], 'Easy')
    assert '#SINGLE:EDIT:9:' in _dwi([(0, [1, 0, 0, 0])], 'Ultra')


def test_the_dwi_header_carries_the_song_fields() -> None:
    text = _dwi([(0, [1, 0, 0, 0])])
    for tag in ('#TITLE:Song;', '#ARTIST:Someone;', '#FILE:Song.mp3;', '#BPM:150.000;'):
        assert tag in text


def test_a_dwi_file_is_written_when_asked_for(tmp_path: Path) -> None:
    audio = tmp_path / 'Song.ogg'
    audio.write_bytes(b'')
    written = write_song(
        SongMetadata(title='Song'), audio, _TIMING, [('Easy', 3, [])], tmp_path, 'dwi'
    )
    assert written.suffix == '.dwi'
    assert '#SINGLE:BASIC:3:' in written.read_text()
