"""Shared fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING

from click.testing import CliRunner
import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]  # No stubs are published.

if TYPE_CHECKING:
    from pathlib import Path

SM_TEXT = """#TITLE:Test Song;
#ARTIST:Nobody;
#MUSIC:test.ogg;
#OFFSET:-0.048;
#BPMS:0.000=150.000;
#STOPS:;

#NOTES:
     dance-single:
     :
     Challenge:
     9:
     0,0,0,0,0:
1000
0100
0010
0001
,
1001
0000
0110
0000
;
"""

DWI_TEXT = """#TITLE:Test Song;
#ARTIST:Nobody;
#FILE:test.mp3;
#BPM:150;
#GAP:48;
#SINGLE:MANIAC:9:
4266;
"""

SSC_TEXT = """#VERSION:0.83;
#TITLE:Test Song;
#MUSIC:test.ogg;
#OFFSET:-0.048;
#BPMS:0.000=150.000;

#NOTEDATA:;
#STEPSTYPE:dance-single;
#DIFFICULTY:Hard;
#METER:8;
#NOTES:
1000
0100
0010
0001
;
"""


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def sm_file(tmp_path: Path) -> Path:
    path = tmp_path / 'test.sm'
    path.write_text(SM_TEXT)
    (tmp_path / 'test.ogg').write_bytes(b'')
    return path


@pytest.fixture
def ssc_file(tmp_path: Path) -> Path:
    path = tmp_path / 'test.ssc'
    path.write_text(SSC_TEXT)
    (tmp_path / 'test.ogg').write_bytes(b'')
    return path


@pytest.fixture
def dwi_file(tmp_path: Path) -> Path:
    path = tmp_path / 'test.dwi'
    path.write_text(DWI_TEXT)
    (tmp_path / 'test.mp3').write_bytes(b'')
    return path


@pytest.fixture
def ogg_bytes(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    path = tmp_path_factory.mktemp('audio') / 'seed.ogg'
    sf.write(path, np.zeros(22050, dtype='float32'), 22050, format='OGG', subtype='VORBIS')
    return path.read_bytes()


@pytest.fixture
def mp3_bytes(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    path = tmp_path_factory.mktemp('audio') / 'seed.mp3'
    sf.write(path, np.zeros(22050, dtype='float32'), 22050, format='MP3', subtype='MPEG_LAYER_III')
    return path.read_bytes()
