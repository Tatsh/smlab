"""Generate StepMania dance-single charts from audio."""

from __future__ import annotations

from .chart import (
    DIFFICULTIES,
    HOLD_HEAD,
    LIFT,
    MINE,
    ROLL_HEAD,
    TAIL,
    TAP,
    Chart,
    NoteRow,
    Simfile,
    normalize_difficulty,
)
from .chart.image import Heading, render_chart, write_chart
from .playability import Style, analyze_rows
from .simfile import SimfileError, load_simfile
from .tempo import estimate_timing
from .timing import TimingData
from .writer import (
    Format,
    SongMetadata,
    render_dwi,
    render_simfile,
    render_ssc,
    safe_directory_name,
    write_song,
)

__all__ = (
    'DIFFICULTIES',
    'HOLD_HEAD',
    'LIFT',
    'MINE',
    'ROLL_HEAD',
    'TAIL',
    'TAP',
    'Chart',
    'Format',
    'Heading',
    'NoteRow',
    'Simfile',
    'SimfileError',
    'SongMetadata',
    'Style',
    'TimingData',
    'analyze_rows',
    'estimate_timing',
    'load_simfile',
    'normalize_difficulty',
    'render_chart',
    'render_dwi',
    'render_simfile',
    'render_ssc',
    'safe_directory_name',
    'write_chart',
    'write_song',
)
__version__ = '0.0.1'
