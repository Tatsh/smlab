"""Command line interface."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast, override
import getpass
import json
import logging
import math
import os

from bascom import setup_logging
import click
import torch

from .audio import audio_duration, load_audio
from .chart import DIFFICULTIES
from .chart.gen import generate_rows, song_features
from .chart.image import Heading, write_chart
from .corpus import EXCLUDED_PACKS, scan_corpus, write_manifest
from .dataset import CODE_BY_CHAR, SUBDIVISIONS_PER_BEAT, beat_features
from .encoder import EncoderConfig
from .generate import (
    CLASSIC_SCALE,
    DEFAULT_BALANCE,
    DEFAULT_SCALE,
    NPS_BY_METER_15,
    NPS_BY_METER_20,
    SCALES,
    GenerationConfig,
    target_nps,
)
from .heads import ChartModel
from .offset import OffsetModel, refine_offset
from .playability import Style, analyze_rows
from .preview import DEFAULT_SAMPLE_LENGTH, PreviewModel, predict_sample_start
from .resources import PREVIEW_ASSET, load_state_dict, load_vocabulary
from .simfile import load_simfile
from .stems import SeparationError, load_separator
from .stems.cache import build_stem_cache
from .tags import apply_tags
from .tempo import estimate_timing
from .timing import BPMSegment, TimingData
from .train import (
    ChartTrainingConfig,
    OffsetTrainingConfig,
    build_envelope_cache,
    train_chart_model,
    train_offset_model,
)
from .vocab import Vocabulary, build_vocabulary
from .warp import DEFAULT_TOLERANCE, Warp, fit_warps, measure_tempo, write_drift
from .weights import (
    CHART_WEIGHTS,
    DEFAULT_REPOSITORY,
    OFFSET_WEIGHTS,
    REPOSITORY_VARIABLE,
    REVISION_VARIABLE,
    WeightsError,
    resolve_weights,
    weights_repository,
)
from .writer import Format, SongMetadata, write_song

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = (
    'DEFAULT_METERS',
    'IMAGE_DIRECTORY',
    'LATENCY_VARIABLE',
    'current_user',
    'default_meter',
    'main',
)

log = logging.getLogger(__name__)


def current_user() -> str:
    """
    Return the name to credit charts to.

    Returns
    -------
    str
        The current user's name, or ``smlab`` when it cannot be determined.
    """
    try:
        return getpass.getuser()
    except (OSError, KeyError):
        return 'smlab'


_DEFAULT_CACHE = Path('cache')
_DEFAULT_CHECKPOINTS = Path('checkpoints')
_STYLES = ('feet', 'hands', 'keyboard')
_ITG_SCALE = 15
IMAGE_DIRECTORY = '.images'
"""
Folder inside a song directory that drawn charts go in.

Named with a leading dot so StepMania's song scanner passes over it rather than trying to read the
pictures as chart assets.

:meta hide-value:
"""
LATENCY_VARIABLE = 'SMLAB_LATENCY'
"""
Environment variable holding a constant latency correction, in seconds.

Playback latency is a property of a setup, not of a song, so it belongs somewhere it can be set once
rather than passed on every run.

:meta hide-value:
"""
DEFAULT_METERS = {
    'Beginner': 2,
    'Easy': 4,
    'Medium': 6,
    'Hard': 9,
    'Challenge': 12,
    'Edit': 9,
}
"""
Difficulty rating used when none is given, on the classic ten-point scale.

The rating conditions the model, so it must follow the difficulty name rather than the order the
difficulties happen to be requested in. Asking for another scale translates these through
:py:func:`default_meter`.

:meta hide-value:
"""


def default_meter(difficulty: str, scale: int = DEFAULT_SCALE) -> int:
    """
    Return the rating a difficulty defaults to on one scale.

    :py:data:`DEFAULT_METERS` is written on the classic scale, so asking for another one has to
    translate. Note rate is the common currency: the answer is whichever rating on the requested
    scale sits closest to the note rate the classic default implies.

    Parameters
    ----------
    difficulty : str
        Difficulty name.
    scale : int
        Rating scale wanted. One of :py:data:`~smlab.generate.SCALES`.

    Returns
    -------
    int
        Rating on that scale.
    """
    classic = DEFAULT_METERS.get(difficulty, 6)
    if scale == CLASSIC_SCALE:
        return classic
    wanted = target_nps(classic, CLASSIC_SCALE)
    ratings = range(1, len(NPS_BY_METER_15 if scale == _ITG_SCALE else NPS_BY_METER_20))
    return min(ratings, key=lambda rating: abs(target_nps(rating, scale) - wanted))


_FIT_WARPS = 'auto'
"""What ``--warp`` stands for when it is given without a value.

:meta hide-value:
"""


class _WarpSpec(click.ParamType['Warp | None']):
    """A tempo change, as the second it happens on and the tempo it changes to."""

    name = 'warp'

    @override
    def convert(
        self, value: str, param: click.Parameter | None, ctx: click.Context | None
    ) -> Warp | None:
        """
        Parse one ``SECONDS:BPM`` argument, or the bare option asking for the changes to be found.

        Parameters
        ----------
        value : str
            The text given on the command line.
        param : :py:class:`click.Parameter` or None
            Parameter being converted.
        ctx : :py:class:`click.Context` or None
            Invocation context.

        Returns
        -------
        Warp or None
            The tempo change, or ``None`` when the option was given without a value and the changes
            are to be fitted instead.
        """
        if value == _FIT_WARPS:
            return None
        seconds, _, tempo = value.partition(':')
        try:
            when, bpm = float(seconds), float(tempo)
        except ValueError:
            self.fail(f'{value!r} is not SECONDS:BPM.', param, ctx)
        if when < 0 or bpm <= 0:
            self.fail(f'{value!r} needs a time of zero or more and a tempo above zero.', param, ctx)
        return Warp(seconds=when, bpm=bpm)


class _DifficultySpec(click.ParamType[tuple[str, int]]):
    """A difficulty name, optionally carrying its own rating."""

    name = 'difficulty'

    @override
    def convert(
        self, value: str, param: click.Parameter | None, ctx: click.Context | None
    ) -> tuple[str, int]:
        """
        Parse one ``NAME`` or ``NAME:METER`` argument.

        Parameters
        ----------
        value : str
            The text given on the command line.
        param : :py:class:`click.Parameter` or None
            Parameter being converted.
        ctx : :py:class:`click.Context` or None
            Invocation context.

        Returns
        -------
        tuple[str, int]
            Difficulty name and rating, where zero means the default for that difficulty on the
            chosen scale.
        """
        name, _, rating = value.partition(':')
        match = next((known for known in DIFFICULTIES if known.lower() == name.lower()), None)
        if match is None:
            self.fail(f'{name!r} is not one of {", ".join(DIFFICULTIES)}.', param, ctx)
        if rating and (not rating.isdigit() or int(rating) < 1):
            self.fail(f'{rating!r} is not a rating.', param, ctx)
        return match, int(rating) if rating else 0


def _load_offset_model(checkpoints: Path | None) -> OffsetModel | None:
    """
    Load the downbeat phase model, or report that it is unavailable.

    Parameters
    ----------
    checkpoints : :py:class:`~pathlib.Path` | None
        Directory of locally trained checkpoints, which wins over a download.

    Returns
    -------
    OffsetModel or None
        The model in evaluation mode, or ``None`` when no weights are to hand. Missing weights are
        not fatal: the tempo estimator supplies an offset of its own, just a worse one.
    """
    try:
        path = resolve_weights(OFFSET_WEIGHTS, checkpoints)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = OffsetModel().to(device)
        model.load_state_dict(torch.load(path, map_location=device)['model'])
    except (WeightsError, OSError, RuntimeError, KeyError, ValueError):
        log.info('No offset model available; keeping the estimated offset.')
        return None
    return model.eval()


def _report_splices(splices: Sequence[float]) -> None:
    """
    Say where the beat was found to jump, if it was.

    Parameters
    ----------
    splices : Sequence[float]
        Times, in seconds, at which the beat moves without changing speed.
    """
    if not splices:
        return
    where = ', '.join(f'{at:g} s' for at in splices)
    click.echo(
        f'The beat jumps at {where}, which is what an edit in the audio looks like rather than a '
        f'change of tempo. Timing near those points is unreliable, and a warp will not put it '
        f'right, because no tempo describes a beat that moves without changing speed.',
        err=True,
    )


def _report_slack(slack: float, wanted: float) -> None:
    """
    Say how far the grid still misses the music by once the tempi have done all they can.

    Parameters
    ----------
    slack : float
        Worst the fitted grid misses by, in seconds.
    wanted : float
        How far the grid was asked to stay within, in seconds.
    """
    if slack <= wanted:
        return
    click.echo(
        f'Even so the grid still misses by up to {slack * 1000:.0f} ms, against the '
        f'{wanted * 1000:.0f} ms asked for. The beat moves in ways no tempo describes, so that '
        f'much is left however the warps are placed.',
        err=True,
    )


def _resolve_timing(
    audio: Path,
    bpm: float,
    offset: float | None,
    shift_beats: float,
    latency: float,
    multiply: float,
    warps: tuple[Warp | None, ...] = (),
    warp_slip: float = DEFAULT_TOLERANCE,
    phase_model: OffsetModel | None = None,
) -> TimingData:
    """
    Settle on the tempo and offset to chart against.

    Parameters
    ----------
    audio : :py:class:`~pathlib.Path`
        Audio file to read.
    bpm : float
        Tempo override, or zero to detect.
    offset : float or None
        Offset override, or None to detect.
    shift_beats : float
        Beats to move beat 0 later by.
    latency : float
        Seconds of constant playback latency to compensate.
    multiply : float
        Factor to scale the tempo by, for when detection lands on the wrong octave.
    warps : tuple[Warp | None, ...]
        Tempo changes, each as the second it happens on and the tempo from then on. A ``None``
        entry asks for the changes to be fitted from the audio instead.
    warp_slip : float
        Seconds the grid may wander from the music before fitting writes another tempo.
    phase_model : OffsetModel or None
        Model used to place the downbeat, or ``None`` to keep the offset the tempo estimator
        produced.

    Returns
    -------
    TimingData
        The timing to use.
    """
    if bpm > 0 and offset is not None:
        timing = TimingData.constant(bpm, offset)
        click.echo(f'Using supplied timing: {bpm:.3f} BPM, offset {offset:+.4f}.')
    else:
        # A supplied tempo has to reach the estimator, not merely replace its answer afterwards.
        # The phase is fitted to whichever tempo the estimator used, so overriding the tempo alone
        # leaves the two describing different grids and the offset wrong by half the drift.
        estimate = estimate_timing(audio, bpm=bpm)
        timing = TimingData.constant(
            estimate['bpm'], offset if offset is not None else estimate['offset']
        )
        if bpm > 0:
            click.echo(
                f'Using {timing.primary_bpm:.3f} BPM, offset {timing.offset:+.4f} fitted to it.'
            )
        else:
            click.echo(
                f'Detected {timing.primary_bpm:.3f} BPM, offset '
                f'{timing.offset:+.4f} (confidence {estimate["confidence"]:.2f}).'
            )
    # The tempo estimator picks its phase by taking the loudest point of a folded envelope, which is
    # not where a downbeat is. Given the tempo, the phase model decides that separately and far
    # better, so it replaces the offset whenever one was not supplied outright.
    if phase_model is not None and offset is None:
        found, weight = refine_offset(phase_model, audio, timing.primary_bpm)
        timing = TimingData.constant(timing.primary_bpm, found)
        click.echo(f'Placed the downbeat at offset {found:+.4f} (confidence {weight:.2f}).')
    if not math.isclose(multiply, 1.0):
        # Scaling the tempo leaves beat zero where it is, so the offset carries over untouched and
        # only the grid spacing changes.
        timing = TimingData.constant(timing.primary_bpm * multiply, timing.offset)
        click.echo(f'Tempo scaled by {multiply:g} to {timing.primary_bpm:.3f} BPM.')
    if shift_beats:
        timing = timing.shifted(shift_beats * 60.0 / timing.primary_bpm)
        click.echo(f'Shifted beat 0 by {shift_beats:+g} beats, offset now {timing.offset:+.4f}.')
    placed = tuple(warp for warp in warps if warp is not None)
    # Splices are worth knowing about whether or not anything is being warped, because they say the
    # detected tempo is describing a beat that jumps, which no amount of tempo is going to fix.
    fit = fit_warps(audio, timing.primary_bpm, tolerance=warp_slip)
    _report_splices(fit.splices)
    if len(placed) < len(warps):
        click.echo('Warping is experimental. Check the result against the audio.')
        if len(fit.warps) <= 1:
            click.echo(f'Tempo holds within {warp_slip * 1000:.0f} ms, so nothing was warped.')
        else:
            click.echo(f'Fitted {len(fit.warps)} tempo segments within {warp_slip * 1000:.0f} ms.')
            _report_slack(fit.slack, warp_slip)
            placed = (*fit.warps, *placed)
    for seconds, tempo in sorted(placed):
        # Each marker is placed at the beat the timing built so far puts that moment on, and the
        # beat is left exactly where it lands. Rounding it to a whole beat would move the change by
        # up to half a beat, which is the same kind of tidying that made the tempo wrong to begin
        # with.
        if (beat := timing.beat_at_time(seconds)) <= 0:
            # Nothing is charted before beat zero, so a marker there is simply the opening tempo.
            timing = TimingData(
                bpms=(BPMSegment(0.0, tempo), *timing.bpms[1:]),
                offset=timing.offset,
                stops=timing.stops,
            )
            click.echo(f'Opening tempo is {tempo:.3f} BPM.')
            continue
        timing = TimingData(
            bpms=(*timing.bpms, BPMSegment(beat, tempo)), offset=timing.offset, stops=timing.stops
        )
        click.echo(f'Warped to {tempo:.3f} BPM at {seconds:g} s, which is beat {beat:.3f}.')
    latency = latency or float(os.environ.get(LATENCY_VARIABLE, '0') or 0)
    if latency:
        timing = timing.shifted(latency)
        click.echo(
            f'Compensated {1000 * latency:+.0f} ms of latency, offset now {timing.offset:+.4f}.'
        )
    return timing


@click.group(context_settings={'help_option_names': ['-h', '--help']})
@click.option('-d', '--debug', is_flag=True, help='Enable verbose output.')
def main(*, debug: bool = False) -> None:
    """Generate StepMania dance-single charts from audio."""
    setup_logging(debug=debug, loggers={'smlab': {}})


@main.command()
@click.argument('songs', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    '-o',
    '--output',
    type=click.Path(path_type=Path),
    default=_DEFAULT_CACHE / 'manifest.json',
    help='Where to write the manifest.',
)
@click.option('-w', '--workers', default=8, help='Number of worker processes.')
@click.option(
    '-x',
    '--exclude',
    multiple=True,
    help='Pack name to leave out, on top of the defaults. Repeatable.',
)
def scan(songs: Path, output: Path, workers: int, exclude: tuple[str, ...]) -> None:
    """Index a Songs tree into a ground-truth timing manifest."""
    excluded = EXCLUDED_PACKS | frozenset(exclude)
    click.echo(f'Excluding {len(excluded)} packs.')
    count = write_manifest(scan_corpus(songs, workers=workers, excluded=excluded), output)
    click.echo(f'Wrote {count} records to {output}.')


@main.command()
@click.option(
    '-c',
    '--cache-dir',
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=_DEFAULT_CACHE / 'stems',
    help='Feature cache to scan.',
)
@click.option(
    '-o',
    '--output',
    type=click.Path(path_type=Path),
    default=_DEFAULT_CACHE / 'vocabulary.json',
    help='Where to write the vocabulary.',
)
@click.option('-l', '--limit', default=512, help='Largest vocabulary to build.')
def vocab(cache_dir: Path, output: Path, limit: int) -> None:
    """Collect the note-row patterns the corpus actually uses."""
    vocabulary = build_vocabulary(cache_dir, limit=limit)
    vocabulary.save(output)
    click.echo(f'Wrote {len(vocabulary)} patterns to {output}.')


@main.command('envelopes')
@click.option(
    '-m',
    '--manifest',
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_CACHE / 'manifest.json',
    help='Manifest to read.',
)
@click.option(
    '-o',
    '--output',
    type=click.Path(path_type=Path),
    default=_DEFAULT_CACHE / 'envelopes',
    help='Cache directory to fill.',
)
def envelopes_command(manifest: Path, output: Path) -> None:
    """Build the banded onset envelopes the offset model trains on."""
    records = json.loads(manifest.read_text(encoding='utf-8'))
    count = build_envelope_cache(records, output)
    click.echo(f'Cached {count} songs in {output}.')


@main.command('train-offset')
@click.option(
    '-c',
    '--cache-dir',
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=_DEFAULT_CACHE / 'envelopes',
    help='Envelope cache to train on.',
)
@click.option(
    '-o',
    '--output',
    type=click.Path(path_type=Path),
    default=_DEFAULT_CHECKPOINTS,
    help='Directory to write the checkpoint into.',
)
@click.option('-e', '--epochs', default=60, help='Passes over the song list.')
@click.option('-b', '--batch-size', default=64, help='Profiles per optimizer step.')
def train_offset_command(cache_dir: Path, output: Path, epochs: int, batch_size: int) -> None:
    """Train the model that places the downbeat."""
    output.mkdir(parents=True, exist_ok=True)
    scores = train_offset_model(
        cache_dir,
        output / OFFSET_WEIGHTS,
        OffsetTrainingConfig(batch_size=batch_size, epochs=epochs),
    )
    for name, value in scores.items():
        click.echo(f'  {name}: {value:.4f}')


@main.command('stems')
@click.option(
    '-m',
    '--manifest',
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_CACHE / 'manifest.json',
    help='Manifest to read.',
)
@click.option(
    '-o',
    '--output',
    type=click.Path(path_type=Path),
    default=_DEFAULT_CACHE / 'stems',
    help='Cache directory to fill.',
)
def stems_command(manifest: Path, output: Path) -> None:
    """Separate every song into stems and build the features training reads."""
    records = json.loads(manifest.read_text(encoding='utf-8'))
    done = failed = 0
    for _, ok in build_stem_cache(records, output):
        done += ok
        failed += not ok
        if (done + failed) % 100 == 0:
            click.echo(f'  {done + failed} of {len(records)}.')
    click.echo(f'Cached {done} of {len(records)} songs in {output}; {failed} failed.')


@main.command()
@click.option(
    '-c',
    '--cache-dir',
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=_DEFAULT_CACHE / 'stems',
    help='Stem feature cache to train on.',
)
@click.option(
    '-m',
    '--manifest',
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_CACHE / 'manifest.json',
    help='Manifest describing the corpus.',
)
@click.option(
    '-v',
    '--vocabulary',
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_CACHE / 'vocabulary.json',
    help='Pattern vocabulary to use.',
)
@click.option(
    '-o',
    '--output',
    type=click.Path(path_type=Path),
    default=_DEFAULT_CHECKPOINTS,
    help='Directory to write checkpoints into.',
)
@click.option('-e', '--epochs', default=40, help='Passes over the chart list.')
@click.option('-b', '--batch-size', default=8, help='Windows per optimizer step.')
@click.option('-w', '--workers', default=6, help='Loader worker processes.')
@click.option('--limit', default=0, help='Cap on charts loaded, for quick runs.')
def train(
    cache_dir: Path,
    manifest: Path,
    vocabulary: Path,
    output: Path,
    epochs: int,
    batch_size: int,
    workers: int,
    limit: int,
) -> None:
    """Train the shared encoder and both heads."""
    records = json.loads(manifest.read_text(encoding='utf-8'))
    config = ChartTrainingConfig(batch_size=batch_size, epochs=epochs, limit=limit, workers=workers)
    output.mkdir(parents=True, exist_ok=True)
    scores = train_chart_model(
        cache_dir, records, Vocabulary.load(vocabulary), output / CHART_WEIGHTS, config
    )
    for name, value in scores.items():
        click.echo(f'  {name}: {value:.4f}')


def _detect_sample_start(checkpoints: Path | None, audio: Path, timing: TimingData) -> float:
    """
    Choose a preview start, falling back to a fixed fraction of the song.

    Parameters
    ----------
    checkpoints : :py:class:`~pathlib.Path` | None
        Directory of locally trained checkpoints, or ``None`` for the bundled models.
    audio : :py:class:`~pathlib.Path`
        The audio file, re-read because the preview model uses its own features.
    timing : TimingData
        Timing for the song.

    Returns
    -------
    float
        Preview start in seconds.
    """
    features = beat_features(load_audio(audio), timing)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PreviewModel()
    try:
        model.load_state_dict(load_state_dict(PREVIEW_ASSET, checkpoints, device))
    except (OSError, RuntimeError, ValueError):
        log.info('No preview model available; falling back to a fixed fraction of the song.')
        return 0.43 * timing.time_at_beat(features.shape[0] / SUBDIVISIONS_PER_BEAT)
    return predict_sample_start(model.to(device).eval(), features, timing)


def _load_chart_model(
    checkpoints: Path | None, vocabulary_path: Path | None
) -> tuple[ChartModel, Vocabulary, torch.device]:
    """
    Load the trained chart model and its vocabulary.

    Parameters
    ----------
    checkpoints : :py:class:`~pathlib.Path` | None
        Directory of locally trained checkpoints, which wins over a download.
    vocabulary_path : :py:class:`~pathlib.Path` | None
        Pattern vocabulary file, or ``None`` for the bundled one.

    Returns
    -------
    tuple[ChartModel, Vocabulary, :py:class:`~torch.device`]
        The model in evaluation mode, its vocabulary, and the compute device.

    Raises
    ------
    click.Abort
        If the weights cannot be found or do not load.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        path = resolve_weights(CHART_WEIGHTS, checkpoints)
        blob = torch.load(path, map_location=device, weights_only=False)
        vocabulary = load_vocabulary(vocabulary_path)
        model = ChartModel(blob['vocabulary'], EncoderConfig(), blob['prior'])
        model.load_state_dict(blob['model'])
    except (WeightsError, OSError, RuntimeError, ValueError, KeyError) as error:
        click.echo(f'Could not load the chart model: {error}', err=True)
        raise click.Abort from error
    click.echo(
        f'Loaded {sum(p.numel() for p in model.parameters()) / 1e6:.0f} M parameters '
        f'from {path.name} on {device.type}.'
    )
    return model.to(device).eval(), vocabulary, device


@main.command()
@click.argument('audio', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    '-o',
    '--output-dir',
    type=click.Path(file_okay=False, path_type=Path),
    default=Path(),
    help='Directory the song folder is created inside, usually a pack.',
)
@click.option('-T', '--title', default='', help='Song title, which also names the song folder.')
@click.option('-A', '--artist', default='', help='Song artist.')
@click.option('-S', '--subtitle', default='', help='Song subtitle.')
@click.option('-G', '--genre', default='', help='Song genre.')
@click.option('-C', '--credit', default='', help='Chart author. Defaults to the current user.')
@click.option('--title-translit', default='', help='Romanised title.')
@click.option('--subtitle-translit', default='', help='Romanised subtitle.')
@click.option('--artist-translit', default='', help='Romanised artist.')
@click.option('--banner', default='', help='Banner image file name.')
@click.option('--background', default='', help='Background image file name.')
@click.option('--cdtitle', default='', help='Mix logo file name.')
@click.option(
    '--sample-start',
    default=None,
    type=float,
    help='Preview start in seconds. Detected when omitted.',
)
@click.option('--sample-length', default=DEFAULT_SAMPLE_LENGTH, help='Preview length in seconds.')
@click.option(
    '-D',
    '--difficulty',
    type=_DifficultySpec(),
    multiple=True,
    default=('Easy', 'Medium', 'Hard'),
    help='Difficulty to generate, as NAME or NAME:RATING. Repeat for several.',
)
@click.option(
    '-m',
    '--meter',
    default=0,
    help='Rating for every difficulty that does not carry its own. Zero uses the defaults.',
)
@click.option(
    '-s',
    '--style',
    type=click.Choice(_STYLES),
    default='feet',
    help='Physical constraints to respect.',
)
@click.option('--bpm', default=0.0, help='Override the detected tempo.')
@click.option(
    '--bpm-multiplier',
    default=1.0,
    help='Scale the tempo. Use 2 when detection halved it, or 0.5 when it doubled it.',
)
@click.option('--offset', default=None, type=float, help='Override the detected offset.')
@click.option(
    '--warp',
    'warps',
    multiple=True,
    is_flag=False,
    flag_value=_FIT_WARPS,
    type=_WarpSpec(),
    help='Experimental. Change tempo partway through, as SECONDS:BPM. Repeatable. Pass it without '
    'a value to fit the changes from the audio. See the drift command.',
)
@click.option(
    '--warp-slip',
    default=DEFAULT_TOLERANCE,
    help='Seconds the grid may wander from the music before a fitted warp writes another tempo.',
    show_default=True,
)
@click.option(
    '--shift-beats',
    default=0.0,
    help='Move beat 0 later by this many beats. Use 0.5 when the grid locked onto the off-beat.',
)
@click.option(
    '--latency',
    default=0.0,
    help=(
        'Seconds of constant playback latency to compensate, applied to every chart. '
        f'Also settable as {LATENCY_VARIABLE}.'
    ),
)
@click.option(
    '-i',
    '--image/--no-image',
    default=False,
    help='Also draw each chart as a picture next to the simfile.',
)
@click.option('--svg/--png', default=False, help='Draw charts as SVG instead of PNG.')
@click.option(
    '-f',
    '--format',
    'fmt',
    default='ssc',
    help='Simfile format to write.',
    show_default=True,
    type=click.Choice(('dwi', 'sm', 'ssc')),
)
@click.option(
    '--balance',
    default=DEFAULT_BALANCE,
    help=(
        'How hard to pull the four panels towards equal use. Zero leaves the model alone, '
        'which leans heavily on the middle two.'
    ),
)
@click.option(
    '--crossovers',
    type=float,
    default=None,
    help=(
        'Share of streamed notes that may land on a crossed foot. Omit for the measured '
        'default, or pass 0 to bar crossovers entirely. A sixteenth stream is held to a '
        'fraction of whatever applies.'
    ),
)
@click.option(
    '--triplets/--no-triplets',
    default=False,
    help='Also place notes on the twelfth grid. Off by default; 74%% of charts never do.',
)
@click.option('--mines/--no-mines', default=False, help='Allow mines. Off by default.')
@click.option(
    '--rolls/--no-rolls',
    default=False,
    help='Allow roll notes. Off by default; 98 per cent of real charts have none.',
)
@click.option(
    '--holds',
    default=0.04,
    help='Largest share of notes that may be freezes. Real charts sit near 0.04.',
)
@click.option('--density', default=1.0, help='Multiplier on how many steps to place.')
@click.option('--temperature', default=0.9, help='Sampling temperature for patterns.')
@click.option(
    '--scale',
    type=click.Choice([str(value) for value in SCALES]),
    default=str(DEFAULT_SCALE),
    help='Rating scale the ratings are on. Classic DDR 10, In The Groove 15, X-era DDR 20.',
)
@click.option(
    '--nps',
    default=0.0,
    help='Aim for this note rate instead of the one the rating implies. Zero uses the rating.',
)
@click.option('--seed', default=0, help='Seed for sampling.')
@click.option(
    '--weights-repo',
    default=None,
    help=f'Repository to download weights from. Also settable as {REPOSITORY_VARIABLE}.',
)
@click.option(
    '--weights-revision',
    default=None,
    help=f'Pin the weights to a revision. Also settable as {REVISION_VARIABLE}.',
)
@click.option(
    '-c',
    '--checkpoints',
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help='Directory of locally trained checkpoints. Uses the bundled models when omitted.',
)
@click.option(
    '-v',
    '--vocabulary',
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help='Pattern vocabulary to use. Uses the bundled one when omitted.',
)
def generate(  # noqa: PLR0917
    audio: Path,
    output_dir: Path,
    title: str,
    artist: str,
    subtitle: str,
    genre: str,
    credit: str,
    title_translit: str,
    subtitle_translit: str,
    artist_translit: str,
    banner: str,
    background: str,
    cdtitle: str,
    sample_start: float | None,
    sample_length: float,
    difficulty: tuple[tuple[str, int], ...],
    meter: int,
    style: str,
    bpm: float,
    bpm_multiplier: float,
    offset: float | None,
    warps: tuple[Warp | None, ...],
    warp_slip: float,
    shift_beats: float,
    latency: float,
    image: bool,  # noqa: FBT001
    svg: bool,  # noqa: FBT001
    balance: float,
    crossovers: float | None,
    fmt: Format,
    triplets: bool,  # noqa: FBT001
    mines: bool,  # noqa: FBT001
    rolls: bool,  # noqa: FBT001
    holds: float,
    density: float,
    temperature: float,
    scale: str,
    nps: float,
    seed: int,
    weights_repo: str | None,
    weights_revision: str | None,
    checkpoints: Path | None,
    vocabulary: Path | None,
) -> None:
    # A Raises section for click.Abort would show up verbatim in --help.
    """Generate a song folder holding a simfile and a copy of the audio."""  # noqa: DOC501
    if weights_repo:
        os.environ[REPOSITORY_VARIABLE] = weights_repo
    if weights_revision:
        os.environ[REVISION_VARIABLE] = weights_revision
    model, vocab, device = _load_chart_model(checkpoints, vocabulary)
    timing = _resolve_timing(
        audio,
        bpm,
        offset,
        shift_beats,
        latency,
        bpm_multiplier,
        warps,
        warp_slip,
        _load_offset_model(checkpoints),
    )
    try:
        separator = load_separator(device)
    except SeparationError as error:
        click.echo(f'{error}', err=True)
        raise click.Abort from error
    click.echo('Separating into ' + ', '.join(separator.sources) + '.')
    features = song_features(separator, audio, timing, device)
    charts = []
    for index, (name, own_rating) in enumerate(difficulty):
        rating = own_rating or meter or default_meter(name, int(scale))
        if rating > int(scale):
            click.echo(
                f'  Rating {rating} is above the {scale}-point scale, so it means the same '
                f'as {scale}. Pass --scale '
                f'{min((s for s in SCALES if s >= rating), default=SCALES[-1])}, '
                f'or set --nps directly.',
                err=True,
            )
        config = GenerationConfig(
            balance=balance,
            crossovers=crossovers,
            density=density,
            difficulty=name,
            meter=rating,
            holds=holds,
            mines=mines,
            nps=nps,
            rolls=rolls,
            scale=int(scale),
            seed=seed + index,
            style=cast('Style', style),
            temperature=temperature,
            triplets=triplets,
        )
        rows = generate_rows(model, vocab, features, timing, config, device)
        report = analyze_rows([(timing.time_at_beat(slot / 12), codes) for slot, codes in rows])
        click.echo(
            f'  {name:10} meter {rating:2}/{scale}  {len(rows):5} rows  style={report.style}  '
            f'target {config.rate:.2f} nps, sustained {report.sustained_nps:.1f} nps'
        )
        charts.append((name, rating, rows))
    if sample_start is None:
        sample_start = _detect_sample_start(checkpoints, audio, timing)
        click.echo(f'Preview starts at {sample_start:.1f} s for {sample_length:.0f} s.')
    metadata = SongMetadata(
        artist=artist,
        artist_translit=artist_translit,
        background=background,
        banner=banner,
        cdtitle=cdtitle,
        credit=credit or current_user(),
        genre=genre,
        sample_length=sample_length,
        sample_start=sample_start,
        subtitle=subtitle,
        subtitle_translit=subtitle_translit,
        title=title,
        title_translit=title_translit,
    )
    metadata = apply_tags(metadata, audio)
    if not metadata.title:
        metadata = replace(metadata, title=audio.stem)
    click.echo(
        f'Metadata: {metadata.title!r} by {metadata.artist or "unknown"}'
        f'{f" [{metadata.genre}]" if metadata.genre else ""}.'
    )
    simfile = write_song(metadata, audio, timing, charts, output_dir, fmt, audio_duration(audio))
    click.echo(f'Wrote {simfile.parent} containing {simfile.name} and the audio.')
    for name, rating, rows in charts if image else ():
        pictures = simfile.parent / IMAGE_DIRECTORY
        pictures.mkdir(exist_ok=True)
        picture = pictures / f'{simfile.stem} {name}.{"svg" if svg else "png"}'
        write_chart(picture, rows, Heading(metadata.title, name, rating, timing.primary_bpm))
        click.echo(f'Drew {IMAGE_DIRECTORY}/{picture.name}.')


@main.command()
@click.option(
    '-c',
    '--checkpoints',
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=_DEFAULT_CHECKPOINTS,
    help='Directory holding the trained weights.',
)
@click.option(
    '-r',
    '--repository',
    default=None,
    help=f'Repository to upload to. Defaults to {DEFAULT_REPOSITORY}.',
)
@click.option('-n', '--dry-run', is_flag=True, help='List what would be uploaded and stop.')
def publish(checkpoints: Path, repository: str | None, *, dry_run: bool) -> None:
    """Upload trained weights so an installed copy can download them."""  # noqa: DOC501
    target = repository or weights_repository()
    present = [(name, checkpoints / name) for name in (CHART_WEIGHTS, OFFSET_WEIGHTS)]
    missing = [name for name, path in present if not path.is_file()]
    if missing:
        click.echo(f'{checkpoints} is missing {", ".join(missing)}.', err=True)
        raise click.Abort
    for name, path in present:
        click.echo(f'  {name}  {path.stat().st_size / 1e6:.1f} MB')
    if dry_run:
        click.echo(f'Would upload the above to {target}.')
        return
    click.echo(f'Uploading to {target}. This needs a write token; see `huggingface-cli login`.')
    try:
        from huggingface_hub import HfApi  # noqa: PLC0415

        api = HfApi()
        for name, path in present:
            api.upload_file(path_or_fileobj=path, path_in_repo=name, repo_id=target)
            click.echo(f'Uploaded {name}.')
    except Exception as error:
        click.echo(f'Upload failed: {error}', err=True)
        raise click.Abort from error


@main.command('drift')
@click.argument('audio', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option('--bpm', default=0.0, help='Tempo to measure against. Detected when omitted.')
@click.option(
    '--slip',
    default=0.020,
    help='Seconds of gain or loss across a stretch before it is worth a warp marker.',
    show_default=True,
)
@click.option(
    '-i',
    '--image',
    type=click.Path(dir_okay=False, path_type=Path),
    help='Also draw the beat against the grid to this PNG.',
)
@click.option(
    '--offset',
    default=None,
    type=float,
    help='Offset to draw the grid from, where beat 0 falls at minus this. Detected when omitted.',
)
def drift_command(
    audio: Path, bpm: float, slip: float, image: Path | None, offset: float | None
) -> None:
    """
    Report how a song's tempo wanders, and where warp markers would go. Experimental.

    A chart is written against one grid, so a song whose tempo moves can only be charted correctly
    by saying where it moves. Pass what this prints back to `generate` as --warp. Where the beat
    jumps rather than changing speed, which is what an edit in the audio looks like, it says so and
    no warp will help.
    """  # noqa: DOC501
    if bpm <= 0:
        estimate = estimate_timing(audio)
        bpm = estimate['bpm']
        click.echo(f'Measuring against the detected {bpm:.3f} BPM.')
    if not (readings := measure_tempo(audio, bpm)):
        click.echo(f'{audio} is too short or too quiet to track a beat through.', err=True)
        raise click.Abort
    tempi = [reading.bpm for reading in readings]
    click.echo(f'{min(tempi):.3f} to {max(tempi):.3f} BPM across {len(readings)} stretches.')
    for reading in readings:
        mark = ' <- warp' if abs(reading.slip) >= slip else ''
        click.echo(
            f'  {reading.seconds:7.1f} s  {reading.bpm:8.3f} BPM  '
            f'slip {reading.slip * 1000:+7.1f} ms{mark}'
        )
    fit = fit_warps(audio, bpm, tolerance=slip)
    _report_splices(fit.splices)
    if image is not None:
        write_drift(image, audio, bpm, fit, origin=None if offset is None else -offset)
        click.echo(f'Wrote the beat against the grid to {image}.')
    if len(fit.warps) <= 1:
        settled = fit.warps[0].bpm if fit.warps else sum(tempi) / len(tempi)
        click.echo(f'Steady enough for one tempo of {settled:.3f} BPM.')
        return
    click.echo(
        f'Holding the grid within {slip * 1000:.0f} ms needs {len(fit.warps)} tempo segments:'
    )
    for warp in fit.warps:
        click.echo(f'  {warp.seconds:7.1f} s  {warp.bpm:8.3f} BPM')
    later = ' '.join(f'--warp {warp.seconds:.0f}:{warp.bpm:.3f}' for warp in fit.warps[1:])
    click.echo(f'Generate with: --bpm {fit.warps[0].bpm:.3f} {later}')
    _report_slack(fit.slack, slip)
    click.echo('Warping is experimental. Listen to the result before trusting it.')


@main.command('image')
@click.argument('simfile', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    '-o',
    '--output',
    type=click.Path(path_type=Path),
    default=None,
    help=f'Directory to write into. Defaults to {IMAGE_DIRECTORY}/ beside the simfile.',
)
@click.option('--svg/--png', default=False, help='Draw as SVG instead of PNG.')
def image_command(simfile: Path, output: Path | None, *, svg: bool) -> None:
    """Draw an existing simfile's charts as pictures."""  # noqa: DOC501
    parsed = load_simfile(simfile)
    charts = parsed.singles()
    if not charts or parsed.timing is None:
        click.echo(f'{simfile} holds no dance-single charts with timing.', err=True)
        raise click.Abort
    tempo = parsed.timing.primary_bpm
    suffix = 'svg' if svg else 'png'
    for chart in charts:
        rows = [
            (round(beat * SUBDIVISIONS_PER_BEAT), [CODE_BY_CHAR.get(char, 0) for char in row[:4]])
            for beat, row in chart.rows()
        ]
        if output is not None and len(charts) == 1:
            destination = output
        else:
            folder = output or simfile.parent / IMAGE_DIRECTORY
            folder.mkdir(parents=True, exist_ok=True)
            destination = folder / f'{simfile.stem} {chart.difficulty}.{suffix}'
        write_chart(
            destination,
            rows,
            Heading(
                parsed.title or simfile.stem,
                chart.difficulty,
                chart.meter,
                tempo,
            ),
        )
        click.echo(f'Drew {destination}.')


@main.command()
@click.argument('simfile', type=click.Path(exists=True, dir_okay=False, path_type=Path))
def analyze(simfile: Path) -> None:
    """Report timing and physical demands of an existing simfile."""  # noqa: DOC501
    parsed = load_simfile(simfile)
    if (timing := parsed.timing) is None:
        click.echo('No usable timing.', err=True)
        raise click.Abort
    low, high = timing.bpm_range()
    click.echo(f'{parsed.title or simfile.stem} - {parsed.artist}')
    click.echo(
        f'  format {parsed.file_format}, offset {timing.offset:+.4f} '
        f'(declared: {parsed.offset_declared}), BPM {low:.2f}-{high:.2f}, '
        f'{len(timing.stops)} stops'
    )
    for chart in parsed.singles():
        rows = [
            (timing.time_at_beat(beat), [CODE_BY_CHAR.get(character, 0) for character in columns])
            for beat, columns in chart.rows()
        ]
        report = analyze_rows(rows)
        click.echo(
            f'  {chart.difficulty:10} meter {chart.meter:2}  {len(rows):5} rows  '
            f'style={report.style:8} peak {report.burst_nps:5.1f} nps  '
            f'sustained {report.sustained_nps:5.1f} nps  chords {report.chord_rows}  '
            f'crossovers {report.crossovers}'
        )
        for reason in report.reasons:
            click.echo(f'      {reason}')
