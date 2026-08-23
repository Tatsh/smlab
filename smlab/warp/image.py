"""
Drawing the beat grid over the waveform it is meant to line up with.

Numbers about tempo are hard to check and easy to argue about. The audio laid out in rows with a
line on every beat is neither: if the grid is right the lines sit on the attacks all the way down,
and if it is wrong they walk off them, which shows both that it drifts and where it starts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image, ImageDraw
import numpy as np

from smlab.audio import load_audio, onset_envelope
from smlab.tempo import PHASE_PARAMS

from .fit import Warp

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from numpy.typing import NDArray

    from .fit import WarpFit

__all__ = ('DEFAULT_ROW_SECONDS', 'write_drift')

DEFAULT_ROW_SECONDS = 8.0
"""
Seconds of audio on each row of the picture.

Eight seconds is about seventeen beats at a dance tempo, which is wide enough to see the grid walk
off the music and narrow enough that a twenty millisecond error is still several pixels.

:meta hide-value:
"""
_WIDTH = 1800
_LEFT = 76
_RIGHT = 16
_ROW = 132
_GAP = 16
_TOP = 58
_BEATS_PER_BAR = 4
_BACKGROUND = (255, 255, 255)
_INK = (28, 28, 44)
_WAVEFORM = (96, 104, 152)
_KICK = (206, 74, 52)
_BEAT = (92, 92, 116)
_FRAME = (196, 196, 212)
_BAR = (120, 150, 226)
_SPLICE = (224, 138, 32)


def _beat_times(
    warps: Sequence[Warp], origin: float, duration: float
) -> Iterator[tuple[int, float]]:
    """
    Walk out the moment of every beat, following each tempo in turn.

    Beats before beat zero are walked out too. An offset of a few seconds is ordinary, and stopping
    at beat zero would leave the whole intro with no grid over it, which reads as the grid starting
    late rather than as there being nothing to draw.

    Parameters
    ----------
    warps : Sequence[Warp]
        Tempo segments in time order, the first taking effect at the start.
    origin : float
        When beat zero happens, in seconds.
    duration : float
        How long the song is, in seconds.

    Yields
    ------
    tuple[int, float]
        Which beat it is, counting from zero at the offset and going negative before it, and when
        it happens in seconds.
    """
    opening = 60.0 / warps[0].bpm
    index = -int(np.ceil(origin / opening))
    at = origin + index * opening
    while at < duration:
        yield index, at
        index += 1
        tempo = next((warp.bpm for warp in reversed(warps) if warp.seconds <= at), warps[0].bpm)
        at += 60.0 / tempo


def _origin(envelope: NDArray[np.float64], frame_rate: float, period: float) -> float:
    """
    Work out where beat zero sits, from the whole song at once.

    Parameters
    ----------
    envelope : :py:class:`~numpy.ndarray`
        Onset strength envelope.
    frame_rate : float
        Envelope frames per second.
    period : float
        Beat period, in seconds.

    Returns
    -------
    float
        When beat zero happens, in seconds.
    """
    weight = np.maximum(envelope - envelope.mean(), 0.0)
    if weight.sum() <= 0.0:
        return 0.0
    times = np.arange(len(envelope)) / frame_rate
    vector = complex(np.sum(weight * np.exp(2j * np.pi * times / period)) / weight.sum())
    return float((np.angle(vector) / (2.0 * np.pi)) % 1.0 * period)


def _bass(samples: NDArray[np.float64], rate: int) -> NDArray[np.float64]:
    """
    Return the low end of the audio, for colouring kicks differently from everything else.

    Parameters
    ----------
    samples : :py:class:`~numpy.ndarray`
        Audio samples.
    rate : int
        Samples per second.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        The same audio with the top taken off it.
    """
    width = max(int(rate / 400), 1)
    running = np.cumsum(np.concatenate([[0.0], samples]))
    averaged = (running[width:] - running[:-width]) / width
    # A running mean is shorter than what it averages, and the tail has to keep lining up with the
    # audio it colours, so the missing end is held rather than left off.
    return np.concatenate([averaged, np.full(len(samples) - len(averaged), averaged[-1])])


def _draw_row(
    draw: ImageDraw.ImageDraw,
    block: NDArray[np.float64],
    low: NDArray[np.float64],
    top: int,
    columns: int,
    loudest: float,
) -> None:
    """
    Draw one row of waveform, redder where the low end dominates.

    Parameters
    ----------
    draw : :py:class:`PIL.ImageDraw.ImageDraw`
        Where to draw.
    block : :py:class:`~numpy.ndarray`
        Audio samples for this row.
    low : :py:class:`~numpy.ndarray`
        The same samples with the top taken off.
    top : int
        Pixel row the waveform starts at.
    columns : int
        How wide the waveform is, in pixels.
    loudest : float
        Loudest sample anywhere in the song, so every row is drawn to the same scale.
    """
    # Each column has to cover exactly its own share of the row. Giving every column the same whole
    # number of samples looks equivalent and is not: the remainder is dropped, so the waveform ends
    # up drawn slightly narrower than the slot the grid is drawn in, and the two slide apart by tens
    # of milliseconds by the right edge, which is the very error the picture exists to show.
    middle = top + _ROW / 2.0
    for column in range(columns):
        first = int(column * len(block) / columns)
        last = max(int((column + 1) * len(block) / columns), first + 1)
        piece = block[first:last]
        if not len(piece):
            break
        peak = float(np.abs(piece).max())
        share = min(float(np.abs(low[first:last]).max()) / (peak + 1e-9), 1.0)
        colour = tuple(
            round(pale + (deep - pale) * share) for pale, deep in zip(_WAVEFORM, _KICK, strict=True)
        )
        reach = peak / loudest * (_ROW / 2.0 - 4.0)
        draw.line([(_LEFT + column, middle - reach), (_LEFT + column, middle + reach)], fill=colour)


def write_drift(
    destination: Path,
    audio: Path,
    bpm: float,
    fit: WarpFit,
    *,
    origin: float | None = None,
    row_seconds: float = DEFAULT_ROW_SECONDS,
) -> float:
    """
    Draw the audio in rows with a line on every beat.

    Parameters
    ----------
    destination : :py:class:`~pathlib.Path`
        PNG file to write.
    audio : :py:class:`~pathlib.Path`
        Audio file to read.
    bpm : float
        Tempo the grid opens at, used when the fit has nothing to say.
    fit : WarpFit
        Tempo segments to lay the grid out by, and jumps to mark.
    origin : float or None
        When beat zero happens, in seconds, or ``None`` to take it from the audio. Worth supplying,
        because the automatic answer is read from an onset envelope and an envelope peaks after the
        attack it belongs to rather than on it.
    row_seconds : float
        Seconds of audio on each row.

    Returns
    -------
    float
        When beat zero was put, in seconds.
    """
    rate = PHASE_PARAMS.sample_rate
    samples = load_audio(audio, sample_rate=rate)
    envelope = onset_envelope(samples, PHASE_PARAMS).astype(np.float64)
    duration = len(samples) / rate
    warps = fit.warps or [Warp(seconds=0.0, bpm=bpm)]
    if origin is None:
        origin = _origin(envelope, PHASE_PARAMS.frame_rate, 60.0 / warps[0].bpm)
    rows = int(np.ceil(duration / row_seconds))
    columns = _WIDTH - _LEFT - _RIGHT
    canvas = Image.new('RGB', (_WIDTH, _TOP + rows * (_ROW + _GAP)), _BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (_LEFT, 14), f'{audio.name}, grid from beat zero at {origin * 1000:.0f} ms', fill=_INK
    )
    draw.text(
        (_LEFT, 34),
        f'{", ".join(f"{w.bpm:.3f} BPM from {w.seconds:g} s" for w in warps)}. Blue lines are '
        f'bars, grey lines are beats, red is the low end. Where the lines walk off the attacks, '
        f'the grid is wrong there.',
        fill=_INK,
    )
    loudest = max(float(np.abs(samples).max()), 1e-9)
    whole = samples.astype(np.float64)
    low = _bass(whole, rate)
    beats = list(_beat_times(warps, origin, duration))
    for row in range(rows):
        start = row * row_seconds
        top = _TOP + row * (_ROW + _GAP)
        draw.text((8, top + _ROW / 2 - 6), f'{start:.0f}s', fill=_INK)
        # The grid goes down before the waveform so the lines never hide the attacks they are there
        # to be judged against.
        for index, at in beats:
            if not start <= at < start + row_seconds:
                continue
            x = _LEFT + (at - start) / row_seconds * columns
            if index % _BEATS_PER_BAR:
                draw.line([(x, top + 12), (x, top + _ROW - 12)], fill=_BEAT, width=1)
                continue
            draw.line([(x, top), (x, top + _ROW)], fill=_BAR, width=2)
            draw.text((x + 4, top + 1), f'{index // _BEATS_PER_BAR + 1}', fill=_BAR)
        first, last = int(start * rate), int((start + row_seconds) * rate)
        _draw_row(draw, whole[first:last], low[first:last], top, columns, loudest)
        draw.rectangle([(_LEFT, top), (_LEFT + columns, top + _ROW)], outline=_FRAME)
        for at in fit.splices:
            if start <= at < start + row_seconds:
                x = _LEFT + (at - start) / row_seconds * columns
                draw.line([(x, top), (x, top + _ROW)], fill=_SPLICE, width=3)
    canvas.save(destination)
    return origin
