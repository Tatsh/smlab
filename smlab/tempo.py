"""
Joint tempo and offset estimation from audio.

Tempo and offset are recovered together rather than in sequence. For a track at a fixed tempo the
beat grid is ``t_k = phase + k * 60 / bpm``, so one objective determines both quantities, and the
phase is exactly what ``#OFFSET`` encodes.

The tempo comes from **folding** the onset envelope onto a candidate beat period and taking the mean
onset strength at the beat phase. That picks the right *octave*: a grid at twice the true tempo
necessarily samples the weak off-beats as well, which drags its mean down, whereas a grid at half
the true tempo merely ties and is separated by a log-normal tempo prior.

Precision matters more than it first appears — an error of 0.05 BPM at 150 BPM drifts by 67 ms
across a three-minute track, worse than the offset error being chased — but the fold is already
precise enough. Its grid steps by 0.2 per cent, 0.28 BPM at 140, and measured against songs whose
tempo it identifies correctly it lands within 0.192 BPM in the worst case and 0.064 at the median,
against a half-BPM tolerance.

An earlier version sharpened it anyway, taking the Fourier peak within one per cent of the fold's
pick. Over two held-out halves that stage fixed no song and broke fourteen, dropping accuracy from
98.6 to 84.3 per cent: with nothing left to gain it could only wander off a tempo that was already
right. It is gone.

The phase, which is what ``#OFFSET`` encodes, is recovered here only as a fallback.
:py:mod:`smlab.offset` does it properly.

One tempo is assumed for the whole song, and an attempt to lift that did not survive measurement. Of
3842 corpus songs, 60 per cent declare one tempo and a further 17 per cent declare several within
five per cent of each other, which is an author correcting drift rather than the music changing. Of
the 872 that genuinely change, 34.6 per cent hold two clean tempi and 19.3 per cent three or four;
the rest accelerate continuously or use tempi outside 40 to 400 BPM as a scroll-speed trick.

Estimating the tempo in overlapping windows and grouping the ones that agree false-splits half of
all constant-tempo songs, because a 25-second window resolves the tempo only 88 per cent of the time
and a song holds twenty of them. Deciding the windows jointly, by a Viterbi pass over the tempo grid
with a penalty for changing, is markedly better: at a 5 per cent false-split rate it finds a change
in 40 per cent of songs that have one, against 10 per cent for grouping. It is still not enough. On
the most favourable population — two to four clean tempi, no gimmicks — it recovers every tempo and
no extras in 5.3 per cent of cases and returns a partial answer in 84.

A partial answer is the worst outcome available. A chart pinned to one wrong tempo drifts
predictably and ``--bpm`` fixes it; a chart whose grid is right for two minutes and wrong for one
cannot be corrected by any single number. Weighed against constant-tempo songs being 77 per cent of
the corpus, every operating point measured came out net harmful, so the attempt was removed rather
than shipped behind a flag.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple
import logging

import numpy as np

from .audio import DEFAULT_SAMPLE_RATE, OnsetParams, load_audio, onset_envelope

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from .typing import TimingEstimate

__all__ = (
    'MAX_BPM',
    'MIN_BPM',
    'ONSET_LATENCY_SECONDS',
    'PHASE_PARAMS',
    'TEMPO_PARAMS',
    'Envelopes',
    'estimate_tempo',
    'estimate_timing',
    'estimate_timing_from_envelopes',
    'snap_bpm',
)

log = logging.getLogger(__name__)

MIN_BPM = 60.0
"""Lowest tempo considered.

:meta hide-value:
"""
MAX_BPM = 300.0
"""Highest tempo considered.

:meta hide-value:
"""
_PRIOR_CENTRE_BPM = 140.0
_PRIOR_WIDTH_OCTAVES = 0.8
_COARSE_STEP = 1.002
"""
Ratio between adjacent tempi on the coarse search grid.

Folding demands that a candidate stay phase-aligned for the whole span being folded, and the drift
it accumulates is the span times the relative tempo error. This step size is therefore only valid
over a short excerpt; applied to a full track it would smear every profile across an entire beat.
"""
_COARSE_EXCERPT_SECONDS = 25.0
"""
Span folded during the coarse tempo search.

Shorter is better here, which is counter-intuitive until the drift arithmetic is written down: a
candidate must stay phase-aligned across the whole span, and the drift it accumulates is the span
times the relative tempo error. A short span tolerates a coarser grid, so it resolves the right
neighbourhood more often, and the grid is fine enough that the neighbourhood is the answer.

Measured across a training split, accuracy within one beat per minute rises from 60% at whole-track
to 84% here, with a clear interior maximum: 76% at ten seconds, 84% at twenty-five, 72% at sixty.
"""
_SNAP_TOLERANCE_BPM = 0.25
"""
Distance within which a tempo is snapped to a round value.

Most simfiles in the corpus are authored at whole or half beats per minute, so an estimate that
lands just beside one almost certainly belongs on it.

Measured over 180 songs with a declared constant tempo, widening this from 0.15 to 0.25 takes exact
agreement from 53.9 to 57.8 per cent, and nothing further is gained past 0.25. The narrower window
missed cases like an estimate of 169.347, which sits 0.153 away from 169.5 and drifts audibly over a
song.
"""
_SNAP_DIVISOR = 2.0
"""Snapping resolution, where two gives whole and half beats per minute."""
_FOLD_BINS = 64
"""Phase bins used when folding the envelope for coarse tempo scoring."""
_PHASE_SUBDIVISIONS = 8
"""Phase search points per envelope frame, giving sub-frame resolution."""
_MIN_FOLD_FRAMES = 64
_EMPTY: TimingEstimate = {'bpm': 0.0, 'confidence': 0.0, 'offset': 0.0}
ONSET_LATENCY_SECONDS = 0.0046
"""
Constant delay between a transient and its detected onset.

Spectral flux reports a transient once it has entered the analysis window, so every beat is detected
slightly late. The magnitude was measured across a training split of the corpus and scales with
window length as smearing predicts: about 4.6 ms at a 256-sample window, rising to 25 ms at 2048.

:meta hide-value:
"""
TEMPO_PARAMS = OnsetParams(n_fft=1024)
"""
Envelope settings for tempo estimation, which prefers a smooth envelope.

:meta hide-value:
"""
PHASE_PARAMS = OnsetParams(n_fft=256)
"""
Envelope settings for locating the beat phase, which prefers sharp transients.

:meta hide-value:
"""


class Envelopes(NamedTuple):
    """The onset envelopes used to recover timing."""

    phase: NDArray[np.float32]
    """Short-window envelope, used to locate the beat phase precisely."""
    tempo: NDArray[np.float32]
    """Long-window envelope, used to choose the tempo."""


def _tempo_prior(bpm: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Return a log-normal preference over tempo.

    Folding scores a tempo and its subharmonics almost equally, so this prior is what separates
    them.

    Parameters
    ----------
    bpm : :py:class:`~numpy.ndarray`
        Candidate tempi.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        Weights in the range zero to one.
    """
    octaves = np.log2(bpm / _PRIOR_CENTRE_BPM) / _PRIOR_WIDTH_OCTAVES
    return np.exp(-0.5 * octaves * octaves)


def _fold_scores(
    envelope: NDArray[np.float64], frame_rate: float, tempi: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Score each candidate tempo by mean onset strength at its beat phase.

    Every frame is assigned the phase it occupies within a candidate beat period, and frames sharing
    a phase bin are averaged. The strongest bin is the candidate's score, which is high only when
    onsets recur at exactly that period.

    Parameters
    ----------
    envelope : :py:class:`~numpy.ndarray`
        The onset strength envelope.
    frame_rate : float
        Envelope frames per second.
    tempi : :py:class:`~numpy.ndarray`
        Candidate tempi in beats per minute.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        One score per candidate tempo.
    """
    frames = np.arange(len(envelope), dtype=np.float64)
    weights = envelope.astype(np.float64)
    scores = np.empty(len(tempi), dtype=np.float64)
    for index, bpm in enumerate(tempi):
        # Fractional position of each frame within one beat period.
        phase = np.mod(frames * (bpm / (60.0 * frame_rate)), 1.0)
        bins = (phase * _FOLD_BINS).astype(np.int64)
        totals = np.bincount(bins, weights=weights, minlength=_FOLD_BINS)
        counts = np.bincount(bins, minlength=_FOLD_BINS)
        means = np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0)
        scores[index] = means.max()
    return scores


def _sample_envelope(
    envelope: NDArray[np.float64], times: NDArray[np.float64], frame_rate: float
) -> NDArray[np.float64]:
    """
    Sample an onset envelope at arbitrary times by linear interpolation.

    Parameters
    ----------
    envelope : :py:class:`~numpy.ndarray`
        The onset strength envelope.
    times : :py:class:`~numpy.ndarray`
        Times in seconds at which to sample.
    frame_rate : float
        Envelope frames per second.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        Interpolated envelope values.
    """
    frames = np.arange(len(envelope), dtype=np.float64)
    return np.interp(times * frame_rate, frames, envelope, left=0.0, right=0.0)


def _refine_phase(envelope: NDArray[np.float64], frame_rate: float, period: float) -> float:
    """
    Find the beat phase that maximises accumulated onset strength.

    A spectral phase would be biased here: an onset envelope is an impulse train with a sharp attack
    and slow decay, so its fundamental sinusoid peaks later than the onsets themselves. Folding
    locks onto the real peaks instead.

    Parameters
    ----------
    envelope : :py:class:`~numpy.ndarray`
        The onset strength envelope.
    frame_rate : float
        Envelope frames per second.
    period : float
        Beat period in seconds.

    Returns
    -------
    float
        The best phase within one period, in seconds.
    """
    duration = len(envelope) / frame_rate
    if period <= 0 or duration <= period:
        return 0.0
    steps = max(int(period * frame_rate * _PHASE_SUBDIVISIONS), 32)
    phases = np.linspace(0.0, period, steps, endpoint=False)
    beats = np.arange(int(duration / period), dtype=np.float64)
    times = phases[:, None] + beats[None, :] * period
    totals = _sample_envelope(envelope, times.ravel(), frame_rate).reshape(times.shape).sum(axis=1)
    peak = int(np.argmax(totals))
    before = totals[(peak - 1) % steps]
    after = totals[(peak + 1) % steps]
    denominator = before - 2.0 * totals[peak] + after
    shift = 0.0 if denominator == 0 else 0.5 * (before - after) / denominator
    return float(np.mod((peak + shift) * period / steps, period))


def snap_bpm(bpm: float, tolerance: float = _SNAP_TOLERANCE_BPM) -> float:
    """
    Snap a tempo onto a round value when it lands close to one.

    Parameters
    ----------
    bpm : float
        The estimated tempo.
    tolerance : float
        Largest distance that will be snapped, in beats per minute.

    Returns
    -------
    float
        The snapped tempo, or the input when nothing is close enough.
    """
    candidate = round(bpm * _SNAP_DIVISOR) / _SNAP_DIVISOR
    return candidate if abs(candidate - bpm) <= tolerance else bpm


def estimate_tempo(
    envelope: NDArray[np.float64], frame_rate: float, *, min_bpm: float, max_bpm: float
) -> tuple[float, float]:
    """
    Estimate the tempo of an onset envelope.

    Parameters
    ----------
    envelope : :py:class:`~numpy.ndarray`
        Onset strength envelope, ideally from a smooth long-window detector.
    frame_rate : float
        Envelope frames per second.
    min_bpm : float
        Lowest tempo considered.
    max_bpm : float
        Highest tempo considered.

    Returns
    -------
    tuple[float, float]
        The tempo in beats per minute and a confidence ratio, or zeros when the envelope carries no
        usable periodicity.
    """
    steps = int(np.log(max_bpm / min_bpm) / np.log(_COARSE_STEP)) + 1
    tempi = min_bpm * _COARSE_STEP ** np.arange(steps, dtype=np.float64)
    # Fold only an excerpt, taken from the middle so that intros and silence do not dominate,
    # because the grid spacing is only fine enough to stay aligned across a short span.
    span = int(_COARSE_EXCERPT_SECONDS * frame_rate)
    start = max((len(envelope) - span) // 2, 0)
    excerpt = envelope[start : start + span] if len(envelope) > span else envelope
    scores = _fold_scores(excerpt, frame_rate, tempi) * _tempo_prior(tempi)
    if not np.any(scores > 0):
        return 0.0, 0.0
    coarse = float(tempi[int(np.argmax(scores))])
    ordered = np.sort(scores)[::-1]
    confidence = float(ordered[0] / ordered[1]) if ordered.size > 1 and ordered[1] > 0 else 0.0
    return snap_bpm(coarse), confidence


def estimate_timing_from_envelopes(
    envelopes: Envelopes,
    frame_rate: float,
    *,
    bpm: float = 0.0,
    min_bpm: float = MIN_BPM,
    max_bpm: float = MAX_BPM,
) -> TimingEstimate:
    """
    Estimate tempo and offset from precomputed onset envelopes.

    Parameters
    ----------
    envelopes : Envelopes
        The envelopes, which must share a frame rate.
    frame_rate : float
        Envelope frames per second.
    bpm : float
        Tempo to fit the phase against, or zero to search for one. A supplied tempo is reported
        back with a confidence of zero, since nothing was chosen.
    min_bpm : float
        Lowest tempo considered.
    max_bpm : float
        Highest tempo considered.

    Returns
    -------
    TimingEstimate
        The estimated tempo, offset, and a confidence ratio.
    """
    tempo_values = envelopes.tempo.astype(np.float64)
    usable = len(tempo_values) >= _MIN_FOLD_FRAMES and bool(np.any(tempo_values > 0))
    if bpm > 0:
        # A tempo given by the caller is not a candidate to be weighed, so there is no confidence
        # to report. The phase still has to be fitted against it: fitting against a tempo the
        # search chose instead leaves the two describing different grids, and the offset is then
        # wrong by half the drift the mismatch accumulates.
        chosen, confidence = bpm, 0.0
    elif not usable:
        return _EMPTY.copy()
    else:
        chosen, confidence = estimate_tempo(
            tempo_values, frame_rate, min_bpm=min_bpm, max_bpm=max_bpm
        )
    if chosen <= 0:
        return _EMPTY.copy()
    if not usable:
        return {'bpm': chosen, 'confidence': 0.0, 'offset': 0.0}
    period = 60.0 / chosen
    phase_values = envelopes.phase.astype(np.float64)
    # Folding already locks onto the downbeat in most music, because that is where the strongest
    # recurring onset sits. Re-picking the loudest of the four beats afterwards moves off it, since
    # the backbeat is usually louder.
    downbeat = _refine_phase(phase_values, frame_rate, period) - ONSET_LATENCY_SECONDS
    # Beat 0 sits at -OFFSET, so the offset is the negated downbeat time.
    return {'bpm': chosen, 'confidence': confidence, 'offset': -downbeat}


def estimate_timing(
    path: Path,
    *,
    bpm: float = 0.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    min_bpm: float = MIN_BPM,
    max_bpm: float = MAX_BPM,
) -> TimingEstimate:
    """
    Estimate tempo and offset for an audio file.

    Parameters
    ----------
    path : :py:class:`~pathlib.Path`
        The audio file to analyse.
    bpm : float
        Tempo to fit the phase against, or zero to search for one.
    sample_rate : int
        Sample rate to decode at.
    min_bpm : float
        Lowest tempo considered.
    max_bpm : float
        Highest tempo considered.

    Returns
    -------
    TimingEstimate
        The estimated tempo, offset, and a confidence ratio.
    """
    samples = load_audio(path, sample_rate=sample_rate)
    envelopes = Envelopes(
        phase=onset_envelope(samples, PHASE_PARAMS),
        tempo=onset_envelope(samples, TEMPO_PARAMS),
    )
    return estimate_timing_from_envelopes(
        envelopes, TEMPO_PARAMS.frame_rate, bpm=bpm, min_bpm=min_bpm, max_bpm=max_bpm
    )
