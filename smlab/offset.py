"""
Learned recovery of the downbeat phase.

Tempo and phase are separate problems and only the second is solved here. Given
a tempo, the beat grid is fixed up to where it starts, and that remaining
degree of freedom is what ``#OFFSET`` encodes.

The heuristic in :py:mod:`smlab.tempo` chooses the phase by taking the loudest
point of a folded onset envelope. That criterion is wrong in a way no amount of
tuning fixes: a downbeat is not the loudest moment in the bar. It is where the
bass lands, where the harmony turns over, where the phrase begins. A hi-hat
playing straight eighths is often louder than the kick, and the grid then locks
onto the off-beat. Measured over 154 songs whose tempo was recovered correctly,
that heuristic lands within 30 ms of the authored offset 59.7 per cent of the
time, puts 13.6 per cent of songs a clean half-beat out, and gets the measure
right in only 45.5 per cent of cases.

What the model sees is the same fold, split into frequency bands so that a kick
and a hi-hat are not summed into one number, and it is asked which of the 96
positions in the bar the downbeat occupies. It is built only from circular
convolutions, so shifting the input shifts the answer by exactly as much: the
network cannot memorise that downbeats tend to land at a particular bin, and
every window of every song teaches it something about every phase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn
import librosa
import numpy as np
import torch

from .audio import OnsetParams, load_audio

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

__all__ = (
    'BEATS_PER_MEASURE',
    'EXCERPT_SECONDS',
    'PHASE_BINS',
    'OffsetModel',
    'band_envelopes',
    'fold_profile',
    'offset_for_phase',
    'phase_of_offset',
    'predict_phase',
    'refine_offset',
)

PHASE_BINS = 96
"""
Positions in a bar the downbeat is chosen from.

Twenty-four per beat, matching the fine grid the chart model works on. At 128
BPM one bin spans 19.5 ms, comfortably below the roughly 30 ms at which a
mis-timed chart starts to feel wrong.

:meta hide-value:
"""
EXCERPT_SECONDS = 25.0
"""
Seconds of audio one folded profile covers.

Long enough to average over many bars, short enough that a tempo estimate a
fraction of a beat per minute out has not drifted a whole bin by the end.

:meta hide-value:
"""
BEATS_PER_MEASURE = 4
"""Beats in a bar, which is what the phase is measured against."""
BAND_EDGES = (0.0, 150.0, 500.0, 2000.0, 11025.0)
"""
Frequency band edges in hertz, one onset envelope taken between each pair.

Summing the spectrum into a single envelope is what lets a loud hi-hat outvote
a kick. Kept apart, the lowest band carries the kick and bass that mark the
downbeat and the highest carries the cymbals that mislead.

These have to be genuine slices. Passing ``fmax`` alone to a flux detector
gives nested low-pass bands rather than disjoint ones, and asking for a full
mel filterbank below 150 Hz produces filters so narrow that every one of them
is empty — a band that is identically zero and contributes nothing at all,
which is what the first version of this did.

:meta hide-value:
"""
_BANDS = len(BAND_EDGES) - 1
_CHANNELS = 64
_KERNEL = 5
_DILATIONS = (1, 2, 4, 8, 16)
"""
Dilations of the stacked convolutions.

With a kernel of five these reach 125 bins, so every position in the bar sees
the whole of it. A downbeat is only recognisable relative to the rest of the
measure, so a receptive field short of that would be guessing.

:meta hide-value:
"""


def band_envelopes(
    samples: NDArray[np.float32], params: OnsetParams | None = None
) -> NDArray[np.float32]:
    """
    Compute one onset envelope per frequency band.

    Parameters
    ----------
    samples : :py:class:`~numpy.ndarray`
        Mono audio samples.
    params : OnsetParams | None
        Settings to use, or ``None`` for the defaults.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        Envelopes shaped ``(bands, frames)``.
    """
    settings = params if params is not None else OnsetParams()
    centres = librosa.mel_frequencies(n_mels=settings.n_mels, fmax=settings.sample_rate / 2)
    # Channel boundaries as mel-bin indices, so each envelope is flux over one
    # slice of the spectrum rather than everything below a ceiling.
    channels = [int(np.searchsorted(centres, edge)) for edge in BAND_EDGES]
    channels[-1] = settings.n_mels
    stacked = np.asarray(
        librosa.onset.onset_strength_multi(
            y=samples,
            sr=settings.sample_rate,
            hop_length=settings.hop_length,
            n_fft=settings.n_fft,
            n_mels=settings.n_mels,
            channels=channels,
            aggregate=np.median,
        ),
        dtype=np.float32,
    )
    # Each band is scaled on its own. Absolute loudness differs enormously
    # between bands and says nothing about where the downbeat is; the shape
    # within a band is the whole signal.
    peak = np.maximum(stacked.max(axis=1, keepdims=True), 1e-6)
    return np.asarray(stacked / peak, dtype=np.float32)


def fold_profile(
    envelopes: NDArray[np.float32], frame_rate: float, bpm: float, start: float = 0.0
) -> NDArray[np.float32]:
    """
    Average each band over its position within the bar.

    Parameters
    ----------
    envelopes : :py:class:`~numpy.ndarray`
        Band envelopes shaped ``(bands, frames)``.
    frame_rate : float
        Envelope frames per second.
    bpm : float
        Tempo the bar length is taken from.
    start : float
        Seconds the fold is measured from, which fixes what phase zero means.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        Profile shaped ``(bands, PHASE_BINS)``, each band scaled to its own
        peak so that only its shape carries.
    """
    period = BEATS_PER_MEASURE * 60.0 / bpm
    times = np.arange(envelopes.shape[1], dtype=np.float64) / frame_rate - start
    bins = np.floor((times % period) / period * PHASE_BINS).astype(np.int64) % PHASE_BINS
    profile = np.zeros((envelopes.shape[0], PHASE_BINS), dtype=np.float64)
    counts = np.bincount(bins, minlength=PHASE_BINS).astype(np.float64)
    for band in range(envelopes.shape[0]):
        profile[band] = np.bincount(
            bins, weights=envelopes[band].astype(np.float64), minlength=PHASE_BINS
        )
    profile /= np.maximum(counts, 1.0)
    profile -= profile.mean(axis=1, keepdims=True)
    spread = np.maximum(np.abs(profile).max(axis=1, keepdims=True), 1e-6)
    return np.asarray(profile / spread, dtype=np.float32)


def phase_of_offset(offset: float, bpm: float) -> int:
    """
    Return the bar position the downbeat sits at for a given offset.

    Parameters
    ----------
    offset : float
        The ``#OFFSET`` value, where beat zero happens at ``-offset`` seconds.
    bpm : float
        Tempo.

    Returns
    -------
    int
        Bin in ``[0, PHASE_BINS)``.
    """
    period = BEATS_PER_MEASURE * 60.0 / bpm
    return int(np.floor(((-offset) % period) / period * PHASE_BINS)) % PHASE_BINS


def offset_for_phase(phase: int, bpm: float) -> float:
    """
    Return the offset that puts the downbeat at a bar position.

    Any offset differing by a whole bar describes the same grid, so the
    representative nearest zero is chosen: beat zero then falls inside the first
    bar of the audio, which is where simfiles conventionally put it.

    Parameters
    ----------
    phase : int
        Bin in ``[0, PHASE_BINS)``.
    bpm : float
        Tempo.

    Returns
    -------
    float
        The ``#OFFSET`` value.
    """
    period = BEATS_PER_MEASURE * 60.0 / bpm
    return -((phase + 0.5) / PHASE_BINS) * period


class OffsetModel(nn.Module):
    """Scores each bar position for holding the downbeat."""

    def __init__(self, channels: int = _CHANNELS) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = _BANDS
        for dilation in _DILATIONS:
            layers.extend((
                nn.Conv1d(
                    width,
                    channels,
                    _KERNEL,
                    padding=dilation * (_KERNEL - 1) // 2,
                    dilation=dilation,
                    padding_mode='circular',
                ),
                nn.GELU(),
            ))
            width = channels
        # A single channel out, read as one score per bar position. Nothing
        # here mixes positions except the circular convolutions, so shifting
        # the profile shifts the scores by the same amount and the model has no
        # way to prefer a bin for its own sake.
        layers.append(nn.Conv1d(width, 1, 1))
        self.stack = nn.Sequential(*layers)

    def forward(self, profile: torch.Tensor) -> torch.Tensor:
        """
        Score every bar position.

        Parameters
        ----------
        profile : :py:class:`~torch.Tensor`
            Folded profile shaped ``(batch, bands, PHASE_BINS)``.

        Returns
        -------
        :py:class:`~torch.Tensor`
            Logits shaped ``(batch, PHASE_BINS)``.
        """
        scored: torch.Tensor = self.stack(profile)
        return scored.squeeze(1)


def predict_phase(
    model: OffsetModel,
    envelopes: NDArray[np.float32],
    frame_rate: float,
    bpm: float,
    excerpt_seconds: float = EXCERPT_SECONDS,
) -> tuple[int, float]:
    """
    Choose the bar position the downbeat occupies.

    The song is read in excerpts of the length the model was trained on and
    their distributions averaged. Folding the whole song at once would hand the
    model a smoother profile than anything it saw in training, and a song that
    changes character partway would have its sections averaged into each other
    before the model ever saw them.

    Parameters
    ----------
    model : OffsetModel
        The trained model.
    envelopes : :py:class:`~numpy.ndarray`
        Band envelopes for the song.
    frame_rate : float
        Envelope frames per second.
    bpm : float
        Tempo.
    excerpt_seconds : float
        Seconds each excerpt covers.

    Returns
    -------
    tuple[int, float]
        The chosen bin and how much of the averaged distribution it holds.
    """
    span = max(int(excerpt_seconds * frame_rate), 1)
    starts = list(range(0, max(envelopes.shape[1] - span, 1), span))
    device = next(model.parameters()).device
    stacked = np.stack([
        fold_profile(envelopes[:, s : s + span], frame_rate, bpm, start=-s / frame_rate)
        for s in starts
    ])
    with torch.no_grad():
        logits = model(torch.from_numpy(stacked).to(device))
        weights = torch.softmax(logits.float(), dim=1).mean(dim=0).cpu().numpy()
    return int(np.argmax(weights)), float(np.max(weights))


def refine_offset(model: OffsetModel, audio: Path, bpm: float) -> tuple[float, float]:
    """
    Recover the offset for a song whose tempo is already known.

    Parameters
    ----------
    model : OffsetModel
        The trained phase model.
    audio : :py:class:`~pathlib.Path`
        Audio file to read.
    bpm : float
        Tempo, which fixes the bar length the phase is measured against.

    Returns
    -------
    tuple[float, float]
        The offset and how much of the averaged distribution its bin holds,
        which is a rough confidence.
    """
    params = OnsetParams()
    samples = load_audio(audio, sample_rate=params.sample_rate)
    envelopes = band_envelopes(samples, params)
    phase, weight = predict_phase(model, envelopes, params.frame_rate, bpm)
    return offset_for_phase(phase, bpm), weight
