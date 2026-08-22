"""
Generation with the shared encoder model.

Inference mirrors training: the song is separated into stems, resampled onto
the fine beat grid, and encoded once. The placement head then ranks every note
slot and the selection head walks the chosen ones in order, so both heads see
the same reading of the music.

How many slots to keep is decided separately from which ones, because the
placement head ranks well but is not calibrated. Keeping the number a rating
implies preserves the ranking and leaves quiet passages empty, which is what a
rest is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import logging

import librosa
import numpy as np
import torch

from .audio import DEFAULT_SAMPLE_RATE
from .chart_data import WINDOW_STEPS
from .constraints import (
    HOLD_CODES,
    HOLD_SLOTS,
    TAIL_CODE,
    Budget,
    crowded,
    panel_bias,
    panel_membership,
    permitted,
)
from .dataset import SUBDIVISIONS_PER_BEAT, difficulty_index
from .encoder import MAX_METER, MAX_RATE, MEASURE_SLOTS, STYLES
from .features import fine_features
from .heads import MAX_DELTA, ChartModel, SelectionBatch
from .slots import choose_slots
from .stems import separate

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from .generate import GenerationConfig
    from .stems import Separator
    from .timing import TimingData
    from .vocab import Vocabulary

__all__ = ('encode_song', 'generate_rows', 'song_features')

log = logging.getLogger(__name__)

_DECIBEL_SCALE = 40.0
_CHUNK_SLOTS = 3072
_CLASSIC_SCALE = 10
_MAX_DELTA = MAX_DELTA
_CONTEXT_STEPS = WINDOW_STEPS
_PRIOR_WEIGHT = 0.25
"""
How much of the metric prior to keep when ranking slots.

The prior is right about probability and wrong for ranking. It puts quarter
notes far above everything else at every point in the song, so taking the
highest scoring slots fills every quarter in the piece before it takes a single
sixteenth anywhere. Charts do not work that way: they put sixteenth runs in
particular bars and leave others sparse.

Damping it lets the audio decide. Measured on one song at rating nine, the
share of sixteenths and finer goes 1.3 per cent at full weight, 6 at a half and
13 at a quarter, against 13.2 per cent for real charts of that rating. Dropping
the prior entirely overshoots to 21 per cent and starts placing notes off the
sixteenth grid altogether.
"""


def song_features(
    separator: Separator, audio: Path, timing: TimingData, device: torch.device
) -> NDArray[np.float16]:
    """
    Separate a song and build its fine-grid features.

    Parameters
    ----------
    separator : Separator
        The stem separation model.
    audio : :py:class:`~pathlib.Path`
        Audio file to read.
    timing : TimingData
        Timing used to map beats onto times.
    device : :py:class:`~torch.device`
        Device separation runs on.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        Features shaped ``(fine_slots, channels)``.
    """
    stems = separate(separator, audio, device)
    rate = separator.samplerate
    resampled = {
        name: librosa.resample(samples, orig_sr=rate, target_sr=DEFAULT_SAMPLE_RATE)
        for name, samples in stems.items()
    }
    mixture, _ = librosa.load(str(audio), sr=DEFAULT_SAMPLE_RATE, mono=True)
    return fine_features(resampled, np.asarray(mixture, dtype=np.float32), timing)


def encode_song(
    model: ChartModel, features: NDArray[np.float16], config: GenerationConfig, device: torch.device
) -> tuple[torch.Tensor, NDArray[np.float32]]:
    """
    Encode a whole song and score every note slot.

    Long songs are encoded in windows, because attention over the full length
    would not fit.

    Parameters
    ----------
    model : ChartModel
        The trained model.
    features : :py:class:`~numpy.ndarray`
        Fine-grid features for the song.
    config : GenerationConfig
        Generation settings supplying the conditioning.
    device : :py:class:`~torch.device`
        Compute device.

    Returns
    -------
    tuple[:py:class:`~torch.Tensor`, :py:class:`~numpy.ndarray`]
        The encoder output for every note slot and the placement logits.
    """
    model.eval()
    note_slots = features.shape[0] // 2
    if note_slots == 0:
        # Audio shorter than two fine slots leaves nothing to chart, and the
        # chunk loop would otherwise concatenate an empty list of tensors.
        return torch.zeros((0, 0), device=device), np.zeros(0, dtype=np.float32)
    difficulty = torch.tensor([difficulty_index(config.difficulty)], device=device)
    meter = torch.tensor([min(max(config.meter, 0), MAX_METER - 1)], device=device)
    scale = torch.tensor([0 if config.scale <= _CLASSIC_SCALE else 1], device=device)
    style = torch.tensor([STYLES.index(config.style)], device=device)
    # The same note rate the slot budget is drawn from, so the encoder is told
    # how dense the chart it is being asked for will actually be.
    rate = torch.tensor([min(int(config.rate), MAX_RATE - 1)], device=device)
    encoded: list[torch.Tensor] = []
    logits = np.zeros(note_slots, dtype=np.float32)
    with (
        torch.no_grad(),
        torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'),
    ):
        for start in range(0, note_slots, _CHUNK_SLOTS):
            stop = min(start + _CHUNK_SLOTS, note_slots)
            window = features[2 * start : 2 * stop].astype(np.float32) / _DECIBEL_SCALE
            tensor = torch.from_numpy(window).unsqueeze(0).to(device)
            positions = (
                torch
                .from_numpy((np.arange(stop - start, dtype=np.int64) + start) % MEASURE_SLOTS)
                .unsqueeze(0)
                .to(device)
            )
            chunk = model.encode(tensor, difficulty, meter, scale, style, rate)
            scored = model.placement(chunk, positions, _PRIOR_WEIGHT)
            logits[start:stop] = scored[0].float().cpu().numpy()
            encoded.append(chunk[0].float())
    return torch.cat(encoded, dim=0), logits


def generate_rows(  # noqa: PLR0914
    model: ChartModel,
    vocabulary: Vocabulary,
    features: NDArray[np.float16],
    timing: TimingData,
    config: GenerationConfig,
    device: torch.device,
) -> list[tuple[int, list[int]]]:
    """
    Generate one chart for a song.

    Parameters
    ----------
    model : ChartModel
        The trained model.
    vocabulary : Vocabulary
        Pattern vocabulary the selection head predicts over.
    features : :py:class:`~numpy.ndarray`
        Fine-grid features for the song.
    timing : TimingData
        Timing for the song.
    config : GenerationConfig
        Generation settings.
    device : :py:class:`~torch.device`
        Compute device.

    Returns
    -------
    list[tuple[int, list[int]]]
        Grid slot and four panel codes for each generated row.
    """
    encoded, logits = encode_song(model, features, config, device)
    slots = choose_slots(logits, timing, config)
    if not slots:
        return []
    rng = np.random.default_rng(config.seed)
    seconds_per_slot = 60.0 / timing.primary_bpm / 12.0
    rows: list[tuple[int, list[int]]] = []
    held: dict[int, int] = {}
    budget = Budget()
    membership = panel_membership(vocabulary)
    tight = crowded(slots)
    history = [len(vocabulary)]
    past_slots: list[int] = []
    past_deltas: list[int] = []
    previous_panels: frozenset[int] = frozenset()
    # Nothing precedes the first note, so it starts a run rather than joining
    # one. Starting the gap at zero would count it as inside a run and flip the
    # alternation, mirroring which foot is due for everything that follows.
    previous_slot = slots[0] - _MAX_DELTA
    with torch.no_grad():
        for step, slot in enumerate(slots):
            past_slots.append(slot)
            past_deltas.append(min(slot - previous_slot, _MAX_DELTA))
            # The head is a causal transformer trained on long stretches of a
            # chart. Handing it only the current step would leave its attention
            # with nothing to look at but that step, and the pattern embedding
            # is then the sole input reaching the output, so it echoes whatever
            # came before and the chart repeats one arrow forever. It has to be
            # re-run over the recent history and read at the last position.
            window = slice(max(len(past_slots) - _CONTEXT_STEPS, 0), len(past_slots))
            context = past_slots[window]
            batch = SelectionBatch(
                delta=torch.tensor([past_deltas[window]], device=device),
                position=torch.tensor(
                    [[value % MEASURE_SLOTS for value in context]], device=device
                ),
                previous=torch.tensor([history[window]], device=device),
                slots=torch.tensor([context], device=device),
            )
            scores = model.selection(encoded.unsqueeze(0), batch)[0, -1].float().cpu().numpy()
            # A hold that has run too long is ended by writing its tail. Merely
            # forgetting it would leave a head with no terminator, which makes
            # the panel read as occupied for the rest of the song and stops the
            # file loading at all.
            _expire_holds(rows, held, slot)
            gap = (slot - previous_slot) * seconds_per_slot
            following = slots[step + 1] - slot if step + 1 < len(slots) else _MAX_DELTA
            room = min(gap, following * seconds_per_slot)
            in_run = budget.enter_run(gap, seconds_per_slot)
            mask = permitted(
                vocabulary,
                config,
                held,
                previous_panels,
                gap,
                room,
                budget.crossed(config.crossover_share) | budget.stale(),
                busy=tight[step],
                crowded_jumps=budget.jumps_spent(config.jump_share),
                overrun=budget.overrun(config.crossover_share),
                spent=budget.freezes_spent(config.holds),
            )
            token = _sample(
                scores + config.balance * panel_bias(membership, budget.usage),
                mask,
                config.temperature,
                rng,
            )
            codes = list(vocabulary.panels_of(token))
            budget.record(vocabulary.stepped_panels(token), codes, in_run=in_run)
            # Nothing releases a panel here. Slots strictly increase, so the
            # expiry above has already closed every open freeze, and with none
            # open the mask bars a tail as an orphan.
            for panel, code in enumerate(codes):
                if code in HOLD_CODES:
                    held[panel] = slot
            rows.append((slot, codes))
            history.append(token)
            previous_panels = vocabulary.stepped_panels(token)
            previous_slot = slot
    if held:
        rows.append((slots[-1] + SUBDIVISIONS_PER_BEAT, _tail_row(list(held))))
    return rows


def _expire_holds(rows: list[tuple[int, list[int]]], held: dict[int, int], slot: int) -> None:
    """
    End any hold that has run longer than a phrase.

    A hold that is merely forgotten leaves a head with no terminator, which
    makes its panel read as occupied for the rest of the song and stops the
    file loading at all, so the tail is written out.

    Parameters
    ----------
    rows : list[tuple[int, list[int]]]
        Rows so far, appended to in place.
    held : dict[int, int]
        Slot at which each open hold began, keyed by panel. Modified in place.
    slot : int
        Slot being decoded.
    """
    # Every freeze ends before the next note lands. A freeze only starts where
    # there is room for it, so ending it here is what keeps the other foot idle
    # for its whole span; letting it run on is what pinned a foot for two bars.
    expired = [panel for panel, began in held.items() if slot > began]
    if not expired:
        return
    finish = min(min(held[panel] for panel in expired) + HOLD_SLOTS, slot - 1)
    rows.append((finish, _tail_row(expired)))
    for panel in expired:
        del held[panel]


def _tail_row(panels: list[int]) -> list[int]:
    """
    Build a row that ends a hold on each of several panels.

    Parameters
    ----------
    panels : list[int]
        Panels whose holds should end.

    Returns
    -------
    list[int]
        Four panel codes, carrying a tail where a hold ends.
    """
    codes = [0, 0, 0, 0]
    for panel in panels:
        codes[panel] = TAIL_CODE
    return codes


def _sample(
    scores: NDArray[np.float32],
    mask: NDArray[np.bool_],
    temperature: float,
    rng: np.random.Generator,
) -> int:
    """
    Sample one pattern from the masked distribution.

    Parameters
    ----------
    scores : :py:class:`~numpy.ndarray`
        Unnormalised scores for each pattern.
    mask : :py:class:`~numpy.ndarray`
        Which patterns are permitted.
    temperature : float
        Softmax temperature.
    rng : :py:class:`~numpy.random.Generator`
        Source of randomness.

    Returns
    -------
    int
        Index of the sampled pattern.
    """
    if not mask.any():
        return int(np.argmax(scores))
    scaled = scores / max(temperature, 1e-3)
    scaled[~mask] = -np.inf
    # Subtracting the largest score leaves at least one entry at zero, so the
    # weights always total one or more and the division below is safe.
    scaled -= scaled.max()
    weights = np.exp(scaled)
    return int(rng.choice(len(weights), p=weights / weights.sum()))
