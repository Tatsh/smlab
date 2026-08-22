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
    on_grid,
    panel_bias,
    panel_membership,
    permitted,
    subdivision_quota,
    thin_measures,
)
from .dataset import SUBDIVISIONS_PER_BEAT, difficulty_index
from .encoder import MAX_METER, MAX_RATE, MEASURE_SLOTS, STYLES
from .features import fine_features
from .heads import MAX_DELTA, ChartModel, SelectionBatch
from .playability import FAST_JACK_SECONDS
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
_MIN_KEEP = 1
_CLASSIC_SCALE = 10
_MIN_SEEDED_MEASURES = 2
_SILENCE_DROP = 6.0
"""
How far below a song's typical measure a measure must fall to be left silent.

In units of the median absolute deviation of the per-measure loudness, so only
a measure that is anomalously quiet for its own song rests.

Half of all corpus charts rated twelve to eighteen have no empty measure inside
their body at all, and three quarters have at most 1.4 per cent, so the bar for
resting has to be high. Measured on one rating sixteen chart, three median
deviations leaves 6.4 per cent of the body empty, four leaves 3.2 and six
leaves 1.0 — a single measure, in the outro.
"""
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
    difficulty = torch.tensor([difficulty_index(config.difficulty)], device=device)
    meter = torch.tensor([min(max(config.meter, 0), MAX_METER - 1)], device=device)
    scale = torch.tensor([0 if config.scale <= _CLASSIC_SCALE else 1], device=device)
    style = torch.tensor(
        [STYLES.index(config.style) if config.style in STYLES else 0], device=device
    )
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


def _choose_slots(
    logits: NDArray[np.float32], timing: TimingData, config: GenerationConfig
) -> list[int]:
    """
    Take the highest scoring slots up to the rate the rating implies.

    Parameters
    ----------
    logits : :py:class:`~numpy.ndarray`
        Placement logits per slot.
    timing : TimingData
        Timing used to convert slots into seconds.
    config : GenerationConfig
        Generation settings.

    Returns
    -------
    list[int]
        Chosen slot indices, in ascending order.
    """
    seconds_per_slot = 60.0 / timing.primary_bpm / 12.0
    duration = len(logits) * seconds_per_slot
    wanted = max(int(config.rate * config.density * duration), _MIN_KEEP)
    gap = (
        0
        if config.style == 'keyboard'
        else max(round(FAST_JACK_SECONDS / max(seconds_per_slot, 1e-6)), 1)
    )
    scores = on_grid(logits, triplets=config.triplets)
    order = np.argsort(scores)[::-1]
    within = np.arange(len(logits)) % SUBDIVISIONS_PER_BEAT
    taken = np.zeros(len(logits), dtype=np.bool_)
    chosen: list[int] = []
    # Quarters are laid down across the whole song before any eighth is taken,
    # and eighths before any sixteenth. Ranking every subdivision together lets
    # sixteenths win wherever the audio is loud, which both floods those bars
    # with fast notes and leaves the quiet ones with nothing at all.
    playable, seeded = _seed_pulse(scores, taken, wanted)
    chosen.extend(seeded)
    families = (
        (within % 12 == 0) & playable,
        (within % 6 == 0) & playable,
        playable,
    )
    for quota, family in zip(subdivision_quota(config.rate, wanted), families, strict=True):
        want = min(quota, wanted - len(chosen))
        chosen.extend(_fill(order, taken, family, want, gap))
    chosen.extend(_fill(order, taken, families[-1], wanted - len(chosen), gap))
    return _tidy_rests(chosen, order, taken, gap, playable)


def _seed_pulse(
    scores: NDArray[np.float32], taken: NDArray[np.bool_], wanted: int
) -> tuple[NDArray[np.bool_], list[int]]:
    """
    Put one note in every measure that has any music in it.

    Ranking a subdivision family across the whole song means a quiet passage is
    outbid by a loud one, however long the song is and wherever the quiet part
    falls. A song whose second half calms down keeps its peaks — the strongest
    slots there score as highly as anywhere — but loses on the average, so the
    quarter budget drains into the louder half and the rest goes silent.

    Seeding the strongest beat of each measure first decouples the two: which
    measures play is decided locally, and how busy each one gets is still
    decided globally by score. Only the quietest measures are left out, at the
    7 per cent rate real charts leave measures empty.

    Parameters
    ----------
    scores : :py:class:`~numpy.ndarray`
        Placement scores per slot, with off-grid slots already out of reach.
    taken : :py:class:`~numpy.ndarray`
        Which slots are spoken for, updated in place.
    wanted : int
        Total notes the chart may hold.

    Returns
    -------
    tuple[:py:class:`~numpy.ndarray`, list[int]]
        Which slots belong to a measure that plays at all, and one seeded slot
        per measure that earns a note.
    """
    everywhere = np.ones(len(scores), dtype=np.bool_)
    # Rounded up, not down. A song almost never ends on a bar line, and
    # truncating leaves the final part-measure out of the silence check
    # altogether: it keeps the default of being playable, so the fill puts
    # notes into a fade-out or the dead air after it. Padding with negative
    # infinity lets a short last measure be judged on the slots it does have.
    measures = -(-len(scores) // MEASURE_SLOTS)
    if measures < _MIN_SEEDED_MEASURES or wanted < measures:
        return everywhere, []
    filled = np.full(measures * MEASURE_SLOTS, -np.inf, dtype=scores.dtype)
    filled[: len(scores)] = scores
    padded = filled.reshape(measures, MEASURE_SLOTS)
    strongest = padded.argmax(axis=1)
    loudest = padded.max(axis=1)
    playing = np.isfinite(loudest)
    if not playing.any():
        return everywhere, []
    quiet = _silence_threshold(loudest[playing])
    playable = everywhere.copy()
    seeded: list[int] = []
    for measure in range(measures):
        start = measure * MEASURE_SLOTS
        if not playing[measure] or loudest[measure] <= quiet:
            # The quietest measures are left alone entirely, so the chart keeps
            # the whole-measure rests real charts have rather than trickling a
            # note into every bar.
            playable[start : start + MEASURE_SLOTS] = False
            continue
        slot = start + int(strongest[measure])
        taken[slot] = True
        seeded.append(slot)
    return playable, seeded


def _silence_threshold(loudest: NDArray[np.float32]) -> float:
    """
    Decide how quiet a measure has to be before it carries no note.

    Dropping a fixed share of measures asks the wrong question. Charts rated
    twelve to eighteen leave 7 per cent of their measures empty counting from
    the start, but the median chart has a five-measure intro and **no** empty
    measure at all inside its body; three quarters have at most 1.4 per cent,
    and 74 per cent of the holes that do occur are a single measure. A quota
    therefore eats into the body of any song whose intro is short.

    Silence is a property of the music instead. A measure rests when it is far
    quieter than the song's own typical measure, which leaves an intro or a
    breakdown empty and a song that never lets up entirely full.

    Parameters
    ----------
    loudest : :py:class:`~numpy.ndarray`
        The best placement score in each measure that has any.

    Returns
    -------
    float
        Score at or below which a measure is left silent.
    """
    middle = float(np.median(loudest))
    spread = float(np.median(np.abs(loudest - middle)))
    if spread <= 0:
        return float(np.min(loudest)) - 1.0
    return middle - _SILENCE_DROP * spread


def _fill(
    order: NDArray[np.int64],
    taken: NDArray[np.bool_],
    family: NDArray[np.bool_],
    wanted: int,
    gap: int,
) -> list[int]:
    """
    Take the best scoring free slots from one subdivision family.

    Parameters
    ----------
    order : :py:class:`~numpy.ndarray`
        All slots in descending score order.
    taken : :py:class:`~numpy.ndarray`
        Which slots are spoken for, updated in place.
    family : :py:class:`~numpy.ndarray`
        Which slots belong to the family being filled.
    wanted : int
        How many to take.
    gap : int
        Minimum spacing between notes, in slots.

    Returns
    -------
    list[int]
        The slots taken up.
    """
    chosen: list[int] = []
    if wanted <= 0:
        return chosen
    for index in order:
        if len(chosen) >= wanted:
            break
        slot = int(index)
        if taken[slot] or not family[slot]:
            continue
        if gap > 1 and taken[max(slot - gap + 1, 0) : slot + gap].any():
            continue
        taken[slot] = True
        chosen.append(slot)
    return chosen


def _tidy_rests(
    chosen: list[int],
    order: NDArray[np.int64],
    taken: NDArray[np.bool_],
    gap: int,
    playable: NDArray[np.bool_],
) -> list[int]:
    """
    Move rests onto bar lines.

    Taking the highest scoring slots over the whole song leaves a hole wherever
    the score dips, and a hole has no reason to line up with anything. Charts
    rest for a phrase: measured over 5893 rests in the corpus, 28.7 per cent
    begin on a downbeat and 29.3 per cent end on a bar line, against 9.5 per
    cent each for holes left this way. Emptying the measures that were barely
    used and giving their notes to measures that were already playing turns
    ragged holes into whole-measure rests without changing the note count.

    Parameters
    ----------
    chosen : list[int]
        Slots picked by score.
    order : :py:class:`~numpy.ndarray`
        All slots in descending score order.
    taken : :py:class:`~numpy.ndarray`
        Which slots are spoken for, updated in place.
    gap : int
        Minimum spacing between notes, in slots.
    playable : :py:class:`~numpy.ndarray`
        Which slots sit in a measure that plays at all.

    Returns
    -------
    list[int]
        Chosen slot indices, in ascending order.
    """
    if not chosen:
        return chosen
    thin = thin_measures(chosen)
    if not thin:
        return sorted(chosen)
    kept = [slot for slot in chosen if slot // MEASURE_SLOTS not in thin]
    for slot in chosen:
        if slot // MEASURE_SLOTS in thin:
            taken[slot] = False
    # A rest that simply begins wherever the previous measure trailed off starts
    # off the beat. Charts hit the downbeat and then fall silent, which is why
    # 28.7 per cent of corpus rests begin on one, so the downbeat of an emptied
    # measure is kept and everything after it dropped.
    for measure in thin:
        downbeat = measure * MEASURE_SLOTS
        if downbeat < len(taken) and playable[downbeat]:
            taken[downbeat] = True
            kept.append(downbeat)
    kept.extend(_refill(len(chosen) - len(kept), order, taken, thin, gap, playable))
    return sorted(kept)


def _refill(
    owed: int,
    order: NDArray[np.int64],
    taken: NDArray[np.bool_],
    thin: set[int],
    gap: int,
    playable: NDArray[np.bool_],
) -> list[int]:
    """
    Give the notes freed by emptying a measure to the measures still playing.

    Parameters
    ----------
    owed : int
        How many notes to place.
    order : :py:class:`~numpy.ndarray`
        All slots in descending score order.
    taken : :py:class:`~numpy.ndarray`
        Which slots are spoken for, updated in place.
    thin : set[int]
        Measures that are resting and must not be filled.
    gap : int
        Minimum spacing between notes, in slots.
    playable : :py:class:`~numpy.ndarray`
        Which slots sit in a measure that plays at all.

    Returns
    -------
    list[int]
        The slots taken up.
    """
    added: list[int] = []
    for index in order:
        if len(added) >= owed:
            break
        slot = int(index)
        if taken[slot] or slot // MEASURE_SLOTS in thin or not playable[slot]:
            continue
        if gap > 1 and taken[max(slot - gap + 1, 0) : slot + gap].any():
            continue
        taken[slot] = True
        added.append(slot)
    return added


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
    slots = _choose_slots(logits, timing, config)
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
    previous_slot = slots[0]
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
            in_run = budget.enter_run(gap, style=config.style)
            mask = permitted(
                vocabulary,
                config,
                held,
                previous_panels,
                gap,
                room,
                budget.crossed() | budget.stale(),
                spent=budget.freezes_spent(config.holds),
                busy=tight[step],
                crowded_jumps=budget.jumps_spent(config.jump_share),
            )
            token = _sample(
                scores + config.balance * panel_bias(membership, budget.usage),
                mask,
                config.temperature,
                rng,
            )
            codes = list(vocabulary.panels_of(token))
            budget.record(vocabulary.stepped_panels(token), codes, in_run=in_run)
            for panel in list(held):
                if codes[panel] == TAIL_CODE:
                    del held[panel]
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
    scaled -= scaled.max()
    weights = np.exp(scaled)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        return int(np.argmax(np.where(mask, scores, -np.inf)))
    return int(rng.choice(len(weights), p=weights / total))
