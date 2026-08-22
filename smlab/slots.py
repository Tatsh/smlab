"""
Deciding which grid slots carry a note.

The placement head ranks every slot, but taking the highest scores outright produces a chart that is
wrong in two ways: fine subdivisions win wherever the audio is loud, so busy bars flood with
sixteenths while quiet ones empty, and the holes that result line up with nothing. Both are fixed
here rather than in the model, by laying down coarse subdivisions before fine ones and by moving
rests onto bar lines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .constraints import on_grid, subdivision_quota, thin_measures
from .dataset import SUBDIVISIONS_PER_BEAT
from .encoder import MEASURE_SLOTS
from .features import SILENT_DECIBELS
from .playability import FAST_JACK_SECONDS

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .generate import GenerationConfig
    from .timing import TimingData

__all__ = ('choose_slots', 'fill', 'refill', 'seed_pulse', 'silence_threshold', 'tidy_rests')

_MIN_KEEP = 1
_MIN_SEEDED_MEASURES = 2
_SILENCE_DROP = 6.0
"""
How far below a song's typical measure a measure must fall to be left silent.

In units of the median absolute deviation of the per-measure loudness, so only a measure that is
anomalously quiet for its own song rests.

Half of all corpus charts rated twelve to eighteen have no empty measure inside their body at all,
and three quarters have at most 1.4 per cent, so the bar for resting has to be high. Measured on one
rating sixteen chart, three median deviations leaves 6.4 per cent of the body empty, four leaves 3.2
and six leaves 1.0 — a single measure, in the outro.
"""


def _audible(loudness: NDArray[np.float32] | None, slots: int) -> NDArray[np.bool_]:
    """
    Mark the slots whose audio carries something to chart.

    Parameters
    ----------
    loudness : :py:class:`~numpy.ndarray` | None
        Decibel level per slot, or ``None`` when it was not measured.
    slots : int
        Number of slots the chart spans.

    Returns
    -------
    :py:class:`~numpy.ndarray`
        One flag per slot, all set when no level was supplied.
    """
    if loudness is None:
        return np.ones(slots, dtype=np.bool_)
    measured = np.full(slots, SILENT_DECIBELS + 1.0, dtype=np.float32)
    measured[: min(len(loudness), slots)] = loudness[:slots]
    return np.asarray(measured > SILENT_DECIBELS, dtype=np.bool_)


def choose_slots(
    logits: NDArray[np.float32],
    timing: TimingData,
    config: GenerationConfig,
    loudness: NDArray[np.float32] | None = None,
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
    loudness : :py:class:`~numpy.ndarray` | None
        Decibel level of the mixture at each slot, or ``None`` to decide silence from the scores
        alone.

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
    # Which measures rest is otherwise decided from the placement scores, and those are a model
    # output: over dead air the network has nothing to read but still returns a number, and a whole
    # song of them can be flat enough that the trailing silence never looks unusual. The audio
    # itself says so outright, so slots with no music in them are put out of reach first.
    audible = _audible(loudness, len(scores))
    scores = np.where(audible, scores, -np.inf).astype(np.float32)
    order = np.argsort(scores)[::-1]
    within = np.arange(len(logits)) % SUBDIVISIONS_PER_BEAT
    taken = np.zeros(len(logits), dtype=np.bool_)
    chosen: list[int] = []
    # Quarters are laid down across the whole song before any eighth is taken, and eighths before
    # any sixteenth. Ranking every subdivision together lets sixteenths win wherever the audio is
    # loud, which both floods those bars with fast notes and leaves the quiet ones with nothing.
    playable, seeded = seed_pulse(scores, taken, wanted)
    playable &= audible
    chosen.extend(seeded)
    families = (
        (within % 12 == 0) & playable,
        (within % 6 == 0) & playable,
        playable,
    )
    for quota, family in zip(subdivision_quota(config.rate, wanted), families, strict=True):
        want = min(quota, wanted - len(chosen))
        chosen.extend(fill(order, taken, family, want, gap))
    chosen.extend(fill(order, taken, families[-1], wanted - len(chosen), gap))
    return tidy_rests(chosen, order, taken, gap, playable)


def seed_pulse(
    scores: NDArray[np.float32], taken: NDArray[np.bool_], wanted: int
) -> tuple[NDArray[np.bool_], list[int]]:
    """
    Put one note in every measure that has any music in it.

    Ranking a subdivision family across the whole song means a quiet passage is outbid by a loud
    one, however long the song is and wherever the quiet part falls. A song whose second half calms
    down keeps its peaks — the strongest slots there score as highly as anywhere — but loses on the
    average, so the quarter budget drains into the louder half and the rest goes silent.

    Seeding the strongest beat of each measure first decouples the two: which measures play is
    decided locally, and how busy each one gets is still decided globally by score. Only the
    quietest measures are left out, at the 7 per cent rate real charts leave measures empty.

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
        Which slots belong to a measure that plays at all, and one seeded slot per measure that
        earns a note.
    """
    everywhere = np.ones(len(scores), dtype=np.bool_)
    # Rounded up, not down. A song almost never ends on a bar line, and truncating leaves the final
    # part-measure out of the silence check altogether: it keeps the default of being playable, so
    # the fill puts notes into a fade-out or the dead air after it. Padding with negative infinity
    # lets a short last measure be judged on the slots it does have.
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
    quiet = silence_threshold(loudest[playing])
    playable = everywhere.copy()
    seeded: list[int] = []
    for measure in range(measures):
        start = measure * MEASURE_SLOTS
        if not playing[measure] or loudest[measure] <= quiet:
            # The quietest measures are left alone entirely, so the chart keeps the whole-measure
            # rests real charts have rather than trickling a note into every bar.
            playable[start : start + MEASURE_SLOTS] = False
            continue
        slot = start + int(strongest[measure])
        taken[slot] = True
        seeded.append(slot)
    return playable, seeded


def silence_threshold(loudest: NDArray[np.float32]) -> float:
    """
    Decide how quiet a measure has to be before it carries no note.

    Dropping a fixed share of measures asks the wrong question. Charts rated twelve to eighteen
    leave 7 per cent of their measures empty counting from the start, but the median chart has a
    five-measure intro and **no** empty measure at all inside its body; three quarters have at most
    1.4 per cent, and 74 per cent of the holes that do occur are a single measure. A quota therefore
    eats into the body of any song whose intro is short.

    Silence is a property of the music instead. A measure rests when it is far quieter than the
    song's own typical measure, which leaves an intro or a breakdown empty and a song that never
    lets up entirely full.

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


def fill(
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


def tidy_rests(
    chosen: list[int],
    order: NDArray[np.int64],
    taken: NDArray[np.bool_],
    gap: int,
    playable: NDArray[np.bool_],
) -> list[int]:
    """
    Move rests onto bar lines.

    Taking the highest scoring slots over the whole song leaves a hole wherever the score dips, and
    a hole has no reason to line up with anything. Charts rest for a phrase: measured over 5893
    rests in the corpus, 28.7 per cent begin on a downbeat and 29.3 per cent end on a bar line,
    against 9.5 per cent each for holes left this way. Emptying the measures that were barely used
    and giving their notes to measures that were already playing turns ragged holes into
    whole-measure rests without changing the note count.

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
    # A rest that simply begins wherever the previous measure trailed off starts off the beat.
    # Charts hit the downbeat and then fall silent, which is why 28.7 per cent of corpus rests begin
    # on one, so the downbeat of an emptied measure is kept and everything after it dropped.
    for measure in thin:
        downbeat = measure * MEASURE_SLOTS
        if downbeat < len(taken) and playable[downbeat]:
            taken[downbeat] = True
            kept.append(downbeat)
    kept.extend(refill(len(chosen) - len(kept), order, taken, thin, gap, playable))
    return sorted(kept)


def refill(
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
