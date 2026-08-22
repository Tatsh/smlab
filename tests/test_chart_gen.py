"""Tests for decoding a chart out of an encoded song."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]  # No stubs are published.
import torch

from smlab.chart.gen import encode_song, generate_rows, song_features
from smlab.dataset import SUBDIVISIONS_PER_BEAT
from smlab.encoder import MEASURE_SLOTS, EncoderConfig
from smlab.features import FINE_SUBDIVISIONS, TOTAL_CHANNELS
from smlab.generate import GenerationConfig
from smlab.heads import ChartModel
from smlab.stems import STEM_NAMES
from smlab.timing import TimingData
from smlab.vocab import Vocabulary, encode_row

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

_RATE = 22050
_BPM = 150.0
_CPU = torch.device('cpu')
_TIMING = TimingData.constant(_BPM, 0.0)
_SMALL = EncoderConfig(
    channels=16, model_dimension=24, local_blocks=1, slot_layers=1, measure_layers=1, heads=2
)
_TAIL = 3
_VOCABULARY = Vocabulary([
    encode_row(row)
    for row in (
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
        (1, 0, 0, 1),
        (2, 0, 0, 0),
        (3, 0, 0, 0),
    )
])


class _Separator:
    """Stands in for a demucs model, which is far too slow to run in a test."""

    def __init__(self) -> None:
        self.samplerate = _RATE
        self.sources = list(STEM_NAMES)


def _features(measures: int = 8) -> np.ndarray:
    slots = measures * MEASURE_SLOTS
    rng = np.random.default_rng(0)
    return rng.normal(size=(2 * slots, TOTAL_CHANNELS)).astype(np.float16)


def _closest_repeat(rows: list[tuple[int, list[int]]]) -> int:
    """Return the smallest gap, in slots, between two steps on one panel."""
    last: dict[int, int] = {}
    closest = 10**6
    for slot, codes in rows:
        for panel, code in enumerate(codes):
            if code not in {1, 2}:
                continue
            if panel in last:
                closest = min(closest, slot - last[panel])
            last[panel] = slot
    return closest


def _model() -> ChartModel:
    torch.manual_seed(0)
    return ChartModel(len(_VOCABULARY), _SMALL).eval()


def test_a_song_is_separated_onto_the_fine_grid(tmp_path: Path, mocker: MockerFixture) -> None:
    audio = tmp_path / 'song.wav'
    sf.write(audio, np.zeros(_RATE * 4, dtype='float32'), _RATE)
    mocker.patch(
        'smlab.chart.gen.separate',
        return_value=dict.fromkeys(STEM_NAMES, np.zeros(_RATE * 4, dtype=np.float32)),
    )
    features = song_features(_Separator(), audio, _TIMING, _CPU)
    assert features.shape[1] == TOTAL_CHANNELS
    # Four seconds at 150 BPM is ten beats, sampled at the fine resolution.
    assert features.shape[0] == pytest.approx(10 * FINE_SUBDIVISIONS, abs=FINE_SUBDIVISIONS)


def test_every_note_slot_is_encoded_and_scored() -> None:
    encoded, logits = encode_song(_model(), _features(), GenerationConfig(), _CPU)
    assert encoded.shape[0] == 8 * MEASURE_SLOTS
    assert logits.shape == (8 * MEASURE_SLOTS,)
    assert np.all(np.isfinite(logits))


def test_a_song_longer_than_one_window_is_encoded_in_chunks() -> None:
    # Attention over a whole song would not fit, so it is encoded in windows and the pieces joined.
    encoded, logits = encode_song(_model(), _features(measures=70), GenerationConfig(), _CPU)
    assert encoded.shape[0] == 70 * MEASURE_SLOTS
    assert logits.shape == (70 * MEASURE_SLOTS,)


def test_a_rating_beyond_the_embedding_is_clamped() -> None:
    config = GenerationConfig(meter=999, nps=4.0, scale=20)
    assert encode_song(_model(), _features(), config, _CPU)[1].shape[0] == 8 * MEASURE_SLOTS


def test_a_chart_is_decoded_at_roughly_the_rate_asked_for() -> None:
    config = GenerationConfig(nps=4.0, seed=1)
    rows = generate_rows(_model(), _VOCABULARY, _features(measures=16), _TIMING, config, _CPU)
    assert rows
    seconds = 16 * 4 * 60.0 / _BPM
    assert len(rows) / seconds == pytest.approx(4.0, rel=0.5)


def test_every_row_lands_on_a_slot_in_ascending_order() -> None:
    rows = generate_rows(
        _model(), _VOCABULARY, _features(), _TIMING, GenerationConfig(nps=4.0, seed=1), _CPU
    )
    slots = [slot for slot, _ in rows]
    assert slots == sorted(slots)
    assert all(len(codes) == 4 for _, codes in rows)


def test_no_row_is_empty() -> None:
    # Where the rests go is the placement head's decision; a blank row would be the selection head
    # quietly overruling it.
    rows = generate_rows(
        _model(), _VOCABULARY, _features(), _TIMING, GenerationConfig(nps=4.0, seed=1), _CPU
    )
    assert all(any(codes) for _, codes in rows)


def test_the_same_seed_decodes_the_same_chart() -> None:
    features = _features()
    config = GenerationConfig(nps=4.0, seed=7)
    first = generate_rows(_model(), _VOCABULARY, features, _TIMING, config, _CPU)
    second = generate_rows(_model(), _VOCABULARY, features, _TIMING, config, _CPU)
    assert first == second


def test_a_different_seed_decodes_a_different_chart() -> None:
    features = _features(measures=16)
    model = _model()
    first = generate_rows(
        model, _VOCABULARY, features, _TIMING, GenerationConfig(nps=4.0, seed=1), _CPU
    )
    second = generate_rows(
        model, _VOCABULARY, features, _TIMING, GenerationConfig(nps=4.0, seed=2), _CPU
    )
    assert first != second


def test_a_song_too_short_to_hold_a_slot_decodes_to_nothing() -> None:
    # The decoder must return rather than index off the end of an empty list.
    empty = np.zeros((0, TOTAL_CHANNELS), dtype=np.float16)
    assert (
        generate_rows(
            _model(), _VOCABULARY, empty, _TIMING, GenerationConfig(nps=4.0, seed=1), _CPU
        )
        == []
    )


def test_a_rate_of_almost_nothing_still_places_one_note() -> None:
    # A chart with no notes at all is not a chart, so one is always kept.
    rows = generate_rows(
        _model(),
        _VOCABULARY,
        _features(measures=1),
        _TIMING,
        GenerationConfig(density=0.0, nps=1e-9, seed=1),
        _CPU,
    )
    assert len(rows) == 1


def test_every_freeze_is_closed() -> None:
    # A head with no tail makes its panel read as occupied for the rest of the song and stops the
    # file loading at all.
    rows = generate_rows(
        _model(),
        _VOCABULARY,
        _features(measures=16),
        _TIMING,
        GenerationConfig(holds=0.5, nps=4.0, seed=3),
        _CPU,
    )
    open_panels: set[int] = set()
    for _, codes in rows:
        for panel, code in enumerate(codes):
            if code == 2:
                open_panels.add(panel)
            elif code == _TAIL:
                open_panels.discard(panel)
    assert not open_panels


def test_a_freeze_still_open_at_the_end_is_closed_after_it() -> None:
    # Seed four leaves a freeze open when the last chosen slot is decoded, so a tail has to be
    # written a beat past it or the file will not load.
    rows = generate_rows(
        _model(),
        _VOCABULARY,
        _features(measures=16),
        _TIMING,
        GenerationConfig(holds=0.9, nps=6.0, seed=4),
        _CPU,
    )
    last_slot, last_codes = rows[-1]
    assert _TAIL in last_codes
    assert last_slot - rows[-2][0] == SUBDIVISIONS_PER_BEAT


def test_a_vocabulary_that_permits_nothing_still_writes_a_row() -> None:
    # Every pattern here is an orphan tail, which the mask bars at every level of relaxation.
    # Something still has to be placed.
    tails_only = Vocabulary([encode_row((3, 0, 0, 0)), encode_row((0, 3, 0, 0))])
    torch.manual_seed(0)
    model = ChartModel(len(tails_only), _SMALL).eval()
    rows = generate_rows(
        model, tails_only, _features(), _TIMING, GenerationConfig(nps=4.0, seed=1), _CPU
    )
    assert rows
    assert all(_TAIL in codes for _, codes in rows)


def test_a_panel_is_free_again_once_its_freeze_has_ended() -> None:
    # A freeze that is merely forgotten leaves its panel reading as occupied for the rest of the
    # song.
    rows = generate_rows(
        _model(),
        _VOCABULARY,
        _features(measures=16),
        _TIMING,
        GenerationConfig(holds=0.9, nps=6.0, seed=1),
        _CPU,
    )
    assert any(codes == [_TAIL, 0, 0, 0] for _, codes in rows)
    seen_tail = False
    stepped_after = False
    for _, codes in rows:
        if codes == [_TAIL, 0, 0, 0]:
            seen_tail = True
        elif seen_tail and codes[0] == 1:
            stepped_after = True
    assert stepped_after


def test_a_keyboard_chart_may_step_a_panel_again_sooner() -> None:
    # A foot has to travel; a finger does not, so the jack limit only applies to a chart meant to
    # be danced.
    features = _features(measures=16)
    model = _model()
    config = GenerationConfig(nps=8.0, seed=1)
    feet = generate_rows(model, _VOCABULARY, features, _TIMING, config, _CPU)
    keys = generate_rows(
        model,
        _VOCABULARY,
        features,
        _TIMING,
        GenerationConfig(nps=8.0, seed=1, style='keyboard'),
        _CPU,
    )
    assert _closest_repeat(keys) <= _closest_repeat(feet)


def test_notes_stay_on_the_sixteenth_grid_unless_triplets_are_asked_for() -> None:
    # Ranking every slot equally scatters notes onto twenty-fourths, which is what a chart full of
    # stray off-colour arrows is. Freeze tails are exempt: they end a freeze rather than being
    # struck, so they carry no colour.
    rows = generate_rows(
        _model(),
        _VOCABULARY,
        _features(measures=16),
        _TIMING,
        GenerationConfig(nps=6.0, seed=1),
        _CPU,
    )
    struck = [slot for slot, codes in rows if _TAIL not in codes]
    assert struck
    assert all(slot % SUBDIVISIONS_PER_BEAT % 3 == 0 for slot in struck)
