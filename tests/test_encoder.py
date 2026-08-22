"""Tests for the shared audio encoder."""

from __future__ import annotations

import pytest
import torch

from smlab.encoder import MEASURE_SLOTS, STYLES, AudioEncoder, EncoderConfig
from smlab.features import TOTAL_CHANNELS

_SMALL = EncoderConfig(
    channels=32, model_dimension=24, local_blocks=1, slot_layers=1, measure_layers=1, heads=2
)


def _conditioning(batch: int) -> tuple[torch.Tensor, ...]:
    zeros = torch.zeros(batch, dtype=torch.long)
    return zeros, torch.full((batch,), 9), zeros, zeros, torch.full((batch,), 4)


@pytest.mark.parametrize(
    'fine_slots',
    [
        MEASURE_SLOTS * 2 * 4,  # whole measures
        MEASURE_SLOTS * 2 * 4 + 8,  # partial final measure
        MEASURE_SLOTS * 2,  # exactly one measure
        MEASURE_SLOTS * 2 - 2,  # shorter than one measure
        2,  # degenerate
    ],
)
def test_any_length_produces_one_vector_per_note_slot(fine_slots: int) -> None:
    # A song rarely ends on a bar line, so the encoder must handle a partial
    # final measure. Every training window is a whole number of measures, so
    # this is only ever exercised at generation time.
    model = AudioEncoder(_SMALL).eval()
    features = torch.randn(1, fine_slots, TOTAL_CHANNELS)
    with torch.no_grad():
        encoded = model(features, *_conditioning(1))
    assert encoded.shape == (1, fine_slots // 2, _SMALL.model_dimension)
    assert torch.isfinite(encoded).all()


def test_batches_are_encoded_independently() -> None:
    model = AudioEncoder(_SMALL).eval()
    features = torch.randn(3, MEASURE_SLOTS * 2 * 2, TOTAL_CHANNELS)
    with torch.no_grad():
        encoded = model(features, *_conditioning(3))
    assert encoded.shape[0] == 3


def test_conditioning_changes_the_encoding() -> None:
    # Style must reach the output, or asking for a keyboard chart would return
    # a pad chart.
    model = AudioEncoder(_SMALL).eval()
    features = torch.randn(1, MEASURE_SLOTS * 2 * 2, TOTAL_CHANNELS)
    zeros = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        rate = torch.full((1,), 4)
        feet = model(features, zeros, torch.full((1,), 9), zeros, zeros, rate)
        keyboard = model(
            features,
            zeros,
            torch.full((1,), 9),
            zeros,
            torch.full((1,), STYLES.index('keyboard')),
            rate,
        )
    assert not torch.allclose(feet, keyboard)


def test_difficulty_changes_the_encoding() -> None:
    model = AudioEncoder(_SMALL).eval()
    features = torch.randn(1, MEASURE_SLOTS * 2 * 2, TOTAL_CHANNELS)
    zeros = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        rate = torch.full((1,), 4)
        easy = model(features, torch.full((1,), 1), torch.full((1,), 3), zeros, zeros, rate)
        hard = model(features, torch.full((1,), 3), torch.full((1,), 12), zeros, zeros, rate)
    assert not torch.allclose(easy, hard)


def test_note_rate_changes_the_encoding() -> None:
    # A rating is a lossy description of density: the classic scale saturates at
    # the top and a keyboard chart runs twice as dense as a pad chart carrying
    # the same number. The measured rate has to reach the output for the model
    # to learn density directly rather than inferring it from that number.
    model = AudioEncoder(_SMALL).eval()
    features = torch.randn(1, MEASURE_SLOTS * 2 * 2, TOTAL_CHANNELS)
    zeros = torch.zeros(1, dtype=torch.long)
    meter = torch.full((1,), 9)
    with torch.no_grad():
        sparse = model(features, zeros, meter, zeros, zeros, torch.full((1,), 2))
        dense = model(features, zeros, meter, zeros, zeros, torch.full((1,), 9))
    assert not torch.allclose(sparse, dense)
