"""Tests for how a rating becomes a note rate."""

from __future__ import annotations

import pytest

from smlab.generate import (
    CLASSIC_SCALE,
    DEFAULT_SCALE,
    NPS_BY_METER_10,
    NPS_BY_METER_15,
    NPS_BY_METER_20,
    NPS_BY_METER_KEYBOARD,
    SCALES,
    GenerationConfig,
    target_nps,
)


@pytest.mark.parametrize(
    ('scale', 'table'),
    list(zip(SCALES, (NPS_BY_METER_10, NPS_BY_METER_15, NPS_BY_METER_20), strict=True)),
)
def test_each_scale_reads_its_own_table(scale: int, table: tuple[float, ...]) -> None:
    assert target_nps(5, scale) == pytest.approx(table[5])


def test_the_same_rating_means_different_speeds_on_different_scales() -> None:
    # A nine is a much harder chart in ITG than on the modern twenty-point one.
    assert target_nps(9, 15) > target_nps(9, CLASSIC_SCALE) > target_nps(9, 20)


def test_a_rating_above_the_scale_is_clamped() -> None:
    assert target_nps(999, 20) == pytest.approx(NPS_BY_METER_20[-1])


def test_a_rating_below_the_scale_is_clamped() -> None:
    assert target_nps(-4, 20) == pytest.approx(NPS_BY_METER_20[0])


def test_an_unknown_scale_rounds_to_the_nearest_known_one() -> None:
    assert target_nps(7, 21) == pytest.approx(target_nps(7, 20))
    assert target_nps(7, 11) == pytest.approx(target_nps(7, CLASSIC_SCALE))


def test_the_default_scale_is_used_when_none_is_given() -> None:
    assert target_nps(9) == pytest.approx(target_nps(9, DEFAULT_SCALE))


def test_a_configuration_reads_its_rate_from_its_rating() -> None:
    assert GenerationConfig(meter=9, scale=20).rate == pytest.approx(target_nps(9, 20))


def test_an_explicit_rate_overrides_the_rating() -> None:
    # The classic scale saturates, so stating the rate outright has to win.
    assert GenerationConfig(meter=9, nps=7.5).rate == pytest.approx(7.5)


def test_a_keyboard_chart_reads_a_scale_of_its_own() -> None:
    # A keyboard chart at a given rating runs roughly twice as dense.
    keyboard = GenerationConfig(meter=9, style='keyboard')
    assert keyboard.rate == pytest.approx(NPS_BY_METER_KEYBOARD[9])
    assert keyboard.rate > GenerationConfig(meter=9, style='feet').rate


def test_a_keyboard_rating_above_the_table_is_clamped() -> None:
    assert GenerationConfig(meter=999, style='keyboard').rate == pytest.approx(
        NPS_BY_METER_KEYBOARD[-1]
    )


def test_a_keyboard_chart_is_allowed_more_jumps() -> None:
    assert GenerationConfig(style='keyboard').jump_share > GenerationConfig(style='feet').jump_share


def test_an_explicit_jump_share_overrides_the_style() -> None:
    assert GenerationConfig(jumps=0.3, style='feet').jump_share == pytest.approx(0.3)
