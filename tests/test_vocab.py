"""Tests for the note-row pattern vocabulary."""

from __future__ import annotations

from typing import TYPE_CHECKING

from smlab.vocab import Vocabulary, coverage, decode_row, encode_row
import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    'codes',
    [
        (0, 0, 0, 0),
        (1, 0, 0, 0),
        (0, 1, 1, 0),
        (2, 0, 0, 3),
        (6, 6, 6, 6),
        (1, 2, 3, 4),
    ],
)
def test_row_encoding_round_trips(codes: tuple[int, int, int, int]) -> None:
    assert decode_row(encode_row(codes)) == codes


def test_distinct_rows_get_distinct_tokens() -> None:
    tokens = {
        encode_row((a, b, c, d))
        for a in range(3)
        for b in range(3)
        for c in range(3)
        for d in range(3)
    }
    assert len(tokens) == 81


def test_vocabulary_indexes_by_frequency_order() -> None:
    vocabulary = Vocabulary([encode_row((0, 1, 0, 0)), encode_row((1, 0, 0, 0))])
    assert vocabulary.token_for((0, 1, 0, 0)) == 0
    assert vocabulary.token_for((1, 0, 0, 0)) == 1
    assert vocabulary.panels_of(0) == (0, 1, 0, 0)


def test_unknown_row_falls_back_to_commonest() -> None:
    vocabulary = Vocabulary([encode_row((0, 1, 0, 0))])
    assert vocabulary.token_for((1, 1, 1, 1)) == 0


@pytest.mark.parametrize(
    ('codes', 'expected'),
    [
        ((1, 0, 0, 0), {0}),
        ((0, 1, 1, 0), {1, 2}),
        ((2, 0, 0, 0), {0}),
        ((3, 0, 0, 0), set()),
        ((0, 0, 0, 5), set()),
        ((1, 0, 0, 1), {0, 3}),
    ],
)
def test_stepped_panels_excludes_tails_and_mines(
    codes: tuple[int, ...], expected: set[int]
) -> None:
    vocabulary = Vocabulary([encode_row(codes)])
    assert vocabulary.stepped_panels(0) == expected


def test_vocabulary_persists(tmp_path: Path) -> None:
    original = Vocabulary([encode_row((1, 0, 0, 0)), encode_row((0, 0, 0, 1))])
    path = tmp_path / 'vocab.json'
    original.save(path)
    assert Vocabulary.load(path).patterns == original.patterns


def test_coverage_reports_share_of_rows() -> None:
    counts = [(1, 70), (2, 20), (3, 10)]
    assert coverage(counts, 1) == pytest.approx(0.7)
    assert coverage(counts, 2) == pytest.approx(0.9)
    assert coverage(counts, 3) == pytest.approx(1.0)
    assert coverage([], 5) == pytest.approx(0.0)
