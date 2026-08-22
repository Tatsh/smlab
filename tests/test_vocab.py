"""Tests for the note-row pattern vocabulary."""

from __future__ import annotations

from typing import TYPE_CHECKING
import json

from smlab.vocab import Vocabulary, build_vocabulary, coverage, decode_row, encode_row
import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path


def _write_entry(root: Path, charts: list[dict[str, np.ndarray]]) -> None:
    shard = root / 'aa'
    shard.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {'features': np.zeros((8, 4), dtype=np.float16)}
    meta = []
    for index, chart in enumerate(charts):
        arrays[f'panels_{index}'] = chart['panels']
        arrays[f'slots_{index}'] = np.arange(len(chart['panels']), dtype=np.int32)
        meta.append({'difficulty': 'Challenge', 'index': index, 'meter': 9})
    arrays['meta'] = np.asarray(json.dumps(meta, sort_keys=True))
    np.savez(shard / 'aa.npz', **arrays)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


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


def test_a_vocabulary_is_built_from_the_cache(tmp_path: Path) -> None:
    # Frequency order matters: the commonest pattern becomes the fallback token.
    charts = [{'panels': np.array([[1, 0, 0, 0]] * 5 + [[0, 1, 0, 0]] * 2, dtype=np.uint8)}]
    _write_entry(tmp_path, charts)
    built = build_vocabulary(tmp_path)
    assert built.panels_of(0) == (1, 0, 0, 0)
    assert built.panels_of(1) == (0, 1, 0, 0)


def test_the_empty_row_is_never_in_the_vocabulary(tmp_path: Path) -> None:
    # It is the commonest pattern in any chart and the one the selection head
    # must never be able to choose.
    charts = [{'panels': np.array([[0, 0, 0, 0]] * 50 + [[1, 0, 0, 0]], dtype=np.uint8)}]
    _write_entry(tmp_path, charts)
    built = build_vocabulary(tmp_path)
    assert len(built) == 1
    assert built.panels_of(0) == (1, 0, 0, 0)


def test_the_vocabulary_is_truncated_to_the_limit(tmp_path: Path) -> None:
    rows = [[code, 0, 0, 0] for code in (1, 2, 4, 6)]
    _write_entry(tmp_path, [{'panels': np.array(rows, dtype=np.uint8)}])
    assert len(build_vocabulary(tmp_path, limit=2)) == 2


def test_an_unreadable_cache_entry_is_skipped(tmp_path: Path) -> None:
    _write_entry(tmp_path, [{'panels': np.array([[1, 0, 0, 0]], dtype=np.uint8)}])
    broken = tmp_path / 'zz'
    broken.mkdir()
    (broken / 'zz.npz').write_bytes(b'not an npz archive')
    assert len(build_vocabulary(tmp_path)) == 1
