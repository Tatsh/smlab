"""Tests for the models bundled with the package."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from smlab.preview import PreviewModel
from smlab.resources import (
    PREVIEW_ASSET,
    VOCABULARY_ASSET,
    asset_bytes,
    has_asset,
    load_checksums,
    load_state_dict,
    load_vocabulary,
)
from smlab.weights import CHART_WEIGHTS, OFFSET_WEIGHTS

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

_CPU = torch.device('cpu')


@pytest.mark.parametrize('name', [PREVIEW_ASSET, VOCABULARY_ASSET])
def test_every_model_is_bundled(name: str) -> None:
    assert has_asset(name)
    assert len(asset_bytes(name)) > 0


def test_missing_asset_is_reported() -> None:
    assert not has_asset('nonexistent.pt')
    with pytest.raises(FileNotFoundError, match='not bundled'):
        asset_bytes('nonexistent.pt')


def test_bundled_vocabulary_loads() -> None:
    vocabulary = load_vocabulary(None)
    assert len(vocabulary) > 0
    # The commonest row in any DDR corpus is a single tap.
    assert sum(1 for code in vocabulary.panels_of(0) if code) == 1


def test_bundled_weights_fit_their_model() -> None:
    PreviewModel().load_state_dict(load_state_dict(PREVIEW_ASSET, None, _CPU))


def test_local_checkpoint_overrides_the_bundled_one(tmp_path: Path) -> None:
    model = PreviewModel()
    for tensor in model.state_dict().values():
        tensor.zero_()
    torch.save(model.state_dict(), tmp_path / PREVIEW_ASSET)
    loaded = load_state_dict(PREVIEW_ASSET, tmp_path, _CPU)
    assert all(not tensor.any() for tensor in loaded.values())


def test_absent_local_directory_falls_back_to_bundled(tmp_path: Path) -> None:
    loaded = load_state_dict(PREVIEW_ASSET, tmp_path / 'nothing-here', _CPU)
    assert any(float(tensor.abs().sum()) > 0.0 for tensor in loaded.values())


def test_local_vocabulary_overrides_the_bundled_one(tmp_path: Path) -> None:
    path = tmp_path / 'vocabulary.json'
    path.write_text('[343, 49]')
    assert load_vocabulary(path).patterns == (343, 49)


def test_every_downloaded_checkpoint_has_a_bundled_digest() -> None:
    checksums = load_checksums()
    assert set(checksums) == {CHART_WEIGHTS, OFFSET_WEIGHTS}
    assert all(len(digest) == 64 for digest in checksums.values())


def test_checksums_are_absent_when_the_manifest_is_not_bundled(mocker: MockerFixture) -> None:
    mocker.patch('smlab.resources.asset_bytes', side_effect=FileNotFoundError)
    assert load_checksums() == {}


def test_lines_that_are_not_digests_are_ignored(mocker: MockerFixture) -> None:
    mocker.patch(
        'smlab.resources.asset_bytes',
        return_value=b'# a comment\n\n' + (b'0' * 64) + b' *chart.pt\n',
    )
    assert load_checksums() == {CHART_WEIGHTS: '0' * 64}


def test_a_missing_asset_package_is_reported_as_absent(mocker: MockerFixture) -> None:
    # An installation stripped of its assets must degrade rather than explode.
    mocker.patch('smlab.resources.resources.files', side_effect=ModuleNotFoundError)
    assert not has_asset(PREVIEW_ASSET)
