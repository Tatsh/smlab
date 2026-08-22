"""Tests for locating trained weights."""

from __future__ import annotations

from typing import TYPE_CHECKING
import sys

from smlab.weights import (
    CHART_WEIGHTS,
    DEFAULT_REPOSITORY,
    REPOSITORY_VARIABLE,
    REVISION_VARIABLE,
    WeightsError,
    resolve_weights,
    weights_repository,
    weights_revision,
)
import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


def test_repository_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(REPOSITORY_VARIABLE, raising=False)
    assert weights_repository() == DEFAULT_REPOSITORY


def test_repository_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(REPOSITORY_VARIABLE, 'someone/else')
    assert weights_repository() == 'someone/else'


def test_revision_is_unset_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(REVISION_VARIABLE, raising=False)
    assert weights_revision() is None


def test_revision_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(REVISION_VARIABLE, 'v2')
    assert weights_revision() == 'v2'


def test_override_directory_wins(tmp_path: Path) -> None:
    weights = tmp_path / CHART_WEIGHTS
    weights.write_bytes(b'x')
    assert resolve_weights(CHART_WEIGHTS, tmp_path) == weights


def test_override_may_name_the_file_directly(tmp_path: Path) -> None:
    weights = tmp_path / 'other.pt'
    weights.write_bytes(b'x')
    assert resolve_weights('other.pt', weights) == weights


def test_refusing_download_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(WeightsError, match='not permitted'):
        resolve_weights(CHART_WEIGHTS, tmp_path / 'empty', allow_download=False)


def test_download_is_used_when_nothing_is_local(
    tmp_path: Path, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    fetched = tmp_path / 'fetched.pt'
    fetched.write_bytes(b'x')
    download = mocker.patch('huggingface_hub.hf_hub_download', return_value=str(fetched))
    assert resolve_weights(CHART_WEIGHTS, tmp_path / 'empty') == fetched
    assert download.call_args.kwargs['filename'] == CHART_WEIGHTS


def test_download_failure_explains_the_options(
    tmp_path: Path, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    mocker.patch('huggingface_hub.hf_hub_download', side_effect=OSError('no network'))
    with pytest.raises(WeightsError, match='--checkpoints'):
        resolve_weights(CHART_WEIGHTS, tmp_path / 'empty')


def test_download_honours_the_configured_repository(
    tmp_path: Path, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(REPOSITORY_VARIABLE, 'someone/else')
    monkeypatch.setenv(REVISION_VARIABLE, 'v3')
    fetched = tmp_path / 'fetched.pt'
    fetched.write_bytes(b'x')
    download = mocker.patch('huggingface_hub.hf_hub_download', return_value=str(fetched))
    resolve_weights(CHART_WEIGHTS, tmp_path / 'empty')
    assert download.call_args.kwargs['repo_id'] == 'someone/else'
    assert download.call_args.kwargs['revision'] == 'v3'


def test_the_hub_client_being_absent_is_explained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    # Downloading is the fallback when nothing is local, so its absence has to
    # point at the other way of supplying weights.
    monkeypatch.delenv(REPOSITORY_VARIABLE, raising=False)
    monkeypatch.chdir(tmp_path)
    mocker.patch.dict(sys.modules, {'huggingface_hub': None})
    with pytest.raises(WeightsError, match='huggingface-hub is needed'):
        resolve_weights('chart.pt', None)
