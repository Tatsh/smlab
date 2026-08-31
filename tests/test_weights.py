"""Tests for locating trained weights."""

from __future__ import annotations

from typing import TYPE_CHECKING
import hashlib
import sys

import pytest

from smlab.weights import (
    CHART_WEIGHTS,
    DEFAULT_RELEASE,
    DEFAULT_REPOSITORY,
    DIRECTORY_VARIABLE,
    RELEASE_VARIABLE,
    REPOSITORY_VARIABLE,
    URL_VARIABLE,
    WeightsError,
    download_directory,
    file_digest,
    resolve_weights,
    weights_directories,
    weights_release,
    weights_repository,
    weights_url,
)

if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import MagicMock

    from pytest_mock import MockerFixture

_PAYLOAD = b'weights' * 512


def _serve(
    mocker: MockerFixture, payload: bytes = _PAYLOAD, length: str | None = None
) -> MagicMock:
    response = mocker.MagicMock()
    response.headers.get.return_value = str(len(payload)) if length is None else length
    response.read.side_effect = [payload, b'']
    response.__enter__.return_value = response
    return mocker.patch('urllib.request.urlopen', return_value=response)


def test_repository_defaults() -> None:
    assert weights_repository() == DEFAULT_REPOSITORY


def test_repository_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(REPOSITORY_VARIABLE, 'someone/else')
    assert weights_repository() == 'someone/else'


def test_release_defaults() -> None:
    assert weights_release() == DEFAULT_RELEASE


def test_release_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RELEASE_VARIABLE, 'weights-2')
    assert weights_release() == 'weights-2'


def test_the_url_names_a_release_asset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(REPOSITORY_VARIABLE, 'someone/else')
    monkeypatch.setenv(RELEASE_VARIABLE, 'v9.9.9')
    assert weights_url(CHART_WEIGHTS) == (
        f'https://github.com/someone/else/releases/download/v9.9.9/{CHART_WEIGHTS}'
    )


@pytest.mark.parametrize('base', ['https://mirror.example/smlab', 'https://mirror.example/smlab/'])
def test_the_url_can_point_at_a_mirror(base: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(URL_VARIABLE, base)
    assert weights_url(CHART_WEIGHTS) == f'https://mirror.example/smlab/{CHART_WEIGHTS}'


def test_the_search_covers_user_and_system_directories(tmp_path: Path) -> None:
    directories = weights_directories(tmp_path)
    assert directories[0] == tmp_path
    assert download_directory() in directories
    assert any(directory.parts[-2:] == ('share', 'smlab') for directory in directories)


def test_the_search_lists_each_directory_once(tmp_path: Path) -> None:
    directories = weights_directories(tmp_path)
    assert len(set(directories)) == len(directories)


def test_override_directory_wins(tmp_path: Path) -> None:
    weights = tmp_path / CHART_WEIGHTS
    weights.write_bytes(b'x')
    assert resolve_weights(CHART_WEIGHTS, tmp_path) == weights


def test_override_may_name_the_file_directly(tmp_path: Path) -> None:
    weights = tmp_path / 'other.pt'
    weights.write_bytes(b'x')
    assert resolve_weights('other.pt', weights) == weights


def test_the_environment_can_name_a_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    installed = tmp_path / 'installed'
    installed.mkdir()
    (installed / CHART_WEIGHTS).write_bytes(b'x')
    monkeypatch.setenv(DIRECTORY_VARIABLE, str(installed))
    assert resolve_weights(CHART_WEIGHTS) == installed / CHART_WEIGHTS


def test_the_user_data_directory_is_searched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'data'))
    installed = tmp_path / 'data' / 'smlab'
    installed.mkdir(parents=True)
    (installed / CHART_WEIGHTS).write_bytes(b'x')
    assert resolve_weights(CHART_WEIGHTS) == installed / CHART_WEIGHTS


def test_a_system_directory_is_searched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('XDG_DATA_DIRS', str(tmp_path / 'usr' / 'share'))
    installed = tmp_path / 'usr' / 'share' / 'smlab'
    installed.mkdir(parents=True)
    (installed / CHART_WEIGHTS).write_bytes(b'x')
    assert resolve_weights(CHART_WEIGHTS) == installed / CHART_WEIGHTS


def test_the_installation_prefix_is_searched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'prefix', str(tmp_path / 'prefix'))
    installed = tmp_path / 'prefix' / 'share' / 'smlab'
    installed.mkdir(parents=True)
    (installed / CHART_WEIGHTS).write_bytes(b'x')
    assert resolve_weights(CHART_WEIGHTS) == installed / CHART_WEIGHTS


def test_refusing_download_names_what_was_searched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(WeightsError, match='not permitted') as error:
        resolve_weights(CHART_WEIGHTS, tmp_path / 'empty', allow_download=False)
    assert str(download_directory()) in str(error.value)


def test_a_download_lands_in_the_cache(
    tmp_path: Path, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _serve(mocker)
    resolved = resolve_weights('other.pt', tmp_path / 'empty')
    assert resolved == download_directory() / 'other.pt'
    assert resolved.read_bytes() == _PAYLOAD


def test_a_download_is_checked_against_the_bundled_digest(
    tmp_path: Path, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _serve(mocker)
    mocker.patch(
        'smlab.weights.load_checksums',
        return_value={CHART_WEIGHTS: hashlib.sha256(_PAYLOAD).hexdigest()},
    )
    assert resolve_weights(CHART_WEIGHTS, tmp_path / 'empty').read_bytes() == _PAYLOAD


def test_a_download_that_does_not_match_is_discarded(
    tmp_path: Path, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _serve(mocker)
    with pytest.raises(WeightsError, match='SHA-256'):
        resolve_weights(CHART_WEIGHTS, tmp_path / 'empty')
    assert not list(download_directory().iterdir())


def test_a_failed_download_leaves_nothing_behind(
    tmp_path: Path, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    mocker.patch('urllib.request.urlopen', side_effect=OSError('no network'))
    with pytest.raises(WeightsError, match='--checkpoints'):
        resolve_weights('other.pt', tmp_path / 'empty')
    assert not list(download_directory().iterdir())


def test_an_insecure_address_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(URL_VARIABLE, 'http://mirror.example/smlab')
    with pytest.raises(WeightsError, match='insecure'):
        resolve_weights('other.pt', tmp_path / 'empty')


def test_progress_is_reported_while_downloading(
    tmp_path: Path, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _serve(mocker)
    progress = mocker.Mock()
    resolve_weights('other.pt', tmp_path / 'empty', progress=progress)
    assert progress.call_args_list == [mocker.call('other.pt', len(_PAYLOAD), len(_PAYLOAD))]


def test_progress_copes_with_an_unstated_length(
    tmp_path: Path, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _serve(mocker, length='')
    progress = mocker.Mock()
    resolve_weights('other.pt', tmp_path / 'empty', progress=progress)
    assert progress.call_args_list == [mocker.call('other.pt', len(_PAYLOAD), len(_PAYLOAD))]


def test_the_digest_of_a_file_is_its_sha256(tmp_path: Path) -> None:
    path = tmp_path / 'file'
    path.write_bytes(_PAYLOAD)
    assert file_digest(path) == hashlib.sha256(_PAYLOAD).hexdigest()
