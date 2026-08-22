"""Tests for splitting a song into stems."""

from __future__ import annotations

from typing import TYPE_CHECKING
import sys

from smlab.stems import STEM_NAMES, SeparationError, load_separator, separate
import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]  # No stubs are published.
import torch

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

_RATE = 22050
_CPU = torch.device('cpu')


class _Model:
    """Stands in for a demucs model, which is far too slow to run in a test."""

    def __init__(self, sources: tuple[str, ...] = STEM_NAMES) -> None:
        self.samplerate = _RATE
        self.sources = list(sources)

    def to(self, device: torch.device) -> _Model:
        del device
        return self

    def eval(self) -> _Model:
        return self


def _audio(tmp_path: Path, *, channels: int = 2, seconds: float = 0.5) -> Path:
    path = tmp_path / 'song.wav'
    shape = (int(_RATE * seconds), channels) if channels > 1 else int(_RATE * seconds)
    sf.write(path, np.zeros(shape, dtype='float32'), _RATE)
    return path


def _separated(sources: int = len(STEM_NAMES), seconds: float = 0.5) -> torch.Tensor:
    return torch.zeros(1, sources, 2, int(_RATE * seconds))


def test_the_model_is_loaded_onto_the_device(mocker: MockerFixture) -> None:
    get_model = mocker.patch('demucs.pretrained.get_model', return_value=_Model())
    assert load_separator(_CPU).samplerate == _RATE
    get_model.assert_called_once_with('htdemucs')


def test_a_missing_extra_is_reported_as_a_separation_error(mocker: MockerFixture) -> None:
    # Separation is an optional dependency, so the failure has to name the cure.
    mocker.patch.dict(sys.modules, {'demucs.pretrained': None})
    with pytest.raises(SeparationError, match='stems extra'):
        load_separator(_CPU)


def test_every_stem_comes_back_as_mono(tmp_path: Path, mocker: MockerFixture) -> None:
    mocker.patch('demucs.apply.apply_model', return_value=_separated())
    stems = separate(_Model(), _audio(tmp_path), _CPU)
    assert sorted(stems) == sorted(STEM_NAMES)
    assert all(value.ndim == 1 for value in stems.values())
    assert all(value.dtype == np.float32 for value in stems.values())


def test_mono_input_is_widened_before_separating(tmp_path: Path, mocker: MockerFixture) -> None:
    # Demucs expects two channels, so a mono file has to be duplicated.
    apply_model = mocker.patch('demucs.apply.apply_model', return_value=_separated())
    separate(_Model(), _audio(tmp_path, channels=1), _CPU)
    assert apply_model.call_args.args[1].shape[1] == 2


def test_a_source_the_model_does_not_emit_is_left_out(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    mocker.patch('demucs.apply.apply_model', return_value=_separated(sources=2))
    stems = separate(_Model(sources=('drums', 'bass')), _audio(tmp_path), _CPU)
    assert sorted(stems) == ['bass', 'drums']


def test_unreadable_audio_is_reported(tmp_path: Path, mocker: MockerFixture) -> None:
    mocker.patch('demucs.apply.apply_model', return_value=_separated())
    missing = tmp_path / 'nothing.wav'
    missing.write_bytes(b'not audio')
    with pytest.raises(SeparationError, match='could not read audio'):
        separate(_Model(), missing, _CPU)


def test_a_separation_failure_is_reported(tmp_path: Path, mocker: MockerFixture) -> None:
    mocker.patch('demucs.apply.apply_model', side_effect=RuntimeError('out of memory'))
    with pytest.raises(SeparationError, match='separation failed'):
        separate(_Model(), _audio(tmp_path), _CPU)
