<!-- markdownlint-configure-file {"MD024": { "siblings_only": true } } -->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.1/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [unreleased]

## [0.1.0] - 2026-10-28

### Added

- `generate` places the beat on the kick drum of the separated drum stem, and retains the offset
  model for choosing the downbeat and for songs without a clear kick.
- `generate` separates on Apple GPUs (MPS) when CUDA is absent. The training stem cache still uses
  CUDA or CPU.

### Changed

- `song_features` takes the stems from the new `separate_song` instead of a separator and device.
- `refine_offset` returns an `OffsetEstimate` (`clarity`, `offset`, `source`, `weight`) instead of
  an `(offset, weight)` tuple. Unpacking the result into two names raises `ValueError`.
  `refine_offset` takes an optional drum stem.
- `predict_phase` returns a `PhaseEstimate` named tuple. The two fields of `PhaseEstimate` are the
  bin and weight that `predict_phase` returned before.

### Fixed

- Detected tempi were always rounded to a whole or half BPM, and charts for songs at other tempi
  drifted by up to half a beat. Tempo is now refined against the whole song and rounded only when
  the round value fits as well.

## [0.0.1] - 2026-08-31

First version.

[unreleased]: https://github.com/Tatsh/smlab/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Tatsh/smlab/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/Tatsh/smlab/releases/tag/v0.0.1
