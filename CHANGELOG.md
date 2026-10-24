<!-- markdownlint-configure-file {"MD024": { "siblings_only": true } } -->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.1/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [unreleased]

### Added

- Separation runs on Apple GPUs (MPS) when CUDA is absent.

### Fixed

- Detected tempi were always rounded to a whole or half BPM, and charts for songs at other tempi
  drifted by up to half a beat. The tempo is now refined against the whole song and rounded only
  when the round value fits as well.

## [0.0.1] - 2026-08-31

First version.

[unreleased]: https://github.com/Tatsh/smlab/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/Tatsh/smlab/releases/tag/v0.0.1
