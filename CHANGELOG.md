# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- GitHub Actions CI to run pytest on every push and pull request (Python 3.10–3.12).
- Pre-commit hook that runs pytest before each commit.
- pip-installable packaging via `pyproject.toml` with `called` CLI entry point.
- `.gitignore` for Python projects.
- `README.md` with features, quick start, usage, and architecture overview.
- GPL-3.0-or-later license.
- GitHub repository at [github.com/godie/called](https://github.com/godie/called).

## [0.1.0] — 2026-06-02

### Added

- Initial release.
- Real-time audio transcription using faster-whisper.
- Dual audio capture: microphone and system audio (BlackHole on macOS).
- Silence detection with adaptive threshold recalibration.
- Checkpoint-based crash recovery.
- Compressed audio archiving to Opus/OGG.
- Multiple output formats: TXT, JSON, SRT.
- 135 tests across all subsystems.

[Unreleased]: https://github.com/godie/called/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/godie/called/releases/tag/v0.1.0
