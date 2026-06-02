# Contributing to called

Thanks for your interest in contributing! called is a real-time audio transcription tool using faster-whisper. Contributions of all kinds are welcome — bug reports, feature ideas, documentation, and code.

## Getting started

```bash
git clone https://github.com/godie/called.git
cd called
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

### Running tests

All changes should pass the test suite:

```bash
pytest tests/ -q
```

The pre-commit hook also runs tests automatically before each commit. If tests fail, inspect the output and fix before pushing.

### Code style

- Follow **PEP 8**.
- Use type hints where practical (Python 3.10+ syntax).
- Match the style and patterns of the surrounding code.
- Keep functions focused; prefer composition over large monoliths.
- Add tests for new functionality in `tests/`.

## How to contribute

### Reporting bugs

Open an issue using the **Bug Report** template. Include:

- What happened vs. what you expected
- Steps to reproduce
- Python version (`python --version`)
- Operating system
- Any relevant logs or terminal output

### Suggesting features

Open an issue using the **Feature Request** template. Describe the use case, how you'd expect it to work, and any alternatives you've considered.

### Pull requests

1. **Fork** the repository and create a branch from `main`.
2. Keep changes focused — one feature or fix per PR.
3. Add or update tests to cover your changes.
4. Ensure all tests pass (`pytest tests/`).
5. Update documentation if your change affects usage or config.
6. Open a PR against `main`. Link any related issues.

#### PR labels

GitHub Actions will label your PR automatically. The labels map to release note categories:

| Label | Section |
|-------|---------|
| `feature`, `enhancement` | 🚀 New Features |
| `bug`, `fix` | 🐛 Bug Fixes |
| `documentation`, `docs` | 📚 Documentation |
| `test`, `testing` | 🧪 Testing |
| `performance`, `perf` | ⚡ Performance |

## Project structure

```
called/
├── app.py                  # Application orchestrator
├── config.py               # Environment-based configuration
├── audio/                  # Audio capture, silence detection, archiving
├── transcription/          # Whisper model and async consumer
├── storage/                # Checkpoints, output writers
├── cli/                    # CLI interface
├── utils/                  # Logging and utilities
├── tests/                  # pytest test suite
├── pyproject.toml          # Package metadata, deps, pytest config
└── .github/                # CI, issue/PR templates, release notes config
```

## License

By contributing, you agree that your contributions will be licensed under the [GNU GPL v3.0 or later](LICENSE).
