"""Unit tests for checkpoint recovery system."""

import json
import tempfile
import time
from pathlib import Path

from storage.checkpoint import find_latest_checkpoint, load_checkpoint_data


class TestFindLatestCheckpoint:
    """Tests for finding the most recent checkpoint file."""

    def test_empty_dir_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = find_latest_checkpoint(Path(tmpdir))
            assert result is None

    def test_nonexistent_dir_returns_none(self) -> None:
        result = find_latest_checkpoint(Path("/nonexistent/checkpoints"))
        assert result is None

    def test_finds_latest_by_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            # Create two checkpoints with different ages
            older = d / "checkpoint_20240101_120000.json"
            newer = d / "checkpoint_20240101_120100.json"

            older.write_text(json.dumps({"entries": [], "saved_at": 1}))
            newer.write_text(json.dumps({"entries": [], "saved_at": 2}))

            # Touch newer to ensure it has later mtime
            time.sleep(0.01)
            newer.touch()

            result = find_latest_checkpoint(d)
            assert result is not None
            assert result.name == "checkpoint_20240101_120100.json"

    def test_ignores_non_checkpoint_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "other_file.txt").write_text("not a checkpoint")
            (d / "checkpoint_20240101_120000.json").write_text(
                json.dumps({"entries": [], "saved_at": 1})
            )

            result = find_latest_checkpoint(d)
            assert result is not None
            assert "checkpoint" in result.name

    def test_single_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            path = d / "checkpoint_20240101_120000.json"
            path.write_text(json.dumps({"entries": [], "saved_at": 1}))

            result = find_latest_checkpoint(d)
            assert result is not None
            assert result == path


class TestLoadCheckpointData:
    """Tests for loading checkpoint JSON data."""

    def test_loads_valid_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.json"
            data = {
                "session_start": 1000.0,
                "saved_at": 2000.0,
                "entries": [
                    {"timestamp": 1100.0, "text": "Hello", "language": "en"},
                    {"timestamp": 1105.0, "text": "World", "language": "en"},
                ],
                "metrics": {"chunks_processed": 10, "chunks_dropped": 2},
                "flushed_entries": 0,
            }
            path.write_text(json.dumps(data))

            loaded = load_checkpoint_data(path)
            assert loaded is not None
            assert loaded["session_start"] == 1000.0
            assert len(loaded["entries"]) == 2
            assert loaded["entries"][0]["text"] == "Hello"
            assert loaded["metrics"]["chunks_processed"] == 10

    def test_returns_none_for_missing_file(self) -> None:
        path = Path("/nonexistent/checkpoint.json")
        result = load_checkpoint_data(path)
        assert result is None

    def test_returns_none_for_corrupt_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "corrupt.json"
            path.write_text("{invalid json")

            result = load_checkpoint_data(path)
            assert result is None

    def test_returns_none_for_missing_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.json"
            path.write_text(json.dumps({"saved_at": 1000}))

            result = load_checkpoint_data(path)
            assert result is None

    def test_returns_none_for_null_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.json"
            path.write_text(json.dumps({"entries": None, "saved_at": 1000}))

            result = load_checkpoint_data(path)
            assert result is None

    def test_empty_entries_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.json"
            path.write_text(json.dumps({"entries": [], "saved_at": 1000}))

            loaded = load_checkpoint_data(path)
            assert loaded is not None
            assert loaded["entries"] == []

    def test_loads_checkpoint_with_unicode_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "unicode.json"
            data = {
                "entries": [
                    {"timestamp": 0.0, "text": "Héllo wörld 日本語"},
                ],
                "saved_at": 1000,
            }
            path.write_text(json.dumps(data, ensure_ascii=False))

            loaded = load_checkpoint_data(path)
            assert loaded is not None
            assert loaded["entries"][0]["text"] == "Héllo wörld 日本語"
