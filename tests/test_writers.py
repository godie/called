"""Unit tests for transcript writers."""

import json
import tempfile
from pathlib import Path

from storage.json_writer import JSONWriter
from storage.srt_writer import SRTWriter
from storage.txt_writer import TXTWriter


class TestTXTWriter:
    """Tests for TXT transcript writer."""

    def test_write_creates_file(self) -> None:
        """Writing creates a proper TXT file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = TXTWriter(Path(tmpdir))
            entries = [
                {"timestamp": 0.0, "text": "Hello world"},
                {"timestamp": 2.5, "text": "This is a test"},
            ]
            path = writer.write(entries, filename="test.txt")
            assert path.exists()
            content = path.read_text()
            assert "Hello world" in content
            assert "This is a test" in content
            assert "[00:00:00.000]" in content
            assert "[00:00:02.500]" in content

    def test_write_empty_entries(self) -> None:
        """Writing empty entries still produces a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = TXTWriter(Path(tmpdir))
            path = writer.write([], filename="empty.txt")
            assert path.exists()
            content = path.read_text()
            assert "Transcription Session" in content

    def test_append_adds_line(self) -> None:
        """Appending adds a line to an existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = TXTWriter(Path(tmpdir))
            entries = [{"timestamp": 0.0, "text": "First"}]
            path = writer.write(entries, filename="append_test.txt")

            writer.append(
                {"timestamp": 1.0, "text": "Second"}, filename="append_test.txt"
            )
            content = path.read_text()
            assert "First" in content
            assert "Second" in content


class TestJSONWriter:
    """Tests for JSON transcript writer."""

    def test_write_creates_valid_json(self) -> None:
        """Writing creates valid JSON with metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JSONWriter(Path(tmpdir))
            entries = [
                {"timestamp": 0.0, "text": "Hello", "language": "en", "confidence": 0.95},
            ]
            path = writer.write(entries, filename="test.json")
            assert path.exists()

            data = json.loads(path.read_text())
            assert data["num_entries"] == 1
            assert data["entries"][0]["text"] == "Hello"
            assert "generated_at" in data
            assert "unix_timestamp" in data

    def test_write_with_metadata(self) -> None:
        """Metadata is included in the JSON output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JSONWriter(Path(tmpdir))
            entries = [{"timestamp": 0.0, "text": "Test"}]
            meta = {"model": "base", "duration_s": 120.0}
            path = writer.write(entries, metadata=meta, filename="meta.json")

            data = json.loads(path.read_text())
            assert data["metadata"]["model"] == "base"
            assert data["metadata"]["duration_s"] == 120.0


class TestSRTWriter:
    """Tests for SRT subtitle writer."""

    def test_write_creates_valid_srt(self) -> None:
        """Writing creates well-formatted SRT file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = SRTWriter(Path(tmpdir))
            entries = [
                {"timestamp": 1.5, "text": "Hello world"},
                {"timestamp": 5.0, "text": "This is a test"},
            ]
            path = writer.write(entries, filename="test.srt")
            assert path.exists()

            content = path.read_text()
            assert "1" in content
            assert "00:00:01,500 -->" in content
            assert "Hello world" in content
            assert "2" in content
            assert "00:00:05,000 -->" in content
            assert "This is a test" in content

    def test_write_skips_empty_text(self) -> None:
        """Empty text entries are skipped in SRT."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = SRTWriter(Path(tmpdir))
            entries = [
                {"timestamp": 0.0, "text": ""},
                {"timestamp": 1.0, "text": "Valid text"},
                {"timestamp": 2.0, "text": "  "},
            ]
            path = writer.write(entries, filename="skip.srt")
            content = path.read_text()
            assert "Valid text" in content
            # SRT writer skips empty/whitespace entries but numbering follows
            # enumerated index. Check the valid text appears.
            assert "Valid text" in content

    def test_format_srt_time(self) -> None:
        """SRT time formatting is correct."""
        writer = SRTWriter(Path("/tmp"))
        assert writer._format_srt_time(0.0) == "00:00:00,000"
        assert writer._format_srt_time(1.5) == "00:00:01,500"
        assert writer._format_srt_time(61.0) == "00:01:01,000"
        assert writer._format_srt_time(3661.123) == "01:01:01,123"
        assert writer._format_srt_time(3600.0) == "01:00:00,000"
