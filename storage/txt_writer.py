"""TXT transcript writer."""

import logging
from pathlib import Path

from utils.timers import format_duration, get_timestamp

logger = logging.getLogger("realtime-transcriber.txt_writer")


class TXTWriter:
    """Writes transcripts to plain text format."""

    def __init__(self, output_dir: Path) -> None:
        """Initialize TXT writer.

        Args:
            output_dir: Directory to save transcript files.
        """
        self._output_dir: Path = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        entries: list[dict],
        filename: str | None = None,
    ) -> Path:
        """Write transcript entries to a TXT file.

        Args:
            entries: List of transcript entry dicts with 'timestamp', 'text'.
            filename: Optional custom filename. Defaults to timestamp.

        Returns:
            Path to the written file.
        """
        if filename is None:
            filename = f"transcript_{get_timestamp()}.txt"

        filepath = self._output_dir / filename

        lines: list[str] = []
        lines.append("=" * 60)
        lines.append(f"Transcription Session: {get_timestamp()}")
        lines.append("=" * 60)
        lines.append("")

        for entry in entries:
            ts = entry.get("timestamp", 0.0)
            text = entry.get("text", "").strip()
            lines.append(f"[{format_duration(ts)}] {text}")
            lines.append("")

        try:
            filepath.write_text("\n".join(lines), encoding="utf-8")
            logger.info("TXT transcript saved: %s (%d entries)", filepath, len(entries))
        except Exception as exc:
            logger.error("Failed to write TXT transcript: %s", exc)
            raise

        return filepath

    def append(self, entry: dict, filename: str) -> None:
        """Append a single entry to an existing transcript file.

        Args:
            entry: Transcript entry dict.
            filename: Filename to append to.
        """
        filepath = self._output_dir / filename
        ts = entry.get("timestamp", 0.0)
        text = entry.get("text", "").strip()

        with open(filepath, "a", encoding="utf-8") as f:
            f.write(f"[{format_duration(ts)}] {text}\n")
