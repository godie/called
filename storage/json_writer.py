"""JSON transcript writer."""

import json
import logging
import time
from pathlib import Path

from utils.timers import get_timestamp

logger = logging.getLogger("realtime-transcriber.json_writer")


class JSONWriter:
    """Writes transcripts to JSON format with full metadata."""

    def __init__(self, output_dir: Path) -> None:
        """Initialize JSON writer.

        Args:
            output_dir: Directory to save transcript files.
        """
        self._output_dir: Path = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        entries: list[dict],
        metadata: dict | None = None,
        filename: str | None = None,
    ) -> Path:
        """Write transcript entries to a JSON file.

        Args:
            entries: List of transcript entry dicts.
            metadata: Optional metadata dict (metrics, config, etc.).
            filename: Optional custom filename.

        Returns:
            Path to the written file.
        """
        if filename is None:
            filename = f"transcript_{get_timestamp()}.json"

        filepath = self._output_dir / filename

        document = {
            "generated_at": get_timestamp(),
            "unix_timestamp": time.time(),
            "num_entries": len(entries),
            "entries": entries,
        }

        if metadata:
            document["metadata"] = metadata

        try:
            filepath.write_text(
                json.dumps(document, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("JSON transcript saved: %s (%d entries)", filepath, len(entries))
        except Exception as exc:
            logger.error("Failed to write JSON transcript: %s", exc)
            raise

        return filepath
