"""SRT subtitle writer."""

import logging
from pathlib import Path

from utils.timers import get_timestamp

logger = logging.getLogger("realtime-transcriber.srt_writer")


class SRTWriter:
    """Writes transcripts to SRT (SubRip) subtitle format."""

    def __init__(self, output_dir: Path) -> None:
        """Initialize SRT writer.

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
        """Write transcript entries to an SRT file.

        Args:
            entries: List of transcript entry dicts with 'timestamp' and 'text'.
            filename: Optional custom filename.

        Returns:
            Path to the written file.
        """
        if filename is None:
            filename = f"transcript_{get_timestamp()}.srt"

        filepath = self._output_dir / filename

        lines: list[str] = []
        for i, entry in enumerate(entries, start=1):
            start_ts = entry.get("timestamp", 0.0)
            # Estimate end time: 3 seconds per entry or next entry's timestamp
            end_ts = start_ts + 3.0
            text = entry.get("text", "").strip()

            if not text:
                continue

            # SRT format:
            # 1
            # 00:00:01,000 --> 00:00:04,000
            # Text here
            lines.append(str(i))
            lines.append(self._format_srt_time(start_ts) + " --> " + self._format_srt_time(end_ts))
            lines.append(text)
            lines.append("")

        try:
            filepath.write_text("\n".join(lines), encoding="utf-8")
            logger.info("SRT transcript saved: %s (%d entries)", filepath, len(entries))
        except Exception as exc:
            logger.error("Failed to write SRT transcript: %s", exc)
            raise

        return filepath

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """Format seconds to SRT timestamp (HH:MM:SS,mmm).

        Args:
            seconds: Time in seconds.

        Returns:
            SRT-formatted time string.
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
