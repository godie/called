"""Checkpoint recovery — load transcripts from a previous crashed session.

Finds the most recent checkpoint file from the checkpoint directory and
restores the transcript buffer so transcription can resume without
losing data from the previous session.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("realtime-transcriber.checkpoint")


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Find the most recent checkpoint file by modification time.

    Args:
        checkpoint_dir: Directory containing checkpoint JSON files.

    Returns:
        Path to the latest checkpoint, or None if none found.
    """
    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        return None

    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return checkpoints[0] if checkpoints else None


def load_checkpoint_data(checkpoint_path: Path) -> Optional[dict]:
    """Load checkpoint JSON data from a file.

    Args:
        checkpoint_path: Path to the checkpoint JSON file.

    Returns:
        Parsed checkpoint dictionary, or None on failure.
    """
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate required fields
        required = ("entries", "saved_at")
        for key in required:
            if key not in data:
                logger.warning(
                    "Checkpoint %s missing required field '%s', skipping",
                    checkpoint_path,
                    key,
                )
                return None

        entries = data.get("entries", [])
        if not isinstance(entries, list):
            logger.warning(
                "Checkpoint %s has invalid entries (not a list), skipping",
                checkpoint_path,
            )
            return None

        logger.info(
            "Loaded checkpoint from %s (%d entries, saved %s)",
            checkpoint_path,
            len(entries),
            data.get("saved_at", "unknown"),
        )
        return data

    except json.JSONDecodeError as exc:
        logger.warning("Corrupt checkpoint %s: %s", checkpoint_path, exc)
        return None
    except Exception as exc:
        logger.error("Failed to load checkpoint %s: %s", checkpoint_path, exc)
        return None
