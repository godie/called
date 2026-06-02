"""Speaker detection placeholder interface.

Provides an abstract base for speaker diarization implementations.
Concrete implementations may use libraries like pyannote.audio,
speechbrain, or custom models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class SpeakerSegment:
    """A segment of speech attributed to a specific speaker."""

    speaker_id: str
    start: float  # Start time in seconds
    end: float  # End time in seconds
    confidence: float = 1.0


class SpeakerDetector(ABC):
    """Abstract interface for speaker detection/diarization.

    Implementations should identify who is speaking and when,
    enabling multi-speaker transcript labeling.
    """

    @abstractmethod
    def detect_speakers(self, audio: np.ndarray, sample_rate: int) -> list[SpeakerSegment]:
        """Detect speaker segments in audio data.

        Args:
            audio: Audio samples as numpy array.
            sample_rate: Sample rate in Hz.

        Returns:
            List of speaker segments with timing and speaker IDs.
        """
        ...

    @abstractmethod
    def label_speaker(self, speaker_id: str, label: str) -> None:
        """Assign a human-readable label to a speaker ID.

        Args:
            speaker_id: The detected speaker's unique ID.
            label: Human-readable label (e.g., "Alice", "Interviewer").
        """
        ...


class NoOpSpeakerDetector(SpeakerDetector):
    """No-op implementation that returns no speaker segments.

    Used when speaker detection is not configured or unavailable.
    """

    def detect_speakers(self, audio: np.ndarray, sample_rate: int) -> list[SpeakerSegment]:
        return []

    def label_speaker(self, speaker_id: str, label: str) -> None:
        pass
