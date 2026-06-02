"""Silence detection for audio chunks.

Identifies periods of silence in audio to:
- Skip processing silent chunks (save CPU)
- Split transcripts at natural pause boundaries
- Improve deduplication accuracy

Uses energy-based detection (RMS) which is fast and requires no ML models.

Supports periodic recalibration: the threshold adapts to the tracked noise
floor over time, handling changing background noise levels automatically.
"""

import logging
import time

import numpy as np

logger = logging.getLogger("realtime-transcriber.silence")


class SilenceDetector:
    """Energy-based silence detection for audio chunks.

    Calculates RMS energy and compares against an adaptive threshold
    to determine if an audio chunk contains meaningful speech.
    """

    def __init__(
        self,
        threshold_db: float = -40.0,
        min_speech_duration: float = 0.5,
        silence_duration: float = 1.0,
        sample_rate: int = 16000,
        recalibration_interval_s: float = 30.0,
        recalibration_margin_db: float = 10.0,
    ) -> None:
        """Initialize silence detector.

        Args:
            threshold_db: RMS threshold in dB below which is silence.
            min_speech_duration: Minimum speech duration in seconds
                before a chunk is considered speech.
            silence_duration: Silence duration in seconds before
                a segment is split.
            sample_rate: Audio sample rate in Hz.
            recalibration_interval_s: Seconds between auto-recalibrations.
                Set to 0 or negative to disable auto-recalibration.
            recalibration_margin_db: dB margin above noise floor to set
                new threshold during recalibration.
        """
        self._threshold_db: float = threshold_db
        self._original_threshold_db: float = threshold_db
        self._min_speech_duration: float = min_speech_duration
        self._silence_duration: float = silence_duration
        self._sample_rate: int = sample_rate

        # Recalibration configuration
        self._recalibration_interval_s: float = recalibration_interval_s
        self._recalibration_margin_db: float = recalibration_margin_db
        self._last_recalibration_time: float = time.time()
        self._recalibration_count: int = 0

        # Adaptive threshold tracking
        self._noise_floor: float = -60.0  # dB, estimated background noise
        self._samples_seen: int = 0

    def is_silence(self, audio: np.ndarray) -> bool:
        """Check if an audio chunk is silent.

        Args:
            audio: Audio samples as numpy array (float32).

        Returns:
            True if the chunk is below the silence threshold.
        """
        rms_db = self._compute_rms_db(audio)
        self._update_noise_floor(rms_db)

        # Periodic recalibration to adapt to changing background noise
        self._maybe_recalibrate()

        # Use the higher of threshold_db or noise_floor + margin
        effective_threshold = max(
            self._threshold_db, self._noise_floor + self._recalibration_margin_db
        )
        return rms_db < effective_threshold

    def get_speech_segments(self, audio: np.ndarray) -> list[tuple[int, int]]:
        """Find speech segments within an audio chunk.

        Args:
            audio: Audio samples as numpy array (float32).

        Returns:
            List of (start_sample, end_sample) tuples for speech regions.
        """
        if len(audio) == 0:
            return []

        # Compute energy per frame (20ms frames)
        frame_size = int(self._sample_rate * 0.02)  # 20ms
        if frame_size == 0:
            return [(0, len(audio))]

        num_frames = max(1, len(audio) // frame_size)

        segments: list[tuple[int, int]] = []
        in_speech = False
        speech_start = 0
        silence_frames = 0
        min_speech_frames = int(self._min_speech_duration / 0.02)
        max_silence_frames = int(self._silence_duration / 0.02)

        for i in range(num_frames):
            start = i * frame_size
            end = min(start + frame_size, len(audio))
            frame = audio[start:end]
            rms_db = self._compute_rms_db(frame)

            threshold = max(self._threshold_db, self._noise_floor + self._recalibration_margin_db)

            if rms_db >= threshold:
                if not in_speech:
                    speech_start = start
                    in_speech = True
                silence_frames = 0
            else:
                if in_speech:
                    silence_frames += 1
                    if silence_frames >= max_silence_frames:
                        # End of speech segment
                        speech_frames = (i - silence_frames) - int(speech_start / frame_size)
                        if speech_frames >= min_speech_frames:
                            segments.append((speech_start, start - silence_frames * frame_size))
                        in_speech = False
                        silence_frames = 0

        # Handle trailing speech
        if in_speech:
            speech_frames = num_frames - int(speech_start / frame_size)
            if speech_frames >= min_speech_frames:
                segments.append((speech_start, len(audio)))

        return segments

    def _compute_rms_db(self, audio: np.ndarray) -> float:
        """Compute RMS energy in dB.

        Args:
            audio: Audio samples.

        Returns:
            RMS level in dB, or -100.0 for zero-energy audio.
        """
        if len(audio) == 0:
            return -100.0

        rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
        if rms < 1e-10:
            return -100.0

        return float(20.0 * np.log10(rms))

    def _update_noise_floor(self, rms_db: float) -> None:
        """Adaptively track the noise floor.

        Args:
            rms_db: Current RMS energy in dB.
        """
        self._samples_seen += 1

        # Slow exponential moving average toward quiet frames
        if rms_db < self._threshold_db:
            alpha = 0.001
            self._noise_floor = (1.0 - alpha) * self._noise_floor + alpha * rms_db

    def _maybe_recalibrate(self) -> None:
        """Check if recalibration is due and trigger if so.

        Called from is_silence() after updating the noise floor.
        Inexpensive — just a timestamp comparison unless recalibration fires.
        Requires a minimum number of samples before first recalibration
        so the noise floor EMA has converged from its default of -60 dB.
        """
        if self._recalibration_interval_s <= 0:
            return
        # Minimum samples before first recalibration — the default
        # noise floor of -60 dB is a placeholder, not a real measurement.
        # At 2-second chunks this is ~100 seconds of audio.
        if self._samples_seen < 50:
            return
        now = time.time()
        if now - self._last_recalibration_time >= self._recalibration_interval_s:
            self.recalibrate()

    def recalibrate(self) -> float:
        """Update threshold based on tracked noise floor.

        Sets the active threshold to noise_floor + margin, clamped so it
        never exceeds the original user-configured threshold (don't get
        overly aggressive) and never falls below -60 dB (pathological case).

        Returns:
            The new threshold value in dB.
        """
        new_threshold = self._noise_floor + self._recalibration_margin_db

        # Clamp: never go above original (could make false positives)
        # and floor at -60 dB to stay in sane range
        new_threshold = max(-60.0, min(new_threshold, self._original_threshold_db))

        if abs(new_threshold - self._threshold_db) > 0.5:
            logger.debug(
                "Recalibrating silence threshold: %.1f dB → %.1f dB "
                "(noise floor: %.1f dB, margin: %.0f dB, recalibration #%d)",
                self._threshold_db,
                new_threshold,
                self._noise_floor,
                self._recalibration_margin_db,
                self._recalibration_count + 1,
            )

        self._threshold_db = new_threshold
        self._last_recalibration_time = time.time()
        self._recalibration_count += 1
        return new_threshold

    @property
    def recalibration_count(self) -> int:
        """Number of recalibrations performed so far."""
        return self._recalibration_count

    @property
    def current_threshold_db(self) -> float:
        """Current active silence threshold (may differ from original due to recalibration)."""
        return self._threshold_db

    @property
    def original_threshold_db(self) -> float:
        """User-configured threshold before any recalibration."""
        return self._original_threshold_db

    def reset(self) -> None:
        """Reset the adaptive noise floor and recalibration state."""
        self._noise_floor = -60.0
        self._samples_seen = 0
        self._threshold_db = self._original_threshold_db
        self._last_recalibration_time = time.time()
        self._recalibration_count = 0
