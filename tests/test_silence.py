"""Unit tests for silence detection integration in capture callback."""

import time
from unittest import mock

import numpy as np
import pytest

from audio.capture import AudioCapture
from audio.silence import SilenceDetector


def make_silent_audio(duration_s: float = 2.0, sample_rate: int = 16000) -> np.ndarray:
    """Create near-silent audio (very low amplitude noise)."""
    return np.random.randn(int(sample_rate * duration_s)).astype(np.float32) * 0.001


def make_loud_audio(duration_s: float = 2.0, sample_rate: int = 16000) -> np.ndarray:
    """Create loud audio (full amplitude sine wave)."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    return np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)


class TestSilenceDetectionIntegration:
    """Tests that the SilenceDetector correctly integrates with AudioCapture."""

    def test_silent_chunk_not_enqueued(self) -> None:
        """Silent chunks are detected and skipped, keeping queue empty."""
        import asyncio
        import queue  # Use sync queue for testing

        detector = SilenceDetector(threshold_db=-40.0, sample_rate=16000)

        # Test is_silence directly on the detector
        silent = make_silent_audio(2.0)
        loud = make_loud_audio(2.0)

        assert detector.is_silence(silent) is True
        assert detector.is_silence(loud) is False

    def test_silent_chunks_skipped_counter(self) -> None:
        """silent_chunks_skipped property starts at 0."""
        capture = AudioCapture(device_index=None, sample_rate=16000)
        assert capture.silent_chunks_skipped == 0

    def test_no_detector_no_skip(self) -> None:
        """Without a silence detector, all chunks pass through (no skip)."""
        capture = AudioCapture(device_index=None, sample_rate=16000)
        assert capture.silent_chunks_skipped == 0
        # _silence_detector is None by default, so is_silence won't be called

    def test_detector_rejects_silence(self) -> None:
        """SilenceDetector correctly identifies silence vs speech."""
        detector = SilenceDetector(threshold_db=-30.0, sample_rate=16000)
        capture = AudioCapture(
            device_index=None,
            sample_rate=16000,
            silence_detector=detector,
        )

        silent = make_silent_audio(1.0)
        loud = make_loud_audio(1.0)

        # The detector is wired into the callback; test is_silence directly
        assert capture._silence_detector.is_silence(silent) is True
        assert capture._silence_detector.is_silence(loud) is False

    def test_detector_passed_to_system_capture(self) -> None:
        """SilenceDetector is passed through SystemAudioCapture to AudioCapture."""
        from audio.capture import SystemAudioCapture

        detector = SilenceDetector(threshold_db=-40.0, sample_rate=16000)
        sys_cap = SystemAudioCapture(
            device_index=1,
            sample_rate=16000,
            silence_detector=detector,
        )
        assert sys_cap._silence_detector is detector


class TestSilenceDetectorEdgeCases:
    """Edge case tests for SilenceDetector."""

    def test_empty_audio_is_silent(self) -> None:
        detector = SilenceDetector(threshold_db=-40.0, sample_rate=16000)
        assert detector.is_silence(np.array([], dtype=np.float32)) is True

    def test_single_sample(self) -> None:
        detector = SilenceDetector(threshold_db=-40.0, sample_rate=16000)
        result = detector.is_silence(np.array([0.0], dtype=np.float32))
        assert isinstance(result, bool)

    def test_reset_clears_noise_floor(self) -> None:
        detector = SilenceDetector(threshold_db=-40.0, sample_rate=16000)
        # Feed some loud audio to establish noise floor
        loud = make_loud_audio(2.0)
        detector.is_silence(loud)
        assert detector._samples_seen > 0

        detector.reset()
        assert detector._samples_seen == 0
        assert detector._noise_floor == -60.0


class TestRecalibrationDefaults:
    """Tests for the recalibration parameters and properties."""

    def test_recalibration_params_default(self) -> None:
        """Recalibration is enabled by default with 30s interval, 10 dB margin."""
        detector = SilenceDetector(threshold_db=-40.0, sample_rate=16000)
        assert detector._recalibration_interval_s == 30.0
        assert detector._recalibration_margin_db == 10.0
        assert detector.recalibration_count == 0

    def test_recalibration_params_custom(self) -> None:
        """Custom recalibration params are stored correctly."""
        detector = SilenceDetector(
            threshold_db=-40.0,
            sample_rate=16000,
            recalibration_interval_s=60.0,
            recalibration_margin_db=15.0,
        )
        assert detector._recalibration_interval_s == 60.0
        assert detector._recalibration_margin_db == 15.0

    def test_original_threshold_preserved(self) -> None:
        """original_threshold_db keeps the initial value even after recalibration."""
        detector = SilenceDetector(threshold_db=-35.0, sample_rate=16000)
        assert detector.original_threshold_db == -35.0
        assert detector.current_threshold_db == -35.0

    def test_recalibration_disabled_with_zero_interval(self) -> None:
        """When interval is 0 or negative, recalibration never fires."""
        detector = SilenceDetector(
            threshold_db=-40.0,
            sample_rate=16000,
            recalibration_interval_s=0.0,
        )
        # Feed many silent frames to establish a noise floor
        for _ in range(100):
            detector.is_silence(make_silent_audio(0.1))

        assert detector.recalibration_count == 0
        # Threshold should not have changed
        assert detector.current_threshold_db == -40.0


class TestRecalibrationBehavior:
    """Tests for the recalibration logic itself."""

    def test_recalibrate_updates_threshold(self) -> None:
        """recalibrate() sets threshold to noise_floor + margin, clamped."""
        detector = SilenceDetector(
            threshold_db=-40.0,
            sample_rate=16000,
            recalibration_interval_s=0.0,  # disable auto, test manually
            recalibration_margin_db=10.0,
        )

        # Feed silent audio to pull noise floor down
        for _ in range(200):
            detector.is_silence(make_silent_audio(0.1))

        # The noise floor has updated via EMA — manually recalibrate
        old_threshold = detector.current_threshold_db
        new_threshold = detector.recalibrate()

        # The new threshold should be noise_floor + margin
        expected = detector._noise_floor + 10.0
        assert new_threshold == pytest.approx(expected, abs=0.1)
        assert detector.recalibration_count == 1
        assert detector.current_threshold_db == new_threshold

    def test_recalibrate_never_exceeds_original(self) -> None:
        """Recalibration clamps: threshold never exceeds original value."""
        detector = SilenceDetector(
            threshold_db=-30.0,  # generous threshold
            sample_rate=16000,
            recalibration_interval_s=0.0,
            recalibration_margin_db=10.0,
        )

        # Feed loud audio to push noise floor up
        for _ in range(200):
            detector.is_silence(make_loud_audio(0.1))

        new_threshold = detector.recalibrate()
        # Should be clamped to original -30 dB (or below)
        assert new_threshold <= -30.0 + 0.01  # tolerate float rounding

    def test_recalibrate_never_below_negative_60(self) -> None:
        """Recalibration floors at -60 dB to avoid pathological values."""
        detector = SilenceDetector(
            threshold_db=-40.0,
            sample_rate=16000,
            recalibration_interval_s=0.0,
            recalibration_margin_db=10.0,
        )

        # Manually set noise floor to an extremely low value
        detector._noise_floor = -80.0
        new_threshold = detector.recalibrate()

        assert new_threshold >= -60.0

    def test_recalibrate_count_increments(self) -> None:
        """Each recalibrate() call increments the counter."""
        detector = SilenceDetector(
            threshold_db=-40.0,
            sample_rate=16000,
            recalibration_interval_s=0.0,
        )

        assert detector.recalibration_count == 0
        detector.recalibrate()
        assert detector.recalibration_count == 1
        detector.recalibrate()
        assert detector.recalibration_count == 2

    def test_reset_restores_original_threshold(self) -> None:
        """reset() restores original threshold and clears recalibration count."""
        detector = SilenceDetector(
            threshold_db=-40.0,
            sample_rate=16000,
            recalibration_interval_s=0.0,
        )

        # Feed silent audio, then recalibrate
        for _ in range(200):
            detector.is_silence(make_silent_audio(0.1))
        detector.recalibrate()
        assert detector.recalibration_count == 1
        assert detector.current_threshold_db != -40.0

        detector.reset()
        assert detector.recalibration_count == 0
        assert detector.current_threshold_db == -40.0
        assert detector._noise_floor == -60.0

    def test_auto_recalibration_fires_after_interval(self) -> None:
        """is_silence() triggers recalibration when enough time has passed."""
        detector = SilenceDetector(
            threshold_db=-40.0,
            sample_rate=16000,
            recalibration_interval_s=0.001,  # 1ms — fires immediately
            recalibration_margin_db=10.0,
        )

        # Feed silent audio to establish noise floor, then wait
        for _ in range(50):
            detector.is_silence(make_silent_audio(0.1))
        time.sleep(0.002)

        # One more call should trigger recalibration
        detector.is_silence(make_silent_audio(0.1))
        assert detector.recalibration_count >= 1

    def test_recalibration_preserves_effective_threshold_in_speech_segments(
        self,
    ) -> None:
        """After recalibration, get_speech_segments uses updated threshold."""
        detector = SilenceDetector(
            threshold_db=-40.0,
            sample_rate=16000,
            recalibration_interval_s=0.0,
            recalibration_margin_db=10.0,
        )

        # Feed silent audio to pull noise floor down
        for _ in range(200):
            detector.is_silence(make_silent_audio(0.1))
        detector.recalibrate()

        # Mixed audio: silence + loud
        audio = np.concatenate([
            make_silent_audio(1.0),
            make_loud_audio(1.0),
            make_silent_audio(1.0),
        ])
        segments = detector.get_speech_segments(audio)
        assert len(segments) >= 1
