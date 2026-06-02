"""Unit tests for audio resampler."""

import numpy as np

from audio.resampler import (
    TARGET_SAMPLE_RATE,
    _resample_linear,
    audio_chunk_duration,
    resample_to_whisper_format,
)


class TestResampleLinear:
    """Tests for linear resampling."""

    def test_passthrough_same_rate(self) -> None:
        """No change when source and dest rates are equal."""
        audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        result = _resample_linear(audio, 16000, 16000)
        assert len(result) == len(audio)
        np.testing.assert_array_almost_equal(result, audio)

    def test_upsample_doubles_length(self) -> None:
        """Upsampling 2x approximately doubles the sample count."""
        audio = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        result = _resample_linear(audio, 8000, 16000)
        expected_len = int(len(audio) * 16000 / 8000)
        assert len(result) == expected_len

    def test_downsample_halves_length(self) -> None:
        """Downsampling 2x reduces sample count."""
        audio = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)
        result = _resample_linear(audio, 32000, 16000)
        assert len(result) == 3

    def test_preserves_amplitude_range(self) -> None:
        """Resampled signal stays within original amplitude range."""
        audio = np.sin(np.linspace(0, 10 * np.pi, 1000)).astype(np.float32)
        result = _resample_linear(audio, 16000, 8000)
        assert np.max(np.abs(result)) <= 1.01  # Allow small interpolation overshoot

    def test_empty_input(self) -> None:
        """Empty input produces empty output."""
        audio = np.array([], dtype=np.float32)
        result = _resample_linear(audio, 16000, 8000)
        assert len(result) == 0

    def test_minimal_input(self) -> None:
        """Minimal input (2 samples) produces at least 1 output sample."""
        audio = np.array([0.5, 0.6], dtype=np.float32)
        result = _resample_linear(audio, 16000, 8000)
        assert len(result) >= 1


class TestResampleToWhisperFormat:
    """Tests for the full whisper-format resampling pipeline."""

    def test_mono_16k_passthrough(self) -> None:
        """Already 16kHz mono audio passes through unchanged."""
        audio = np.sin(np.linspace(0, 2 * np.pi, 16000)).astype(np.float32)
        result = resample_to_whisper_format(audio, 16000, 1)
        assert result.shape == audio.shape
        assert result.dtype == np.float32

    def test_stereo_to_mono(self) -> None:
        """Stereo audio is converted to mono by averaging channels."""
        left = np.ones(16000, dtype=np.float32) * 0.5
        right = np.ones(16000, dtype=np.float32) * (-0.5)
        stereo = np.column_stack((left, right))
        result = resample_to_whisper_format(stereo, 16000, 2)
        assert result.ndim == 1
        # Average of 0.5 and -0.5 = 0.0
        assert np.allclose(result, 0.0, atol=1e-6)

    def test_44k_to_16k(self) -> None:
        """44100 Hz audio is downsampled to 16000 Hz."""
        audio = np.sin(np.linspace(0, 2 * np.pi, 44100)).astype(np.float32)
        result = resample_to_whisper_format(audio, 44100, 1)
        expected_len = int(44100 * 16000 / 44100)
        assert len(result) == expected_len
        assert result.dtype == np.float32

    def test_output_is_1d_float32(self) -> None:
        """Output is always 1D float32."""
        audio = np.zeros((8000, 1), dtype=np.float32)
        result = resample_to_whisper_format(audio, 8000, 1)
        assert result.ndim == 1
        assert result.dtype == np.float32

    def test_2d_mono_flattened(self) -> None:
        """2D mono (samples, 1) is flattened to 1D."""
        audio = np.ones((100, 1), dtype=np.float32) * 0.5
        result = resample_to_whisper_format(audio, 16000, 1)
        assert result.ndim == 1

    def test_matches_whisper_target_constants(self) -> None:
        """Output matches the TARGET_SAMPLE_RATE constant."""
        audio = np.zeros(8000, dtype=np.float32)
        result = resample_to_whisper_format(audio, 8000, 1)
        assert result.dtype == np.float32
        # Duration preserved
        assert abs(len(result) / TARGET_SAMPLE_RATE - len(audio) / 8000) < 0.001


class TestAudioChunkDuration:
    """Tests for audio_chunk_duration."""

    def test_1hz_10samples(self) -> None:
        assert audio_chunk_duration(np.zeros(10), 1) == 10.0

    def test_16000hz(self) -> None:
        assert audio_chunk_duration(np.zeros(16000), 16000) == 1.0

    def test_zero_rate(self) -> None:
        assert audio_chunk_duration(np.zeros(100), 0) == 0.0

    def test_empty(self) -> None:
        assert audio_chunk_duration(np.array([]), 16000) == 0.0
