"""Unit tests for Config loading."""

import os
from pathlib import Path
from unittest import mock

import pytest

from config import AppConfig, load_config


class TestConfig:
    """Tests for configuration loading."""

    def test_load_config_defaults(self) -> None:
        """Default values are used when env vars are not set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            config = load_config()

            assert config.audio.sample_rate == 16000
            assert config.audio.chunk_duration == 2.0
            assert config.audio.channels == 1
            assert config.audio.mic_device_index is None
            assert config.audio.system_audio_device_index is None

            assert config.ring_buffer.capacity == 128

            assert config.whisper.model_size == "base"
            assert config.whisper.device == "auto"
            assert config.whisper.compute_type == "default"
            assert config.whisper.beam_size == 5
            assert config.whisper.language is None
            assert config.whisper.vad_threshold == 0.5
            assert config.whisper.min_speech_duration == 0.5
            assert config.whisper.silence_duration == 1.0

            assert config.queue.max_size == 64
            assert config.queue.num_workers == 1

            assert isinstance(config.output.output_dir, Path)
            assert config.output.rotation_interval == 3600.0

            assert config.metrics.enabled is True
            assert config.metrics.interval == 30.0

            assert config.checkpoint.interval == 60.0
            assert isinstance(config.checkpoint.checkpoint_dir, Path)

            assert config.log.level == "INFO"
            assert config.log.log_file is None

    def test_load_config_from_env(self) -> None:
        """Environment variables override defaults."""
        env = {
            "SAMPLE_RATE": "44100",
            "CHUNK_DURATION": "1.5",
            "CHANNELS": "2",
            "MIC_DEVICE_INDEX": "3",
            "RING_BUFFER_CAPACITY": "256",
            "WHISPER_MODEL_SIZE": "large-v3",
            "WHISPER_DEVICE": "cuda",
            "WHISPER_COMPUTE_TYPE": "float16",
            "WHISPER_BEAM_SIZE": "8",
            "WHISPER_LANGUAGE": "en",
            "QUEUE_MAX_SIZE": "32",
            "NUM_WORKERS": "4",
            "OUTPUT_DIR": "/tmp/transcripts",
            "ROTATION_INTERVAL": "1800",
            "METRICS_ENABLED": "false",
            "METRICS_INTERVAL": "60",
            "CHECKPOINT_INTERVAL": "120",
            "CHECKPOINT_DIR": "/tmp/checkpoints",
            "LOG_LEVEL": "DEBUG",
            "LOG_FILE": "/tmp/app.log",
        }

        with mock.patch.dict(os.environ, env, clear=True):
            config = load_config()

            assert config.audio.sample_rate == 44100
            assert config.audio.chunk_duration == 1.5
            assert config.audio.channels == 2
            assert config.audio.mic_device_index == 3

            assert config.ring_buffer.capacity == 256

            assert config.whisper.model_size == "large-v3"
            assert config.whisper.device == "cuda"
            assert config.whisper.compute_type == "float16"
            assert config.whisper.beam_size == 8
            assert config.whisper.language == "en"

            assert config.queue.max_size == 32
            assert config.queue.num_workers == 4

            assert config.output.output_dir == Path("/tmp/transcripts")
            assert config.output.rotation_interval == 1800.0

            assert config.metrics.enabled is False
            assert config.metrics.interval == 60.0

            assert config.checkpoint.interval == 120.0
            assert config.checkpoint.checkpoint_dir == Path("/tmp/checkpoints")

            assert config.log.level == "DEBUG"
            assert config.log.log_file == "/tmp/app.log"

    def test_config_is_frozen(self) -> None:
        """Configuration dataclasses are frozen (immutable)."""
        config = AppConfig()
        with pytest.raises(Exception):
            config.audio.sample_rate = 48000  # type: ignore[misc]

    def test_empty_language_means_none(self) -> None:
        """Empty WHISPER_LANGUAGE env var results in None."""
        with mock.patch.dict(os.environ, {"WHISPER_LANGUAGE": ""}, clear=True):
            config = load_config()
            assert config.whisper.language is None

    def test_empty_device_indices(self) -> None:
        """Empty device index env vars result in None."""
        env = {"MIC_DEVICE_INDEX": "", "SYSTEM_AUDIO_DEVICE_INDEX": ""}
        with mock.patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.audio.mic_device_index is None
            assert config.audio.system_audio_device_index is None

    def test_silence_detection_config_defaults(self) -> None:
        """Silence detection defaults: enabled, threshold -40 dB, recalibration on."""
        with mock.patch.dict(os.environ, {}, clear=True):
            config = load_config()
            assert config.silence_detection.enabled is True
            assert config.silence_detection.threshold_db == -40.0
            assert config.silence_detection.recalibration_enabled is True
            assert config.silence_detection.recalibration_interval_s == 30.0
            assert config.silence_detection.recalibration_margin_db == 10.0

    def test_silence_detection_config_from_env(self) -> None:
        """Silence detection env vars override defaults."""
        env = {
            "SILENCE_DETECTION_ENABLED": "false",
            "SILENCE_THRESHOLD_DB": "-30.0",
            "SILENCE_RECALIBRATION_ENABLED": "false",
            "SILENCE_RECALIBRATION_INTERVAL": "60.0",
            "SILENCE_RECALIBRATION_MARGIN_DB": "15.0",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = load_config()
            assert config.silence_detection.enabled is False
            assert config.silence_detection.threshold_db == -30.0
            assert config.silence_detection.recalibration_enabled is False
            assert config.silence_detection.recalibration_interval_s == 60.0
            assert config.silence_detection.recalibration_margin_db == 15.0
