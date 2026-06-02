"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class SilenceDetectionConfig:
    """Silence detection configuration."""

    enabled: bool = True
    threshold_db: float = -40.0
    recalibration_enabled: bool = True
    recalibration_interval_s: float = 30.0
    recalibration_margin_db: float = 10.0


@dataclass(frozen=True)
class AudioConfig:
    """Audio capture configuration."""

    mic_device_index: int | None = None
    system_audio_device_index: int | None = None
    sample_rate: int = 16000
    chunk_duration: float = 2.0
    channels: int = 1


@dataclass(frozen=True)
class RingBufferConfig:
    """Ring buffer configuration."""

    capacity: int = 128


@dataclass(frozen=True)
class WhisperConfig:
    """Transcription model configuration."""

    model_size: str = "base"
    device: str = "auto"
    compute_type: str = "default"
    beam_size: int = 5
    language: str | None = None
    vad_threshold: float = 0.5
    min_speech_duration: float = 0.5
    silence_duration: float = 1.0


@dataclass(frozen=True)
class QueueConfig:
    """Queue configuration for producer/consumer."""

    max_size: int = 64
    num_workers: int = 1


@dataclass(frozen=True)
class AudioArchiveConfig:
    """Audio archiving configuration."""

    codec: str = "opus"
    bitrate: str = "16k"
    save_raw_audio: bool = False
    save_compressed_audio: bool = True
    archive_dir: Path = Path("./audio_archive")


@dataclass(frozen=True)
class OutputConfig:
    """Output configuration for transcripts."""

    output_dir: Path = Path("./transcripts")
    rotation_interval: float = 3600.0


@dataclass(frozen=True)
class MetricsConfig:
    """Metrics collection configuration."""

    enabled: bool = True
    interval: float = 30.0


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpoint configuration."""

    interval: float = 60.0
    recovery_enabled: bool = True
    checkpoint_dir: Path = Path("./checkpoints")


@dataclass(frozen=True)
class LogConfig:
    """Logging configuration."""

    level: str = "INFO"
    log_file: str | None = None


@dataclass(frozen=True)
class AppConfig:
    """Master application configuration."""

    silence_detection: SilenceDetectionConfig = field(default_factory=SilenceDetectionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    ring_buffer: RingBufferConfig = field(default_factory=RingBufferConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    audio_archive: AudioArchiveConfig = field(default_factory=AudioArchiveConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    log: LogConfig = field(default_factory=LogConfig)


def _parse_optional_int(value: str) -> int | None:
    """Parse an optional integer from a string."""
    if not value:
        return None
    return int(value)


def load_config() -> AppConfig:
    """Load configuration from environment variables."""

    silence_detection = SilenceDetectionConfig(
        enabled=os.getenv("SILENCE_DETECTION_ENABLED", "true").lower() == "true",
        threshold_db=float(os.getenv("SILENCE_THRESHOLD_DB", "-40.0")),
        recalibration_enabled=os.getenv("SILENCE_RECALIBRATION_ENABLED", "true").lower() == "true",
        recalibration_interval_s=float(os.getenv("SILENCE_RECALIBRATION_INTERVAL", "30.0")),
        recalibration_margin_db=float(os.getenv("SILENCE_RECALIBRATION_MARGIN_DB", "10.0")),
    )

    audio = AudioConfig(
        mic_device_index=_parse_optional_int(os.getenv("MIC_DEVICE_INDEX", "")),
        system_audio_device_index=_parse_optional_int(os.getenv("SYSTEM_AUDIO_DEVICE_INDEX", "")),
        sample_rate=int(os.getenv("SAMPLE_RATE", "16000")),
        chunk_duration=float(os.getenv("CHUNK_DURATION", "2.0")),
        channels=int(os.getenv("CHANNELS", "1")),
    )

    ring_buffer = RingBufferConfig(
        capacity=int(os.getenv("RING_BUFFER_CAPACITY", "128")),
    )

    whisper = WhisperConfig(
        model_size=os.getenv("WHISPER_MODEL_SIZE", "base"),
        device=os.getenv("WHISPER_DEVICE", "auto"),
        compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "default"),
        beam_size=int(os.getenv("WHISPER_BEAM_SIZE", "5")),
        language=os.getenv("WHISPER_LANGUAGE") or None,
        vad_threshold=float(os.getenv("VAD_THRESHOLD", "0.5")),
        min_speech_duration=float(os.getenv("MIN_SPEECH_DURATION", "0.5")),
        silence_duration=float(os.getenv("SILENCE_DURATION", "1.0")),
    )

    queue = QueueConfig(
        max_size=int(os.getenv("QUEUE_MAX_SIZE", "64")),
        num_workers=int(os.getenv("NUM_WORKERS", "1")),
    )

    audio_archive = AudioArchiveConfig(
        codec=os.getenv("AUDIO_ARCHIVE_CODEC", "opus"),
        bitrate=os.getenv("AUDIO_ARCHIVE_BITRATE", "16k"),
        save_raw_audio=os.getenv("SAVE_RAW_AUDIO", "false").lower() == "true",
        save_compressed_audio=os.getenv("SAVE_COMPRESSED_AUDIO", "true").lower() == "true",
        archive_dir=Path(os.getenv("AUDIO_ARCHIVE_DIR", "./audio_archive")),
    )

    output = OutputConfig(
        output_dir=Path(os.getenv("OUTPUT_DIR", "./transcripts")),
        rotation_interval=float(os.getenv("ROTATION_INTERVAL", "3600")),
    )

    metrics = MetricsConfig(
        enabled=os.getenv("METRICS_ENABLED", "true").lower() == "true",
        interval=float(os.getenv("METRICS_INTERVAL", "30")),
    )

    checkpoint = CheckpointConfig(
        interval=float(os.getenv("CHECKPOINT_INTERVAL", "60")),
        recovery_enabled=os.getenv("CHECKPOINT_RECOVERY_ENABLED", "true").lower() == "true",
        checkpoint_dir=Path(os.getenv("CHECKPOINT_DIR", "./checkpoints")),
    )

    log = LogConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        log_file=os.getenv("LOG_FILE") or None,
    )

    return AppConfig(
        silence_detection=silence_detection,
        audio=audio,
        ring_buffer=ring_buffer,
        whisper=whisper,
        queue=queue,
        audio_archive=audio_archive,
        output=output,
        metrics=metrics,
        checkpoint=checkpoint,
        log=log,
    )
