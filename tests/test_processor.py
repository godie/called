"""Unit tests for the producer/consumer processor."""

import asyncio
from unittest import mock

import numpy as np
import pytest

import json
import tempfile
from pathlib import Path

from audio.buffers import AudioChunk
from config import AppConfig, load_config
from transcription.processor import ProcessingMetrics, TranscriptionProcessor
from transcription.whisper_service import TranscriptionResult, WhisperService


@pytest.fixture
def config() -> AppConfig:
    """Create a test configuration."""
    return load_config()


@pytest.fixture
def mock_whisper() -> WhisperService:
    """Create a mock WhisperService."""
    svc = mock.MagicMock(spec=WhisperService)
    svc.is_loaded = True
    svc.transcribe.return_value = TranscriptionResult(
        text="Hello world",
        segments=[],
        language="en",
        language_probability=0.98,
        processing_time=0.1,
        input_duration=2.0,
    )
    return svc


def make_chunk(text: str = "", seq: int = 0) -> AudioChunk:
    """Create a test audio chunk."""
    return AudioChunk(
        data=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        timestamp=float(seq),
        source="test",
        sequence_id=seq,
    )


class TestProcessingMetrics:
    """Tests for ProcessingMetrics."""

    def test_record_chunk_updates_all_fields(self) -> None:
        m = ProcessingMetrics()
        m.record_chunk(processing_time=0.5, audio_duration=2.0, queue_size=3)
        assert m.chunks_processed == 1
        assert m.min_latency == 0.5
        assert m.max_latency == 0.5
        assert m.total_audio_duration == 2.0

    def test_record_dropped(self) -> None:
        m = ProcessingMetrics()
        m.record_dropped()
        assert m.chunks_dropped == 1

    def test_samples_capped_at_1000(self) -> None:
        m = ProcessingMetrics()
        for _ in range(1500):
            m.record_chunk(0.1, 1.0, 1)
        assert len(m.latency_samples) <= 1000

    def test_summary_no_division_by_zero(self) -> None:
        m = ProcessingMetrics()
        summary = m.get_summary()
        assert summary["realtime_factor"] == 0.0

    def test_realtime_factor_calculation(self) -> None:
        m = ProcessingMetrics()
        m.record_chunk(processing_time=0.2, audio_duration=1.0, queue_size=1)
        m.record_chunk(processing_time=0.4, audio_duration=1.0, queue_size=1)
        assert m.get_summary()["realtime_factor"] == 0.3


@pytest.mark.asyncio
class TestTranscriptionProcessor:
    """Async tests for TranscriptionProcessor."""

    async def test_processor_start_and_stop(
        self, config: AppConfig, mock_whisper: WhisperService
    ) -> None:
        """Processor can start and stop without errors."""
        queue: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=8)
        processor = TranscriptionProcessor(
            config=config,
            whisper_service=mock_whisper,
            input_queue=queue,
        )
        await processor.start()
        assert processor.is_running

        await processor.stop()
        assert not processor.is_running

    async def test_processor_processes_chunks(
        self, config: AppConfig, mock_whisper: WhisperService
    ) -> None:
        """Processor consumes and transcribes chunks from the queue."""
        queue: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=8)
        processor = TranscriptionProcessor(
            config=config,
            whisper_service=mock_whisper,
            input_queue=queue,
        )

        # Put some chunks in the queue
        for i in range(3):
            await queue.put(make_chunk(seq=i))

        await processor.start()

        # Wait for processing
        await asyncio.sleep(0.5)

        await processor.stop()

        # Verify chunks were processed
        assert mock_whisper.transcribe.call_count >= 1
        assert processor.metrics.chunks_processed >= 1

    async def test_processor_handles_empty_transcription(
        self, config: AppConfig, mock_whisper: WhisperService
    ) -> None:
        """Empty transcription results are handled gracefully."""
        mock_whisper.transcribe.return_value = TranscriptionResult(
            text="",
            segments=[],
            language="",
            language_probability=0.0,
            processing_time=0.05,
            input_duration=2.0,
        )

        queue: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=8)
        processor = TranscriptionProcessor(
            config=config,
            whisper_service=mock_whisper,
            input_queue=queue,
        )

        await queue.put(make_chunk(seq=0))
        await processor.start()
        await asyncio.sleep(0.5)
        await processor.stop()

        # No entries should be added to transcript
        assert len(processor.get_transcript_entries()) == 0
        assert mock_whisper.transcribe.call_count >= 1

    async def test_processor_handles_whisper_failure(
        self, config: AppConfig, mock_whisper: WhisperService
    ) -> None:
        """Whisper failures are handled without crashing the worker."""
        mock_whisper.transcribe.side_effect = RuntimeError("Whisper failed")

        queue: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=8)
        processor = TranscriptionProcessor(
            config=config,
            whisper_service=mock_whisper,
            input_queue=queue,
        )

        await queue.put(make_chunk(seq=0))
        await processor.start()
        await asyncio.sleep(0.5)
        await processor.stop()

        # Processor should still be alive
        assert not processor.is_running

    async def test_processor_backpressure(
        self, config: AppConfig, mock_whisper: WhisperService
    ) -> None:
        """Queue respects max size (backpressure)."""
        queue: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=2)

        # Fill the queue to max
        await queue.put(make_chunk(seq=0))
        await queue.put(make_chunk(seq=1))

        # This should block until a slot is freed
        put_task = asyncio.create_task(queue.put(make_chunk(seq=2)))

        processor = TranscriptionProcessor(
            config=config,
            whisper_service=mock_whisper,
            input_queue=queue,
        )
        await processor.start()

        # Wait for the put to complete (should happen quickly since worker drains)
        try:
            await asyncio.wait_for(put_task, timeout=5.0)
        except asyncio.TimeoutError:
            await processor.stop()
            pytest.fail("Backpressure blocked forever")

        await asyncio.sleep(0.3)
        await processor.stop()

    async def test_multiple_workers(
        self, config: AppConfig, mock_whisper: WhisperService
    ) -> None:
        """Multiple workers can run concurrently."""
        # Create config with multiple workers using object.__setattr__
        object.__setattr__(config.queue, "num_workers", 2)

        queue: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=16)
        processor = TranscriptionProcessor(
            config=config,
            whisper_service=mock_whisper,
            input_queue=queue,
        )

        # Put many chunks
        for i in range(10):
            await queue.put(make_chunk(seq=i))

        await processor.start()
        await asyncio.sleep(1.0)
        await processor.stop()

        assert mock_whisper.transcribe.call_count >= 1

    async def test_checkpoints_saved(
        self, config: AppConfig, mock_whisper: WhisperService
    ) -> None:
        """Checkpoints are saved periodically."""
        import os
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            object.__setattr__(config.checkpoint, "checkpoint_dir", Path(tmpdir))
            object.__setattr__(config.checkpoint, "interval", 0.3)

            mock_whisper.transcribe.return_value = TranscriptionResult(
                text="Test checkpoint",
                segments=[],
                language="en",
                language_probability=0.98,
                processing_time=0.1,
                input_duration=2.0,
            )

            queue: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=8)
            processor = TranscriptionProcessor(
                config=config,
                whisper_service=mock_whisper,
                input_queue=queue,
            )

            await queue.put(make_chunk(seq=0))
            await processor.start()
            await asyncio.sleep(1.0)
            await processor.stop()

            checkpoint_files = os.listdir(tmpdir)
            assert len(checkpoint_files) > 0, f"No checkpoints in {tmpdir}"

    async def test_metrics_collected(
        self, config: AppConfig, mock_whisper: WhisperService
    ) -> None:
        """Metrics are collected during processing."""
        object.__setattr__(config.metrics, "enabled", True)

        queue: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=8)
        processor = TranscriptionProcessor(
            config=config,
            whisper_service=mock_whisper,
            input_queue=queue,
        )

        for i in range(5):
            await queue.put(make_chunk(seq=i))

        initial_processed = processor.metrics.chunks_processed
        await processor.start()
        await asyncio.sleep(1.0)
        await processor.stop()

        assert processor.metrics.chunks_processed > initial_processed

    async def test_get_full_transcript(
        self, config: AppConfig, mock_whisper: WhisperService
    ) -> None:
        """Full transcript can be retrieved."""
        mock_whisper.transcribe.return_value = TranscriptionResult(
            text="Hello world",
            segments=[],
            language="en",
            language_probability=0.98,
            processing_time=0.1,
            input_duration=2.0,
        )

        queue: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=8)
        processor = TranscriptionProcessor(
            config=config,
            whisper_service=mock_whisper,
            input_queue=queue,
        )

        await queue.put(make_chunk(seq=0))
        await processor.start()
        await asyncio.sleep(0.5)
        await processor.stop()

        transcript = processor.get_full_transcript()
        assert isinstance(transcript, str)

    async def test_load_checkpoint_restores_state(
        self, config: AppConfig, mock_whisper: WhisperService
    ) -> None:
        """load_checkpoint restores transcript buffer and metadata from JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            checkpoint_file = checkpoint_dir / "checkpoint_20240101_120000.json"
            entries = [
                {"timestamp": 1000.0, "text": "Recovered text", "language": "en"},
                {"timestamp": 1002.0, "text": "Second entry", "language": "en"},
            ]
            data = {
                "session_start": 900.0,
                "saved_at": 1200.0,
                "entries": entries,
                "metrics": {"chunks_processed": 5, "chunks_dropped": 1},
                "flushed_entries": 10,
            }
            checkpoint_file.write_text(json.dumps(data))

            object.__setattr__(config.checkpoint, "checkpoint_dir", checkpoint_dir)

            queue: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=8)
            processor = TranscriptionProcessor(
                config=config,
                whisper_service=mock_whisper,
                input_queue=queue,
            )

            recovered = processor.load_checkpoint()
            assert recovered == 2

            # Verify state restoration
            assert len(processor._transcript_buffer) == 2
            assert processor._transcript_buffer[0]["text"] == "Recovered text"
            assert processor._flushed_count == 10
            assert processor._session_start == 1200.0
            assert processor._last_full_text == "Second entry"
            assert processor._metrics.chunks_processed == 5
            assert processor._metrics.chunks_dropped == 1
            assert processor._session_texts == ["Recovered text", "Second entry"]

            # Verify start() preserves checkpoint's _session_start
            await processor.start()
            assert processor._session_start == 1200.0  # Not overwritten
            await processor.stop()

    async def test_load_checkpoint_empty_dir_returns_zero(
        self, config: AppConfig, mock_whisper: WhisperService
    ) -> None:
        """load_checkpoint returns 0 when no checkpoint files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            object.__setattr__(config.checkpoint, "checkpoint_dir", Path(tmpdir))

            queue: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=8)
            processor = TranscriptionProcessor(
                config=config,
                whisper_service=mock_whisper,
                input_queue=queue,
            )

            recovered = processor.load_checkpoint()
            assert recovered == 0

    async def test_load_checkpoint_corrupt_file_returns_zero(
        self, config: AppConfig, mock_whisper: WhisperService
    ) -> None:
        """load_checkpoint returns 0 when checkpoint JSON is corrupt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            (checkpoint_dir / "checkpoint_20240101_120000.json").write_text(
                "{bad json"
            )
            object.__setattr__(config.checkpoint, "checkpoint_dir", checkpoint_dir)

            queue: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=8)
            processor = TranscriptionProcessor(
                config=config,
                whisper_service=mock_whisper,
                input_queue=queue,
            )

            recovered = processor.load_checkpoint()
            assert recovered == 0
