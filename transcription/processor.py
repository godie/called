"""Producer/consumer processor for audio transcription pipeline.

Orchestrates the flow: audio chunks → queue → whisper transcription → output.
Includes metrics collection, checkpointing, and backpressure management.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

from audio.buffers import AudioChunk
from config import AppConfig
from storage.checkpoint import find_latest_checkpoint, load_checkpoint_data
from storage.txt_writer import TXTWriter
from transcription.whisper_service import TranscriptionResult, WhisperService

logger = logging.getLogger("realtime-transcriber.processor")


@dataclass
class ProcessingMetrics:
    """Metrics for the processing pipeline."""

    chunks_processed: int = 0
    chunks_dropped: int = 0
    total_processing_time: float = 0.0
    total_audio_duration: float = 0.0
    min_latency: float = float("inf")
    max_latency: float = 0.0
    avg_latency: float = 0.0
    queue_size_samples: list[int] = field(default_factory=list)
    latency_samples: list[float] = field(default_factory=list)
    last_report_time: float = field(default_factory=time.time)

    def record_chunk(
        self,
        processing_time: float,
        audio_duration: float,
        queue_size: int,
    ) -> None:
        """Record metrics for a processed chunk.

        Args:
            processing_time: Time spent processing in seconds.
            audio_duration: Duration of audio in seconds.
            queue_size: Current queue size.
        """
        self.chunks_processed += 1
        self.total_processing_time += processing_time
        self.total_audio_duration += audio_duration

        if processing_time < self.min_latency:
            self.min_latency = processing_time
        if processing_time > self.max_latency:
            self.max_latency = processing_time

        self.latency_samples.append(processing_time)
        self.queue_size_samples.append(queue_size)

        # Keep only last 1000 samples
        if len(self.latency_samples) > 1000:
            self.latency_samples = self.latency_samples[-1000:]
        if len(self.queue_size_samples) > 1000:
            self.queue_size_samples = self.queue_size_samples[-1000:]

        self.avg_latency = (
            sum(self.latency_samples) / len(self.latency_samples)
            if self.latency_samples
            else 0.0
        )

    def record_dropped(self) -> None:
        """Record a dropped chunk."""
        self.chunks_dropped += 1

    def get_summary(self) -> dict:
        """Get a summary of current metrics.

        Returns:
            Dictionary with metrics summary.
        """
        avg_queue = (
            sum(self.queue_size_samples) / len(self.queue_size_samples)
            if self.queue_size_samples
            else 0.0
        )

        realtime_factor = (
            self.total_processing_time / self.total_audio_duration
            if self.total_audio_duration > 0
            else 0.0
        )

        return {
            "chunks_processed": self.chunks_processed,
            "chunks_dropped": self.chunks_dropped,
            "min_latency_ms": round(self.min_latency * 1000, 2),
            "max_latency_ms": round(self.max_latency * 1000, 2),
            "avg_latency_ms": round(self.avg_latency * 1000, 2),
            "avg_queue_size": round(avg_queue, 1),
            "realtime_factor": round(realtime_factor, 3),
            "total_audio_s": round(self.total_audio_duration, 1),
            "total_processing_s": round(self.total_processing_time, 1),
        }


class TranscriptionProcessor:
    """Consumer that processes audio chunks from a queue through Whisper.

    Implements:
    - Queue-based processing with backpressure
    - Metrics collection (latency, queue size, processing time)
    - Automatic checkpoint saves
    - Graceful shutdown
    """

    def __init__(
        self,
        config: AppConfig,
        whisper_service: WhisperService,
        input_queue: asyncio.Queue[AudioChunk],
    ) -> None:
        """Initialize the processor.

        Args:
            config: Application configuration.
            whisper_service: Loaded Whisper service.
            input_queue: Queue to read audio chunks from.
        """
        self._config: AppConfig = config
        self._whisper: WhisperService = whisper_service
        self._queue: asyncio.Queue[AudioChunk] = input_queue
        self._metrics: ProcessingMetrics = ProcessingMetrics()
        self._transcript_buffer: list[dict] = []
        self._running: bool = False
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._checkpoint_event: asyncio.Event = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

        # Last full transcript for deduplication
        self._last_full_text: str = ""
        self._session_start: float = 0.0  # Set in start() or load_checkpoint()
        self._session_texts: list[str] = []

        # Hourly rotation state
        self._last_rotation_time: float = time.time()
        self._rotation_count: int = 0
        self._flushed_count: int = 0  # Entries flushed to disk

        # Max entries derived from config (prevent unbounded memory)
        self._max_entries: int = self._config.queue.max_size * 150

    @property
    def metrics(self) -> ProcessingMetrics:
        """Get the current processing metrics."""
        return self._metrics

    @property
    def is_running(self) -> bool:
        """Whether the processor is running."""
        return self._running

    def load_checkpoint(self) -> int:
        """Restore state from the most recent checkpoint after a crash.

        Searches the checkpoint directory for the latest checkpoint file,
        loads entries into the transcript buffer, and restores metadata.
        Logs a summary of what was recovered.

        Returns:
            Number of entries restored (0 if no checkpoint found).
        """
        checkpoint_dir = self._config.checkpoint.checkpoint_dir
        latest = find_latest_checkpoint(checkpoint_dir)

        if latest is None:
            logger.info("No checkpoint found — starting fresh session")
            return 0

        data = load_checkpoint_data(latest)
        if data is None:
            return 0

        entries = data.get("entries", [])
        if not entries:
            logger.info("Checkpoint %s has no entries, starting fresh", latest)
            return 0

        # Restore transcript buffer
        self._transcript_buffer = entries
        self._session_texts = [e.get("text", "") for e in entries]
        self._flushed_count = data.get("flushed_entries", 0)

        # Restore last text for dedup continuity
        if entries:
            self._last_full_text = entries[-1].get("text", "")

        # Restore approximate session start from checkpoint timestamp
        saved_at = data.get("saved_at", time.time())
        try:
            saved_at = float(saved_at)
        except (TypeError, ValueError):
            saved_at = time.time()
        self._session_start = saved_at

        # Partially restore metrics (counters, not latency samples)
        saved_metrics = data.get("metrics", {})
        self._metrics.chunks_processed = saved_metrics.get("chunks_processed", 0)
        self._metrics.chunks_dropped = saved_metrics.get("chunks_dropped", 0)

        logger.info(
            "Checkpoint recovery complete: %d entries restored from %s",
            len(entries),
            latest.name,
        )
        return len(entries)

    async def start(self) -> None:
        """Start the processor and all background tasks."""
        self._running = True
        self._last_rotation_time = time.time()

        # Set session start if no checkpoint was loaded (checkpoint sets it)
        if self._session_start == 0.0:
            self._session_start = time.time()

        # Start worker tasks
        for i in range(self._config.queue.num_workers):
            task = asyncio.create_task(
                self._worker(i), name=f"transcriber-worker-{i}"
            )
            self._tasks.append(task)

        # Start metrics reporter
        if self._config.metrics.enabled:
            metrics_task = asyncio.create_task(
                self._metrics_reporter(), name="metrics-reporter"
            )
            self._tasks.append(metrics_task)

        # Start checkpoint saver
        checkpoint_task = asyncio.create_task(
            self._checkpoint_saver(), name="checkpoint-saver"
        )
        self._tasks.append(checkpoint_task)

        # Start hourly rotation task
        rotation_task = asyncio.create_task(
            self._rotation_scheduler(), name="rotation-scheduler"
        )
        self._tasks.append(rotation_task)

        logger.info(
            "Processor started with %d workers",
            self._config.queue.num_workers,
        )

    async def stop(self) -> None:
        """Gracefully stop the processor."""
        logger.info("Processor stopping...")
        self._running = False
        self._shutdown_event.set()

        # Wait for all tasks to finish
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()

        # Final checkpoint
        await self._save_checkpoint()

        logger.info(
            "Processor stopped. Processed %d chunks in %.1fs",
            self._metrics.chunks_processed,
            time.time() - self._session_start,
        )

    async def _worker(self, worker_id: int) -> None:
        """Worker coroutine that processes audio chunks.

        Args:
            worker_id: Worker identifier for logging.
        """
        logger.info("Worker %d started", worker_id)

        while self._running:
            chunk = None
            try:
                # Wait for chunk with timeout to check shutdown
                try:
                    chunk = await asyncio.wait_for(
                        self._queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Process the chunk
                result = await self._process_chunk(chunk)

                # Record metrics
                queue_size = self._queue.qsize()
                self._metrics.record_chunk(
                    processing_time=result.processing_time,
                    audio_duration=result.input_duration,
                    queue_size=queue_size,
                )

                # Check for transcription errors
                if result.error:
                    logger.warning(
                        "Chunk seq=%d had transcription error: %s",
                        chunk.sequence_id,
                        result.error,
                    )
                    self._metrics.record_dropped()
                # Add to transcript buffer if we got text
                elif result.text.strip():
                    self._add_to_transcript(result, chunk.timestamp)

            except asyncio.CancelledError:
                logger.info("Worker %d cancelled", worker_id)
                break
            except Exception as exc:
                logger.error(
                    "Worker %d error: %s", worker_id, exc, exc_info=True
                )
            finally:
                if chunk is not None:
                    self._queue.task_done()

    async def _process_chunk(self, chunk: AudioChunk) -> TranscriptionResult:
        """Process a single audio chunk through Whisper.

        Args:
            chunk: Audio chunk to transcribe.

        Returns:
            Transcription result.
        """
        # Run transcription in thread executor to avoid blocking
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            self._whisper.transcribe,
            chunk.data,
            chunk.sample_rate,
        )
        return result

    def _add_to_transcript(
        self, result: TranscriptionResult, timestamp: float
    ) -> None:
        """Add a transcription result to the transcript buffer.

        Handles deduplication of overlapping chunks using word overlap ratio.

        Args:
            result: Transcription result.
            timestamp: Timestamp of the audio chunk.
        """
        text = result.text.strip()

        if not text:
            return

        # Advanced dedup: use word overlap ratio with the last chunk
        if self._last_full_text:
            overlap_ratio = self._word_overlap_ratio(self._last_full_text, text)
            if overlap_ratio > 0.7:
                # Keep the longer version
                if len(text) > len(self._last_full_text):
                    self._last_full_text = text
                self._print_partial(text)
                return

        self._last_full_text = text

        # Language info
        lang_info = f" [{result.language}]" if result.language else ""

        entry = {
            "timestamp": timestamp,
            "text": text,
            "language": result.language,
            "confidence": result.language_probability,
        }
        self._transcript_buffer.append(entry)
        self._session_texts.append(text)

        # Bound memory: trim oldest entries if exceeding max
        self._trim_transcript_buffer()

        self._print_transcript(text + lang_info)

    @staticmethod
    def _word_overlap_ratio(text_a: str, text_b: str) -> float:
        """Calculate word overlap ratio between two texts.

        Uses Jaccard-like overlap on word trigrams for robustness.

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            Overlap ratio between 0.0 and 1.0.
        """
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())

        if not words_a or not words_b:
            return 0.0

        intersection = words_a & words_b
        union = words_a | words_b

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def _trim_transcript_buffer(self) -> None:
        """Trim transcript buffer to prevent unbounded memory growth."""
        if len(self._transcript_buffer) > self._max_entries:
            excess = len(self._transcript_buffer) - self._max_entries
            self._transcript_buffer = self._transcript_buffer[excess:]
            self._flushed_count += excess
            logger.debug(
                "Trimmed %d old transcript entries (total flushed: %d)",
                excess,
                self._flushed_count,
            )

        if len(self._session_texts) > self._max_entries:
            excess = len(self._session_texts) - self._max_entries
            self._session_texts = self._session_texts[excess:]

    def _print_partial(self, text: str) -> None:
        """Print a partial/interim transcription result.

        Args:
            text: Partial transcription text.
        """
        # Clear line and print partial
        print(f"\r\033[K[partial] {text}", end="", flush=True)

    def _print_transcript(self, text: str) -> None:
        """Print a final transcription result.

        Args:
            text: Final transcription text.
        """
        # Clear any partial output and print final
        print(f"\r\033[K{text}", flush=True)

    async def _metrics_reporter(self) -> None:
        """Periodically report processing metrics."""
        interval = self._config.metrics.interval

        while self._running:
            try:
                await asyncio.sleep(interval)
                summary = self._metrics.get_summary()
                logger.info(
                    "METRICS | chunks=%d dropped=%d latency=%.1fms "
                    "queue=%.1f rtf=%.3f",
                    summary["chunks_processed"],
                    summary["chunks_dropped"],
                    summary["avg_latency_ms"],
                    summary["avg_queue_size"],
                    summary["realtime_factor"],
                )
            except asyncio.CancelledError:
                break

    async def _checkpoint_saver(self) -> None:
        """Periodically save transcript checkpoints."""
        interval = self._config.checkpoint.interval
        checkpoint_dir = self._config.checkpoint.checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        while self._running:
            try:
                await asyncio.sleep(interval)
                await self._save_checkpoint()
            except asyncio.CancelledError:
                break

    async def _save_checkpoint(self) -> None:
        """Save the current transcript as a checkpoint."""
        if not self._transcript_buffer:
            return

        checkpoint_dir = self._config.checkpoint.checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_path = checkpoint_dir / f"checkpoint_{ts}.json"

        try:
            loop = asyncio.get_running_loop()
            data = {
                "session_start": self._session_start,
                "saved_at": time.time(),
                "entries": self._transcript_buffer[-2000:],  # Cap checkpoint size
                "metrics": self._metrics.get_summary(),
                "flushed_entries": self._flushed_count,
            }
            await loop.run_in_executor(
                None,
                lambda: json.dump(
                    data,
                    checkpoint_path.open("w"),
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            logger.debug("Checkpoint saved: %s", checkpoint_path)
        except Exception as exc:
            logger.error("Failed to save checkpoint: %s", exc)

    async def _rotation_scheduler(self) -> None:
        """Periodically rotate transcript files to prevent data loss."""
        interval = self._config.output.rotation_interval
        output_dir = self._config.output.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        while self._running:
            try:
                await asyncio.sleep(interval)
                await self._rotate_transcript()
            except asyncio.CancelledError:
                break

    async def _rotate_transcript(self) -> None:
        """Flush current transcript to disk and start a new rotation file."""
        if not self._transcript_buffer:
            return

        output_dir = self._config.output.output_dir
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"transcript_rotation_{self._rotation_count + 1:04d}_{ts}"

        try:
            writer = TXTWriter(output_dir)
            writer.write(
                list(self._transcript_buffer),
                filename=f"{filename}.txt",
            )

            self._rotation_count += 1
            self._last_rotation_time = time.time()
            logger.info(
                "Transcript rotation #%d saved (%d entries) to %s/",
                self._rotation_count,
                len(self._transcript_buffer),
                output_dir,
            )
        except Exception as exc:
            logger.error("Failed to rotate transcript: %s", exc)

    def get_full_transcript(self) -> str:
        """Get the complete session transcript as a single string.

        Returns:
            Full transcript text.
        """
        return "\n".join(self._session_texts)

    def get_transcript_entries(self) -> list[dict]:
        """Get all transcript entries with metadata.

        Returns:
            List of transcript entry dictionaries.
        """
        return list(self._transcript_buffer)
