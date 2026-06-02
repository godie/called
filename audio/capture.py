"""Audio capture module using sounddevice.

Captures audio from microphone and/or system audio (via BlackHole on macOS)
in small chunks and pushes them to a ring buffer and asyncio queue.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from audio.archiver import AudioArchiver
from audio.buffers import AudioChunk, RingBuffer
from audio.devices import get_default_input_device, validate_device
from audio.silence import SilenceDetector

logger = logging.getLogger("realtime-transcriber.capture")


class AudioCapture:
    """Captures audio from input devices in chunks.

    Uses sounddevice's InputStream with callback mode for non-blocking
    capture. Audio data is pushed to a ring buffer and optionally to
    an asyncio queue for downstream processing.
    """

    def __init__(
        self,
        device_index: Optional[int],
        sample_rate: int = 16000,
        chunk_duration: float = 2.0,
        channels: int = 1,
        ring_buffer: Optional[RingBuffer] = None,
        queue: Optional[asyncio.Queue[AudioChunk]] = None,
        archiver: Optional["AudioArchiver"] = None,
        silence_detector: Optional[SilenceDetector] = None,
        on_permanent_failure: Optional[Callable[[], None]] = None,
    ) -> None:
        """Initialize audio capture.

        Args:
            device_index: Audio input device index.
            sample_rate: Sample rate in Hz.
            chunk_duration: Duration of each audio chunk in seconds.
            channels: Number of audio channels (1 = mono).
            ring_buffer: Optional ring buffer for chunk storage.
            queue: Optional asyncio queue for consumer processing.
            archiver: Optional audio archiver for compressed storage.
            silence_detector: Optional silence detector to skip silent chunks.
            on_permanent_failure: Called when auto-recovery is exhausted.
        """
        self._device_index: Optional[int] = device_index
        self._sample_rate: int = sample_rate
        self._chunk_duration: float = chunk_duration
        self._channels: int = channels
        self._blocksize: int = int(sample_rate * chunk_duration)
        self._ring_buffer: Optional[RingBuffer] = ring_buffer
        self._queue: Optional[asyncio.Queue[AudioChunk]] = queue
        self._archiver: Optional["AudioArchiver"] = archiver
        self._silence_detector: Optional[SilenceDetector] = silence_detector
        self._silent_chunks_skipped: int = 0
        self._on_permanent_failure: Optional[Callable[[], None]] = on_permanent_failure

        self._stream: Optional[sd.InputStream] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._chunk_counter: int = 0
        self._running: bool = False

        # Auto-recovery state
        # NOTE: _recovering is read/written from both the PortAudio callback
        # thread and the event loop thread. CPython's GIL makes bool
        # assignments atomic, but this should use threading.Event in
        # free-threaded Python builds.
        self._recovering: bool = False
        self._stopped_permanently: bool = False
        self._restart_count: int = 0
        self._max_restarts: int = 10
        self._backoff_base: float = 1.0  # seconds, doubles each attempt

    @property
    def is_running(self) -> bool:
        """Whether the capture stream is active."""
        return self._running

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        timestamp: sd.CallbackTime,
        status: sd.CallbackFlags,
    ) -> None:
        """Called by sounddevice for each audio block.

        This runs in the PortAudio callback thread. Heavy processing
        should be deferred to the asyncio loop.

        Args:
            indata: Captured audio samples.
            frames: Number of frames captured.
            timestamp: Callback timestamp information.
            status: Status flags.
        """
        if status:
            logger.warning("Audio callback status: %s", status)
            # Check for device disconnection or error flags
            if self._maybe_recover_from_status(status):
                return

        if not self._running or self._recovering:
            return

        try:
            # Skip silent chunks BEFORE copying — saves CPU allocation
            if self._silence_detector is not None:
                if self._silence_detector.is_silence(indata):
                    self._silent_chunks_skipped += 1
                    return

            raw_data = indata.copy()

            chunk = AudioChunk(
                data=raw_data,
                sample_rate=self._sample_rate,
                timestamp=time.time(),
                source="mic" if self._device_index else "unknown",
                sequence_id=self._chunk_counter,
            )
            self._chunk_counter += 1

            # Schedule non-blocking archival on the event loop thread
            if self._archiver is not None and self._loop is not None:
                asyncio.run_coroutine_threadsafe(
                    self._archive_async(raw_data.copy()),
                    self._loop,
                )

            # Push to ring buffer if configured
            if self._ring_buffer is not None:
                self._ring_buffer.put(chunk)

            # Push to asyncio queue if configured
            if self._queue is not None and self._loop is not None:
                # Thread-safe: schedule the put on the event loop
                asyncio.run_coroutine_threadsafe(
                    self._queue.put(chunk), self._loop
                )

        except Exception as exc:
            logger.error("Error in audio callback: %s", exc, exc_info=True)

    async def _archive_async(self, data: np.ndarray) -> None:
        """Archive audio chunk asynchronously on the event loop.

        Offloads the actual disk I/O to a thread executor to avoid
        blocking the event loop.

        Args:
            data: Raw audio samples to archive.
        """
        if self._archiver is None:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._archiver.archive_chunk, data)

    def start(
        self,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        """Start audio capture.

        Args:
            loop: The asyncio event loop for thread-safe queue operations.
        """
        if self._running:
            logger.warning("Audio capture already running")
            return

        if self._device_index is None:
            self._device_index = get_default_input_device()
            if self._device_index is None:
                raise RuntimeError("No input audio device found")

        # Validate the device
        if not validate_device(
            self._device_index, self._sample_rate, self._channels
        ):
            raise RuntimeError(f"Device {self._device_index} validation failed")

        self._loop = loop or asyncio.get_event_loop()
        self._running = True
        self._stopped_permanently = False  # Explicit start clears permanent-stop

        try:
            self._stream = sd.InputStream(
                device=self._device_index,
                channels=self._channels,
                samplerate=self._sample_rate,
                blocksize=self._blocksize,
                callback=self._audio_callback,
                dtype=np.float32,
            )
            self._stream.start()
            logger.info(
                "Audio capture started [device=%d, rate=%d, chunk=%.1fs, channels=%d]",
                self._device_index,
                self._sample_rate,
                self._chunk_duration,
                self._channels,
            )
        except sd.PortAudioError as exc:
            self._running = False
            logger.error("Failed to start audio stream: %s", exc)
            raise RuntimeError(f"Audio stream failed: {exc}") from exc

    def stop(self, permanent: bool = False) -> None:
        """Stop audio capture and close the stream.

        Args:
            permanent: If True, prevents auto-recovery from restarting.
        """
        self._stopped_permanently = self._stopped_permanently or permanent

        # Stop the stream FIRST to prevent callback races
        stream_to_stop = self._stream
        self._stream = None

        if stream_to_stop is not None:
            try:
                stream_to_stop.stop()
                stream_to_stop.close()
                logger.info("Audio capture stopped")
            except Exception as exc:
                logger.error("Error stopping audio stream: %s", exc)

        # Now mark as not running AFTER stream is closed
        self._running = False

    def _maybe_recover_from_status(self, status: sd.CallbackFlags) -> bool:
        """Check status flags and schedule recovery if needed.

        Returns True if recovery was triggered (caller should skip processing).
        """
        status_str = str(status).lower()

        # PortAudio error flags that indicate device issues
        error_indicators = (
            "input overflow" in status_str
            or "input underflow" in status_str
            or "aborted" in status_str
            or "error" in status_str
        )

        if error_indicators and not self._recovering and not self._stopped_permanently:
            if self._loop is not None and self._loop.is_running():
                self._recovering = True
                logger.warning(
                    "Device error detected (restart #%d): %s",
                    self._restart_count + 1,
                    status,
                )
                # Schedule restart coroutine directly (single event-loop hop)
                asyncio.run_coroutine_threadsafe(
                    self._do_restart(), self._loop
                )
                return True

        return False

    async def _do_restart(self) -> None:
        """Async restart with exponential backoff.

        Runs on the event loop thread. Handles retry scheduling
        with backoff up to _max_restarts attempts.
        """
        # Abort if deliberately stopped (e.g., app shutdown)
        if self._stopped_permanently:
            self._recovering = False
            return

        if self._restart_count >= self._max_restarts:
            logger.error(
                "Max restart attempts (%d) reached. Giving up on device %d.",
                self._max_restarts,
                self._device_index,
            )
            self._running = False
            self._recovering = False
            # Notify the application that capture has permanently failed
            if self._on_permanent_failure is not None:
                try:
                    self._on_permanent_failure()
                except Exception as exc:
                    logger.error("on_permanent_failure callback error: %s", exc)
            return

        # Exponential backoff
        delay = self._backoff_base * (2 ** self._restart_count)
        logger.info(
            "Waiting %.1fs before restart attempt #%d...",
            delay,
            self._restart_count + 1,
        )
        await asyncio.sleep(delay)

        # Also abort if stopped during the backoff sleep
        if self._stopped_permanently:
            self._recovering = False
            return

        success = await asyncio.get_running_loop().run_in_executor(
            None, self.restart
        )

        if success:
            self._restart_count = 0
            self._backoff_base = 1.0
            self._recovering = False
        else:
            self._restart_count += 1
            # Schedule next retry; keep _recovering=True to prevent
            # _maybe_recover_from_status from spawning a parallel chain.
            # _recovering is only cleared on success or permanent failure.
            if not self._stopped_permanently:
                asyncio.run_coroutine_threadsafe(
                    self._do_restart(), self._loop
                )
            else:
                self._recovering = False

    def restart(self) -> bool:
        """Attempt to restart the audio stream after a disconnection.

        Returns:
            True if restart succeeded, False otherwise.
        """
        # Don't restart if deliberately stopped (e.g., app shutdown)
        if self._stopped_permanently:
            logger.info("Skipping restart: capture was permanently stopped")
            return False

        logger.info("Attempting to restart audio capture...")
        self.stop()
        try:
            time.sleep(0.5)
            self._stopped_permanently = False  # Clear permanent stop on successful restart
            self.start(loop=self._loop)
            logger.info("Audio capture restarted successfully")
            return True
        except Exception as exc:
            logger.error("Failed to restart audio capture: %s", exc)
            return False

    def get_device_index(self) -> Optional[int]:
        """Get the current device index."""
        return self._device_index

    @property
    def is_recovering(self) -> bool:
        """Whether the capture is currently attempting auto-recovery."""
        return self._recovering

    @property
    def restart_attempts(self) -> int:
        """Number of restart attempts since last success."""
        return self._restart_count

    @property
    def silent_chunks_skipped(self) -> int:
        """Number of silent chunks skipped (saves CPU/disk)."""
        return self._silent_chunks_skipped


class SystemAudioCapture(AudioCapture):
    """Specialized capture for system audio via BlackHole on macOS.

    On macOS, BlackHole creates a virtual input device that mirrors
    system audio output when configured as a Multi-Output Device.
    """

    def __init__(
        self,
        device_index: int,
        sample_rate: int = 16000,
        chunk_duration: float = 2.0,
        channels: int = 2,  # BlackHole typically outputs stereo
        ring_buffer: Optional[RingBuffer] = None,
        queue: Optional[asyncio.Queue[AudioChunk]] = None,
        archiver: Optional[AudioArchiver] = None,
        silence_detector: Optional[SilenceDetector] = None,
    ) -> None:
        """Initialize system audio capture.

        Args:
            device_index: BlackHole device index.
            sample_rate: Sample rate in Hz.
            chunk_duration: Duration of each audio chunk in seconds.
            channels: Number of channels (2 for stereo).
            ring_buffer: Optional ring buffer for chunk storage.
            queue: Optional asyncio queue for consumer processing.
            archiver: Optional audio archiver for compressed storage.
            silence_detector: Optional silence detector to skip silent chunks.
        """
        super().__init__(
            device_index=device_index,
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            channels=channels,
            ring_buffer=ring_buffer,
            queue=queue,
            archiver=archiver,
            silence_detector=silence_detector,
        )
