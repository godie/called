"""Real-time Audio Transcription Application.

Continuously captures microphone and/or system audio and generates
live transcriptions using faster-whisper.

Usage:
    python app.py                  # Auto-detect mic
    python app.py --list-devices   # List available devices
    python app.py --device 2       # Use specific device index

Interactive commands (while running):
    r — Toggle recording (pause/resume)
    s — Show status
    q — Quit (graceful shutdown)
    h — Show help
"""

import argparse
import asyncio
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

from audio.archiver import AudioArchiver
from audio.capture import AudioCapture, SystemAudioCapture
from audio.silence import SilenceDetector
from audio.devices import (
    find_device_by_name,
    get_default_input_device,
    print_available_devices,
)
from cli.command_interface import CommandInterface
from config import AppConfig, load_config
from storage.json_writer import JSONWriter
from storage.srt_writer import SRTWriter
from storage.txt_writer import TXTWriter
from transcription.processor import TranscriptionProcessor
from transcription.whisper_service import WhisperService
from utils.logger import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-time audio transcription using faster-whisper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py                        # Auto-detect microphone
  python app.py --list-devices         # List all audio devices
  python app.py --device 2             # Use device index 2
  python app.py --system-audio 4       # Capture system audio (BlackHole)
  python app.py --model large-v3       # Use large-v3 model
  python app.py --language en          # Force English transcription
        """,
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Microphone device index (auto-detected if not set)",
    )
    parser.add_argument(
        "--system-audio",
        type=int,
        default=None,
        help="System audio device index (BlackHole on macOS)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Whisper model size (tiny, base, small, medium, large-v3)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Force transcription language (e.g., en, es)",
    )
    return parser.parse_args()


class Application:
    """Main application orchestrating audio capture and transcription."""

    def __init__(self, config: AppConfig) -> None:
        """Initialize the application.

        Args:
            config: Application configuration.
        """
        self._config: AppConfig = config
        self._queue: asyncio.Queue = asyncio.Queue(
            maxsize=config.queue.max_size
        )
        self._whisper: Optional[WhisperService] = None
        self._processor: Optional[TranscriptionProcessor] = None
        self._mic_capture: Optional[AudioCapture] = None
        self._system_capture: Optional[SystemAudioCapture] = None
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._start_time: float = 0.0
        self._recording: bool = True
        self._archiver: Optional[AudioArchiver] = None

    async def run(self) -> int:
        """Run the application.

        Returns:
            Exit code (0 = success).
        """
        _logger = logging.getLogger("realtime-transcriber")
        self._start_time = time.time()

        _logger.info("=" * 60)
        _logger.info("Real-time Audio Transcription Starting...")
        _logger.info("=" * 60)

        try:
            # 1. Load Whisper model
            _logger.info("Step 1/4: Loading Whisper model...")
            self._whisper = WhisperService(
                model_size=self._config.whisper.model_size,
                device=self._config.whisper.device,
                compute_type=self._config.whisper.compute_type,
                beam_size=self._config.whisper.beam_size,
                language=self._config.whisper.language,
                vad_threshold=self._config.whisper.vad_threshold,
            )
            self._whisper.load_model()

            # 2. Start processor (consumer)
            _logger.info("Step 2/4: Starting transcription processor...")
            self._processor = TranscriptionProcessor(
                config=self._config,
                whisper_service=self._whisper,
                input_queue=self._queue,
            )

            # Attempt checkpoint recovery from previous crashed session
            if self._config.checkpoint.recovery_enabled:
                recovered = self._processor.load_checkpoint()
                if recovered > 0:
                    _logger.info(
                        "🔄 Recovered %d transcript entries from checkpoint",
                        recovered,
                    )
                    print(f"\n🔄 Recovered {recovered} transcript entries from previous session")

            await self._processor.start()

            # 3. Start audio capture (producers)
            _logger.info("Step 3/4: Starting audio capture...")
            loop = asyncio.get_running_loop()

            # Initialize silence detector to skip silent chunks
            mic_silence: Optional[SilenceDetector] = None
            sys_silence: Optional[SilenceDetector] = None
            if self._config.silence_detection.enabled:
                mic_silence = SilenceDetector(
                    threshold_db=self._config.silence_detection.threshold_db,
                    sample_rate=self._config.audio.sample_rate,
                    recalibration_interval_s=(
                        self._config.silence_detection.recalibration_interval_s
                        if self._config.silence_detection.recalibration_enabled
                        else 0.0
                    ),
                    recalibration_margin_db=self._config.silence_detection.recalibration_margin_db,
                )
                # System audio may have different source characteristics —
                # use a separate detector instance per capture for thread safety
                sys_silence = SilenceDetector(
                    threshold_db=self._config.silence_detection.threshold_db,
                    sample_rate=self._config.audio.sample_rate,
                    recalibration_interval_s=(
                        self._config.silence_detection.recalibration_interval_s
                        if self._config.silence_detection.recalibration_enabled
                        else 0.0
                    ),
                    recalibration_margin_db=self._config.silence_detection.recalibration_margin_db,
                )
                _logger.info(
                    "Silence detection enabled: threshold=%.0f dB, recalibration=%s",
                    self._config.silence_detection.threshold_db,
                    "on" if self._config.silence_detection.recalibration_enabled else "off",
                )

            # Initialize audio archiver if compressed storage is enabled
            if self._config.audio_archive.save_compressed_audio:
                self._archiver = AudioArchiver(
                    archive_dir=self._config.audio_archive.archive_dir,
                    sample_rate=self._config.audio.sample_rate,
                    channels=1,
                    bitrate=self._config.audio_archive.bitrate,
                    save_raw=self._config.audio_archive.save_raw_audio,
                )
                _logger.info(
                    "Audio archiver enabled: codec=%s bitrate=%s dir=%s",
                    self._config.audio_archive.codec,
                    self._config.audio_archive.bitrate,
                    self._config.audio_archive.archive_dir,
                )

            # Microphone capture
            mic_device = self._config.audio.mic_device_index
            if mic_device is None:
                mic_device = get_default_input_device()

            if mic_device is not None:
                self._mic_capture = AudioCapture(
                    device_index=mic_device,
                    sample_rate=self._config.audio.sample_rate,
                    chunk_duration=self._config.audio.chunk_duration,
                    channels=self._config.audio.channels,
                    queue=self._queue,
                    archiver=self._archiver,
                    silence_detector=mic_silence,
                )
                self._mic_capture.start(loop=loop)

            # System audio capture
            sys_device = self._config.audio.system_audio_device_index
            if sys_device is not None:
                self._system_capture = SystemAudioCapture(
                    device_index=sys_device,
                    sample_rate=self._config.audio.sample_rate,
                    chunk_duration=self._config.audio.chunk_duration,
                    channels=2,
                    queue=self._queue,
                    archiver=self._archiver,
                    silence_detector=sys_silence,
                )
                self._system_capture.start(loop=loop)

            # 4. Start command interface and wait for shutdown
            _logger.info("Step 4/4: Running. Use 'r' to toggle recording, 'q' to quit.")
            print("\n" + "=" * 60)
            print("  🎤  Listening... Press 'q' to quit, 'h' for help")
            print("=" * 60)

            cli = CommandInterface(
                on_toggle_recording=self._toggle_recording,
                on_show_status=self._get_status,
                on_quit=self.signal_shutdown,
            )
            # Run CLI concurrently with shutdown signal so both 'q' and
            # SIGTERM trigger exit (whichever fires first)
            cli_task = asyncio.create_task(cli.run())
            shutdown_waiter = asyncio.create_task(
                self._shutdown_event.wait()
            )

            done, pending = await asyncio.wait(
                [cli_task, shutdown_waiter],
                return_when=asyncio.FIRST_COMPLETED,
            )
            # Cancel whichever task didn't complete
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        except Exception as exc:
            _logger.error("Application error: %s", exc, exc_info=True)
            return 1

        finally:
            await self._shutdown()
            return 0

    async def _shutdown(self) -> None:
        """Gracefully shut down all components."""
        _logger = logging.getLogger("realtime-transcriber")
        _logger.info("Initiating graceful shutdown...")

        # Stop capture first (producers) — permanent prevents auto-recovery
        if self._mic_capture and self._mic_capture.is_running:
            self._mic_capture.stop(permanent=True)

        if self._system_capture and self._system_capture.is_running:
            self._system_capture.stop(permanent=True)

        # Stop processor (consumers drain the queue)
        if self._processor and self._processor.is_running:
            await self._processor.stop()

        # Log per-source silence stats for debugging dead-air sources
        mic_skipped = (
            self._mic_capture.silent_chunks_skipped if self._mic_capture else 0
        )
        sys_skipped = (
            self._system_capture.silent_chunks_skipped
            if self._system_capture
            else 0
        )
        if mic_skipped or sys_skipped:
            chunk_s = self._config.audio.chunk_duration
            _logger.info(
                "Silence stats — mic: %d chunks (%ds), sys: %d chunks (%ds)",
                mic_skipped,
                int(mic_skipped * chunk_s),
                sys_skipped,
                int(sys_skipped * chunk_s),
            )

        # Save final transcripts
        await self._save_transcripts()

        elapsed = time.time() - self._start_time
        _logger.info(
            "Shutdown complete. Ran for %s",
            f"{elapsed:.1f}s ({elapsed/3600:.1f}h)",
        )

    async def _save_transcripts(self) -> None:
        """Save the session transcript to all output formats."""
        if self._processor is None:
            return

        _logger = logging.getLogger("realtime-transcriber")

        entries = self._processor.get_transcript_entries()
        if not entries:
            _logger.info("No transcript entries to save")
            return
        output_dir = self._config.output.output_dir
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        try:
            # TXT
            txt_writer = TXTWriter(output_dir)
            txt_writer.write(entries, filename=f"transcript_{timestamp}.txt")

            # JSON
            json_writer = JSONWriter(output_dir)
            json_writer.write(
                entries,
                metadata={
                    "model": self._config.whisper.model_size,
                    "duration_s": time.time() - self._start_time,
                    "metrics": self._processor.metrics.get_summary(),
                },
                filename=f"transcript_{timestamp}.json",
            )

            # SRT
            srt_writer = SRTWriter(output_dir)
            srt_writer.write(entries, filename=f"transcript_{timestamp}.srt")

            _logger.info("Transcripts saved to %s/", output_dir)
            print(f"\n✅ Transcripts saved to {output_dir}/")

        except Exception as exc:
            _logger.error("Failed to save transcripts: %s", exc)

    def signal_shutdown(self) -> None:
        """Signal the application to shut down."""
        self._shutdown_event.set()

    def _toggle_recording(self) -> Optional[bool]:
        """Toggle recording on/off. Pauses or resumes audio capture.

        Returns:
            The new recording state (True=recording, False=paused),
            or None if toggle failed.
        """
        _logger = logging.getLogger("realtime-transcriber")

        if self._recording:
            # Pause: stop audio capture, keep processor running
            # Use permanent=True to prevent auto-recovery from overriding
            self._recording = False

            if self._mic_capture and self._mic_capture.is_running:
                self._mic_capture.stop(permanent=True)

            if self._system_capture and self._system_capture.is_running:
                self._system_capture.stop(permanent=True)

            _logger.info("Recording paused")
            return False
        else:
            # Resume: restart audio capture
            self._recording = True
            success = True
            loop = asyncio.get_running_loop()

            if self._mic_capture is not None:
                try:
                    self._mic_capture.start(loop=loop)
                except Exception as exc:
                    _logger.error("Failed to resume mic capture: %s", exc)
                    success = False

            if self._system_capture is not None:
                try:
                    self._system_capture.start(loop=loop)
                except Exception as exc:
                    _logger.error("Failed to resume system capture: %s", exc)
                    success = False

            if success:
                _logger.info("Recording resumed")
            return success or None

    def _get_status(self) -> str:
        """Get current application status for display.

        Returns:
            Formatted status string.
        """
        elapsed = time.time() - self._start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        secs = int(elapsed % 60)

        recording_state = "🎤 RECORDING" if self._recording else "⏸  PAUSED"

        lines = [
            "",
            "┌─────────────────────────────────────────┐",
            f"│  Status:  {recording_state:<30} │",
            f"│  Uptime:  {hours:02d}:{minutes:02d}:{secs:02d}                           │",
        ]

        if self._processor is not None:
            m = self._processor.metrics
            summary = m.get_summary()
            lines.extend([
                f"│  Chunks:  {summary['chunks_processed']:>5d} processed, "
                f"{summary['chunks_dropped']:>4d} dropped      │",
                f"│  Latency: {summary['avg_latency_ms']:>6.1f}ms avg, "
                f"{summary['realtime_factor']:.3f} RTF         │",
                f"│  Queue:   {summary['avg_queue_size']:>5.1f} avg size               │",
            ])

        if self._mic_capture is not None:
            dev = self._mic_capture.get_device_index()
            recovering = (
                " (recovering)" if self._mic_capture.is_recovering else ""
            )
            lines.append(
                f"│  Mic:     device {dev}{recovering:<20} │"
            )

        if self._system_capture is not None:
            dev = self._system_capture.get_device_index()
            lines.append(
                f"│  SysAudio: device {dev:<12} │"
            )

        # Show per-source silence skipped counts
        mic_skipped = (
            self._mic_capture.silent_chunks_skipped
            if self._mic_capture is not None
            else 0
        )
        sys_skipped = (
            self._system_capture.silent_chunks_skipped
            if self._system_capture is not None
            else 0
        )
        total_skipped = mic_skipped + sys_skipped

        if total_skipped > 0:
            chunk_s = self._config.audio.chunk_duration
            parts: list[str] = []
            if mic_skipped > 0:
                parts.append(
                    f"mic: {mic_skipped} ({mic_skipped * chunk_s:.0f}s)"
                )
            if sys_skipped > 0:
                parts.append(
                    f"sys: {sys_skipped} ({sys_skipped * chunk_s:.0f}s)"
                )
            lines.append(
                f"│  Silent:  {', '.join(parts)}        │"
            )

        # Show recalibration info if a silence detector is active
        if (self._mic_capture is not None
                and self._mic_capture._silence_detector is not None):
            det = self._mic_capture._silence_detector
            if det.recalibration_count > 0:
                lines.append(
                    f"│  Recalib: {det.recalibration_count:>5d} times, "
                    f"threshold now {det.current_threshold_db:.0f} dB      │"
                )

        lines.append("└─────────────────────────────────────────┘")
        return "\n".join(lines)


def setup_signal_handlers(app: Application) -> None:
    """Register signal handlers for graceful shutdown.

    Args:
        app: The running application instance.
    """
    loop = asyncio.get_running_loop()

    def handle_signal() -> None:
        logger = logging.getLogger("realtime-transcriber")
        logger.info("Received shutdown signal")
        app.signal_shutdown()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_signal)
        except NotImplementedError:
            # Windows: use signal.signal
            signal.signal(sig, lambda *_: handle_signal())


async def main() -> int:
    """Application entry point."""
    args = parse_args()

    # List devices mode
    if args.list_devices:
        print_available_devices()
        return 0

    # Load configuration
    config = load_config()

    # Override config with CLI args
    config = AppConfig(
        audio=config.audio.__class__(
            mic_device_index=args.device or config.audio.mic_device_index,
            system_audio_device_index=args.system_audio
            or config.audio.system_audio_device_index,
            sample_rate=config.audio.sample_rate,
            chunk_duration=config.audio.chunk_duration,
            channels=config.audio.channels,
        ),
        ring_buffer=config.ring_buffer,
        whisper=config.whisper.__class__(
            model_size=args.model or config.whisper.model_size,
            device=config.whisper.device,
            compute_type=config.whisper.compute_type,
            beam_size=config.whisper.beam_size,
            language=args.language or config.whisper.language,
            vad_threshold=config.whisper.vad_threshold,
            min_speech_duration=config.whisper.min_speech_duration,
            silence_duration=config.whisper.silence_duration,
        ),
        queue=config.queue,
        output=config.output,
        metrics=config.metrics,
        checkpoint=config.checkpoint,
        log=config.log,
        audio_archive=config.audio_archive,
    )

    # Setup logging
    setup_logging(level=config.log.level, log_file=config.log.log_file)

    # Create and run application
    app = Application(config)
    setup_signal_handlers(app)
    return await app.run()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
