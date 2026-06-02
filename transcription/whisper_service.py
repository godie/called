"""Transcription module using faster-whisper for speech-to-text."""

import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment, TranscriptionInfo

from audio.resampler import resample_to_whisper_format

logger = logging.getLogger("realtime-transcriber.whisper")


@dataclass
class TranscriptionResult:
    """Result of a transcription operation."""

    text: str
    segments: list["TranscribedSegment"] = field(default_factory=list)
    language: str = ""
    language_probability: float = 0.0
    processing_time: float = 0.0
    input_duration: float = 0.0
    error: str | None = None  # Set when transcription fails


@dataclass
class TranscribedSegment:
    """A single transcribed segment with timing."""

    start: float
    end: float
    text: str


class WhisperService:
    """Wrapper around faster-whisper for transcription.

    Handles model loading, transcription, and language detection.
    Thread-safe for concurrent use.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "default",
        beam_size: int = 5,
        language: str | None = None,
        vad_threshold: float = 0.5,
    ) -> None:
        """Initialize the Whisper service.

        Args:
            model_size: Model size (tiny, base, small, medium, large-v3).
            device: Device to run on (auto, cpu, cuda).
            compute_type: Computation type (default, int8, float16).
            beam_size: Beam size for decoding.
            language: Force a specific language or None for auto-detection.
            vad_threshold: VAD sensitivity threshold.
        """
        self._model_size: str = model_size
        self._device: str = device
        self._compute_type: str = compute_type
        self._beam_size: int = beam_size
        self._language: str | None = language
        self._vad_threshold: float = vad_threshold

        self._model: WhisperModel | None = None
        self._lock: threading.Lock = threading.Lock()
        self._loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        """Whether the model is loaded."""
        return self._loaded

    def load_model(self) -> None:
        """Load the Whisper model.

        Downloads the model on first use if not cached.
        """
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            logger.info(
                "Loading Whisper model: %s (device=%s, compute=%s)...",
                self._model_size,
                self._device,
                self._compute_type,
            )
            try:
                self._model = WhisperModel(
                    self._model_size,
                    device=self._device,
                    compute_type=self._compute_type,
                )
                self._loaded = True
                logger.info("Whisper model loaded successfully")
            except Exception as exc:
                logger.error("Failed to load Whisper model: %s", exc)
                raise

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> TranscriptionResult:
        """Transcribe audio data to text.

        Args:
            audio: Audio samples as numpy array (float32, shape: [samples, 1]).
            sample_rate: Sample rate of the audio data.

        Returns:
            TranscriptionResult with the transcribed text and metadata.

        Raises:
            RuntimeError: If the model is not loaded.
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("Whisper model not loaded. Call load_model() first.")

        start_time = time.perf_counter()

        # Ensure audio is 16 kHz mono float32 (clean input for Whisper)
        audio_1d = resample_to_whisper_format(
            audio,
            input_sample_rate=sample_rate,
            input_channels=1,
        )

        input_duration = len(audio_1d) / sample_rate if sample_rate > 0 else 0.0

        try:
            segments: list[Segment]
            info: TranscriptionInfo
            segments, info = self._model.transcribe(
                audio_1d,
                beam_size=self._beam_size,
                language=self._language,
                vad_filter=True,
                vad_parameters=dict(
                    threshold=self._vad_threshold,
                    min_speech_duration_ms=250,
                    min_silence_duration_ms=100,
                ),
            )

            # Convert segments and join text
            transcribed_segments: list[TranscribedSegment] = []
            texts: list[str] = []
            for seg in segments:
                transcribed_segments.append(
                    TranscribedSegment(
                        start=seg.start,
                        end=seg.end,
                        text=seg.text.strip(),
                    )
                )
                texts.append(seg.text.strip())

            processing_time = time.perf_counter() - start_time
            full_text = " ".join(texts)

            return TranscriptionResult(
                text=full_text,
                segments=transcribed_segments,
                language=info.language,
                language_probability=info.language_probability,
                processing_time=processing_time,
                input_duration=input_duration,
            )

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.error("Transcription failed: %s", error_msg, exc_info=True)
            processing_time = time.perf_counter() - start_time
            return TranscriptionResult(
                text="",
                segments=[],
                language="",
                language_probability=0.0,
                processing_time=processing_time,
                input_duration=input_duration,
                error=error_msg,
            )

    def detect_language(self, audio: np.ndarray) -> tuple[str, float]:
        """Detect the language of audio data without full transcription.

        Args:
            audio: Audio samples as numpy array.

        Returns:
            Tuple of (language_code, probability).
        """
        result = self.transcribe(audio, sample_rate=16000)
        return result.language, result.language_probability
