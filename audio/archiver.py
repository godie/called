"""Audio archiving with Opus/OGG compression.

Saves compressed audio chunks to .opus files for long-term storage.
Uses the `ogg` module from pyogg for Opus encoding. Falls back to
storing raw audio if pyogg is unavailable.

Target bitrate: 12-24 kbps for speech (default 16 kbps).
"""

import logging
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("realtime-transcriber.archiver")

# Try to import pyogg for Opus encoding
try:
    import ogg
    from pyogg import OpusEncoder  # type: ignore[import-untyped]

    _HAS_PYOGG = True
except ImportError:
    _HAS_PYOGG = False
    logger.debug("pyogg not available; will save WAV instead of Opus")


class AudioArchiver:
    """Archives audio chunks with Opus/OGG compression.

    Compresses audio to Opus at speech-optimized bitrates (12-24 kbps)
    for efficient long-term storage. Files are saved per-chunk and can
    be concatenated later.
    """

    # Opus encoder constants
    OPUS_APPLICATION_VOIP = 2048  # Optimized for speech

    BITRATE_MAP: dict[str, int] = {
        "12k": 12000,
        "16k": 16000,
        "24k": 24000,
    }

    def __init__(
        self,
        archive_dir: Path,
        sample_rate: int = 16000,
        channels: int = 1,
        bitrate: str = "16k",
        save_raw: bool = False,
    ) -> None:
        """Initialize the audio archiver.

        Args:
            archive_dir: Directory to save compressed archives.
            sample_rate: Audio sample rate in Hz.
            channels: Number of audio channels.
            bitrate: Target bitrate (\"12k\", \"16k\", \"24k\").
            save_raw: If True, also save raw WAV alongside compressed.
        """
        self._archive_dir: Path = archive_dir
        self._sample_rate: int = sample_rate
        self._channels: int = channels
        self._bitrate: int = self.BITRATE_MAP.get(bitrate, 16000)
        self._save_raw: bool = save_raw

        self._archive_dir.mkdir(parents=True, exist_ok=True)
        self._chunk_index: int = 0

    def archive_chunk(self, audio: np.ndarray) -> Path | None:
        """Compress and save an audio chunk to disk.

        Args:
            audio: Audio samples as float32 numpy array.

        Returns:
            Path to the saved file (compressed), or None on failure.
        """
        ts = time.strftime("%Y%m%d_%H%M%S")
        seq = self._chunk_index
        self._chunk_index += 1

        filename = f"audio_{ts}_{seq:06d}"

        try:
            compressed_path = self._archive_dir / f"{filename}.opus"
            self._encode_opus(audio, compressed_path)

            if self._save_raw:
                raw_path = self._archive_dir / f"{filename}.wav"
                self._save_wav(audio, raw_path)

            return compressed_path

        except Exception as exc:
            logger.error("Failed to archive audio chunk %d: %s", seq, exc)
            return None

    def _encode_opus(self, audio: np.ndarray, output_path: Path) -> None:
        """Encode audio to Opus format.

        Tries pyogg first, falls back to ffmpeg subprocess.

        Args:
            audio: Float32 audio samples.
            output_path: Path for the output .opus file.
        """
        if _HAS_PYOGG:
            self._encode_opus_pyogg(audio, output_path)
        else:
            self._encode_opus_ffmpeg(audio, output_path)

    def _encode_opus_pyogg(self, audio: np.ndarray, output_path: Path) -> None:
        """Encode using pyogg library (pure Python, no external deps).

        Args:
            audio: Float32 audio samples.
            output_path: Output .opus file path.
        """
        # Convert float32 [-1.0, 1.0] to int16
        audio_int16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)

        encoder = OpusEncoder()
        encoder.set_application(self.OPUS_APPLICATION_VOIP)
        encoder.set_sampling_frequency(self._sample_rate)
        encoder.set_channels(self._channels)
        encoder.set_bitrate(self._bitrate)

        # Encode in frames of 20ms
        frame_size = int(self._sample_rate * 0.02)  # 20ms frames

        with open(output_path, "wb") as f:
            ogg_writer = ogg.OggBasicWriter(f)

            pos = 0
            while pos < len(audio_int16):
                frame = audio_int16[pos : pos + frame_size]

                # Pad last frame with silence if needed
                if len(frame) < frame_size:
                    padded = np.zeros(frame_size, dtype=np.int16)
                    padded[: len(frame)] = frame
                    frame = padded

                encoded = encoder.encode(frame.tobytes())
                ogg_writer.write(encoded)
                pos += frame_size

        logger.debug(
            "Archived Opus chunk %d: %s (%.1f kB)",
            self._chunk_index - 1,
            output_path.name,
            output_path.stat().st_size / 1024,
        )

    def _encode_opus_ffmpeg(self, audio: np.ndarray, output_path: Path) -> None:
        """Encode using ffmpeg subprocess.

        Args:
            audio: Float32 audio samples.
            output_path: Output .opus file path.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            self._save_wav(audio, Path(tmp.name))

            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        tmp.name,
                        "-c:a",
                        "libopus",
                        "-b:a",
                        str(self._bitrate),
                        "-application",
                        "voip",
                        "-vbr",
                        "on",
                        "-frame_duration",
                        "20",
                        str(output_path),
                    ],
                    capture_output=True,
                    check=True,
                    timeout=30,
                )
                logger.debug(
                    "Archived Opus chunk %d via ffmpeg: %s",
                    self._chunk_index - 1,
                    output_path.name,
                )
            except subprocess.CalledProcessError as exc:
                logger.error(
                    "ffmpeg encoding failed: %s (stderr: %s)",
                    exc,
                    exc.stderr.decode() if exc.stderr else "",
                )
                raise RuntimeError(f"ffmpeg encoding failed: {exc}") from exc
            except FileNotFoundError:
                logger.error("ffmpeg not found. Install ffmpeg or pyogg to enable Opus archiving.")
                raise RuntimeError(
                    "Opus archiving requires ffmpeg or pyogg. "
                    "Install with: brew install ffmpeg  (macOS) "
                    "or: pip install pyogg"
                ) from None

    @staticmethod
    def _save_wav(audio: np.ndarray, output_path: Path) -> None:
        """Save audio as a simple WAV file.

        Writes a minimal 16-bit PCM WAV header followed by audio data.
        Suitable for temporary files that are deleted after transcription.

        Args:
            audio: Float32 audio samples.
            output_path: Output .wav file path.
        """
        import wave

        audio_int16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)

        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(16000)
            wf.writeframes(audio_int16.tobytes())
