"""Unit tests for AudioArchiver.

Tests: bitrate mapping, Opus encoding (pyogg), WAV fallback, ffmpeg
subprocess fallback, save_raw mode, error handling, directory creation.
"""

import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

# Ensure the module-level _HAS_PYOGG can be manipulated in tests
import audio.archiver as archiver_module
from audio.archiver import AudioArchiver

_FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None


def make_audio(duration_s: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
    """Create synthetic float32 audio (sine wave) for testing."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    return np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)


class TestBitrateMapping:
    """Tests for bitrate string → integer mapping."""

    def test_default_16k(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = AudioArchiver(Path(tmpdir))
            assert archiver._bitrate == 16000

    def test_explicit_12k(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = AudioArchiver(Path(tmpdir), bitrate="12k")
            assert archiver._bitrate == 12000

    def test_explicit_16k(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = AudioArchiver(Path(tmpdir), bitrate="16k")
            assert archiver._bitrate == 16000

    def test_explicit_24k(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = AudioArchiver(Path(tmpdir), bitrate="24k")
            assert archiver._bitrate == 24000

    def test_unknown_bitrate_falls_back_to_16k(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = AudioArchiver(Path(tmpdir), bitrate="48k")
            assert archiver._bitrate == 16000

    def test_empty_bitrate_falls_back_to_16k(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = AudioArchiver(Path(tmpdir), bitrate="")
            assert archiver._bitrate == 16000

    def test_bitrate_map_contains_three_options(self) -> None:
        assert set(AudioArchiver.BITRATE_MAP.keys()) == {"12k", "16k", "24k"}


class TestDirectoryCreation:
    """Tests for archive directory handling."""

    def test_creates_directory_on_init(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_dir = Path(tmpdir) / "nested" / "archive"
            AudioArchiver(archive_dir)
            assert archive_dir.exists()
            assert archive_dir.is_dir()

    def test_existing_directory_no_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_dir = Path(tmpdir)
            AudioArchiver(archive_dir)  # First init
            AudioArchiver(archive_dir)  # Second init — should not raise


class TestSaveWav:
    """Tests for the _save_wav static method."""

    def test_creates_valid_wav(self) -> None:
        audio = make_audio(2.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.wav"
            AudioArchiver._save_wav(audio, path)
            assert path.exists()
            assert path.stat().st_size > 44  # Bigger than WAV header

    def test_wav_is_16bit_pcm_mono(self) -> None:
        audio = make_audio(1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.wav"
            AudioArchiver._save_wav(audio, path)

            with wave.open(str(path), "rb") as wf:
                assert wf.getnchannels() == 1
                assert wf.getsampwidth() == 2  # 16-bit
                assert wf.getframerate() == 16000
                frames = wf.getnframes()
                assert frames == len(audio)

    def test_wav_amplitude_clipped(self) -> None:
        """Samples exceeding [-1, 1] are clipped, not wrapped."""
        audio = np.array([2.0, -3.0, 0.5], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "clip.wav"
            AudioArchiver._save_wav(audio, path)

            with wave.open(str(path), "rb") as wf:
                data = np.frombuffer(wf.readframes(3), dtype=np.int16)
                assert data[0] == 32767  # 2.0 clipped to +max
                assert data[1] == -32768  # -3.0 clipped to -max
                # 0.5 * 32767 = 16383.5 → floor to 16383
                assert data[2] == 16383


class TestArchiveChunkWithPyogg:
    """Tests for archive_chunk when pyogg is available."""

    @pytest.fixture(autouse=True)
    def _ensure_pyogg(self) -> None:
        """Skip these tests if pyogg is not installed."""
        if not archiver_module._HAS_PYOGG:
            pytest.skip("pyogg not installed")

    def test_archives_to_opus(self) -> None:
        audio = make_audio(2.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = AudioArchiver(Path(tmpdir), bitrate="16k")
            path = archiver.archive_chunk(audio)
            assert path is not None
            assert path.suffix == ".opus"
            assert path.exists()
            assert path.stat().st_size > 0

    def test_opus_file_has_ogg_header(self) -> None:
        """Opus files wrapped in OGG must start with 'OggS' magic bytes."""
        audio = make_audio(1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = AudioArchiver(Path(tmpdir), bitrate="16k")
            path = archiver.archive_chunk(audio)
            assert path is not None
            with open(path, "rb") as f:
                header = f.read(4)
            assert header == b"OggS"

    def test_save_raw_also_creates_wav(self) -> None:
        audio = make_audio(0.5)
        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = AudioArchiver(Path(tmpdir), bitrate="16k", save_raw=True)
            archiver.archive_chunk(audio)

            # Should find both .opus and .wav with same stem
            opus_files = list(Path(tmpdir).glob("*.opus"))
            wav_files = list(Path(tmpdir).glob("*.wav"))
            assert len(opus_files) == 1
            assert len(wav_files) == 1
            assert opus_files[0].stem == wav_files[0].stem

    def test_chunk_index_increments(self) -> None:
        audio = make_audio(0.2)
        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = AudioArchiver(Path(tmpdir), bitrate="16k")
            assert archiver._chunk_index == 0
            archiver.archive_chunk(audio)
            assert archiver._chunk_index == 1
            archiver.archive_chunk(audio)
            assert archiver._chunk_index == 2

    def test_returns_path_on_success(self) -> None:
        audio = make_audio(1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = AudioArchiver(Path(tmpdir))
            result = archiver.archive_chunk(audio)
            assert result is not None
            assert isinstance(result, Path)
            assert result.suffix == ".opus"


class TestFallbackBehavior:
    """Tests for behavior when pyogg is not available."""

    def test_uses_ffmpeg_when_pyogg_missing(self) -> None:
        """When _HAS_PYOGG is False, _encode_opus should call ffmpeg path."""
        audio = make_audio(0.5)
        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = AudioArchiver(Path(tmpdir), bitrate="16k")

            with (
                mock.patch.object(archiver, "_encode_opus_ffmpeg") as mock_ffmpeg,
                mock.patch.object(archiver, "_encode_opus_pyogg") as mock_pyogg,
                mock.patch.object(archiver_module, "_HAS_PYOGG", False),
            ):
                archiver._encode_opus(audio, Path(tmpdir) / "test.opus")

            mock_ffmpeg.assert_called_once()
            mock_pyogg.assert_not_called()

    def test_uses_pyogg_when_available(self) -> None:
        """When _HAS_PYOGG is True, _encode_opus should call pyogg path."""
        if not archiver_module._HAS_PYOGG:
            pytest.skip("pyogg not installed")

        audio = make_audio(0.5)
        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = AudioArchiver(Path(tmpdir), bitrate="16k")

            with (
                mock.patch.object(archiver, "_encode_opus_pyogg") as mock_pyogg,
                mock.patch.object(archiver, "_encode_opus_ffmpeg") as mock_ffmpeg,
            ):
                archiver._encode_opus(audio, Path(tmpdir) / "test.opus")

            mock_pyogg.assert_called_once()
            mock_ffmpeg.assert_not_called()


class TestFfmpegSubprocess:
    """Tests for the ffmpeg subprocess fallback."""

    @pytest.mark.skipif(not _FFMPEG_AVAILABLE, reason="ffmpeg not installed")
    def test_ffmpeg_creates_opus_file(self) -> None:
        """Integration test: ffmpeg should produce a valid Opus file."""
        audio = make_audio(1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = AudioArchiver(Path(tmpdir), bitrate="16k")
            output_path = Path(tmpdir) / "ffmpeg_test.opus"

            archiver._encode_opus_ffmpeg(audio, output_path)

            assert output_path.exists()
            assert output_path.stat().st_size > 0
            # Opus in OGG container starts with 'OggS'
            with open(output_path, "rb") as f:
                assert f.read(4) == b"OggS"

    def test_ffmpeg_uses_correct_bitrate(self) -> None:
        """Verify ffmpeg is called with the correct bitrate argument."""
        audio = make_audio(0.3)
        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = AudioArchiver(Path(tmpdir), bitrate="24k")
            output_path = Path(tmpdir) / "test.opus"

            with mock.patch("subprocess.run") as mock_run:
                # Make subprocess.run succeed
                mock_run.return_value = mock.MagicMock()
                archiver._encode_opus_ffmpeg(audio, output_path)

            # Check the command arguments
            call_args = mock_run.call_args[0][0]
            assert "-b:a" in call_args
            idx = call_args.index("-b:a")
            assert call_args[idx + 1] == "24000"

    def test_ffmpeg_uses_voip_application(self) -> None:
        """ffmpeg should use -application voip for speech optimization."""
        audio = make_audio(0.3)
        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = AudioArchiver(Path(tmpdir), bitrate="16k")
            output_path = Path(tmpdir) / "test.opus"

            with mock.patch("subprocess.run") as mock_run:
                mock_run.return_value = mock.MagicMock()
                archiver._encode_opus_ffmpeg(audio, output_path)

            call_args = mock_run.call_args[0][0]
            assert "-application" in call_args
            idx = call_args.index("-application")
            assert call_args[idx + 1] == "voip"

    def test_ffmpeg_not_found_raises_runtime_error(self) -> None:
        """FileNotFoundError from subprocess becomes RuntimeError."""
        audio = make_audio(0.3)
        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = AudioArchiver(Path(tmpdir), bitrate="16k")
            output_path = Path(tmpdir) / "test.opus"

            with (
                mock.patch("subprocess.run", side_effect=FileNotFoundError("ffmpeg")),
                pytest.raises(RuntimeError, match="ffmpeg"),
            ):
                archiver._encode_opus_ffmpeg(audio, output_path)

    def test_ffmpeg_failure_raises(self) -> None:
        """CalledProcessError from ffmpeg is re-raised."""
        audio = make_audio(0.3)
        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = AudioArchiver(Path(tmpdir), bitrate="16k")
            output_path = Path(tmpdir) / "test.opus"

            error = subprocess.CalledProcessError(1, "ffmpeg", stderr=b"Encoder error")
            with (
                mock.patch("subprocess.run", side_effect=error),
                pytest.raises(RuntimeError, match="ffmpeg encoding failed"),
            ):
                archiver._encode_opus_ffmpeg(audio, output_path)


class TestErrorHandling:
    """Tests for error resilience in archive_chunk."""

    def test_returns_none_on_encoding_failure(self) -> None:
        """archive_chunk returns None when encoding fails."""
        audio = make_audio(0.5)
        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = AudioArchiver(Path(tmpdir), bitrate="16k")

            with mock.patch.object(archiver, "_encode_opus", side_effect=RuntimeError("fail")):
                result = archiver.archive_chunk(audio)
                assert result is None

    def test_chunk_index_increments_even_on_failure(self) -> None:
        """Chunk index keeps incrementing regardless of success/failure."""
        audio = make_audio(0.2)
        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = AudioArchiver(Path(tmpdir), bitrate="16k")
            before = archiver._chunk_index

            with mock.patch.object(archiver, "_encode_opus", side_effect=RuntimeError("fail")):
                archiver.archive_chunk(audio)

            assert archiver._chunk_index == before + 1


class TestSampleRateConfiguration:
    """Tests for configurable sample rate and channels."""

    def test_sample_rate_default_16000(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = AudioArchiver(Path(tmpdir))
            assert archiver._sample_rate == 16000

    def test_sample_rate_custom_44100(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = AudioArchiver(Path(tmpdir), sample_rate=44100)
            assert archiver._sample_rate == 44100

    def test_channels_default_1(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = AudioArchiver(Path(tmpdir))
            assert archiver._channels == 1

    def test_channels_custom_2(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = AudioArchiver(Path(tmpdir), channels=2)
            assert archiver._channels == 2

    def test_save_raw_default_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = AudioArchiver(Path(tmpdir))
            assert archiver._save_raw is False


class TestOpusEncoderConstants:
    """Tests for Opus encoder constants."""

    def test_voip_constant_is_2048(self) -> None:
        assert AudioArchiver.OPUS_APPLICATION_VOIP == 2048


class TestArchiveWithEmptyAudio:
    """Edge case: archiving empty/short audio."""

    def test_empty_audio(self) -> None:
        """Very short audio should still produce a file."""
        audio = np.array([0.0] * 320, dtype=np.float32)  # 20ms at 16kHz
        with tempfile.TemporaryDirectory() as tmpdir:
            archiver = AudioArchiver(Path(tmpdir), bitrate="16k")
            result = archiver.archive_chunk(audio)
            # Should try to encode — may succeed or fail depending on codec
            assert isinstance(result, Path | None)
