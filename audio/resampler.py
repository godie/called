"""Audio resampling utilities.

Ensures all audio is 16 kHz mono before being passed to Whisper.
Uses numpy for simple, dependency-light resampling via linear interpolation.
"""

import logging

import numpy as np

logger = logging.getLogger("realtime-transcriber.resampler")

# Whisper's expected input format
TARGET_SAMPLE_RATE: int = 16000
TARGET_CHANNELS: int = 1


def resample_to_whisper_format(
    audio: np.ndarray,
    input_sample_rate: int,
    input_channels: int = 1,
) -> np.ndarray:
    """Resample audio to 16 kHz mono (float32) for Whisper.

    Steps:
    1. Convert to mono if stereo (average channels)
    2. Resample to 16 kHz if needed using linear interpolation
    3. Cast to float32

    Args:
        audio: Input audio samples, shape (samples,) or (samples, channels).
        input_sample_rate: Sample rate of the input audio in Hz.
        input_channels: Number of channels in input audio.

    Returns:
        Resampled audio as float32 numpy array, shape (samples,).
    """
    # Ensure float32 for processing
    audio = audio.astype(np.float32, copy=False)

    # Step 1: Convert to mono
    if input_channels > 1:
        if audio.ndim == 2 and audio.shape[1] > 1:
            audio = audio.mean(axis=1)
        elif audio.ndim == 2 and audio.shape[1] == 1:
            audio = audio.flatten()

    # Ensure 1D
    if audio.ndim > 1:
        audio = audio.flatten()

    # Step 2: Resample if needed
    if input_sample_rate != TARGET_SAMPLE_RATE:
        audio = _resample_linear(audio, input_sample_rate, TARGET_SAMPLE_RATE)
        logger.debug("Resampled from %d Hz to %d Hz", input_sample_rate, TARGET_SAMPLE_RATE)

    return audio.astype(np.float32, copy=False)


def _resample_linear(
    audio: np.ndarray,
    src_rate: int,
    dst_rate: int,
) -> np.ndarray:
    """Resample audio using linear interpolation.

    Simple, fast, dependency-free resampling suitable for speech.
    For production use with high-quality requirements, consider
    replacing with scipy.signal.resample or soxr.

    Args:
        audio: 1D numpy array of audio samples.
        src_rate: Source sample rate in Hz.
        dst_rate: Destination sample rate in Hz.

    Returns:
        Resampled 1D numpy array.
    """
    if src_rate == dst_rate:
        return audio

    num_src = len(audio)
    num_dst = int(num_src * dst_rate / src_rate)

    if num_dst == 0:
        return np.array([], dtype=np.float32)

    # Source positions as float indices
    src_indices = np.arange(num_dst) * (num_src - 1) / max(num_dst - 1, 1)
    src_indices = np.clip(src_indices, 0, num_src - 1)

    # Linear interpolation
    lo = np.floor(src_indices).astype(np.int64)
    hi = np.minimum(lo + 1, num_src - 1)
    frac = src_indices - lo

    result = audio[lo] * (1.0 - frac) + audio[hi] * frac
    return result.astype(np.float32, copy=False)


def audio_chunk_duration(audio: np.ndarray, sample_rate: int) -> float:
    """Calculate duration of an audio chunk in seconds.

    Args:
        audio: Audio samples.
        sample_rate: Sample rate in Hz.

    Returns:
        Duration in seconds.
    """
    if sample_rate <= 0:
        return 0.0
    return float(len(audio) / sample_rate)
