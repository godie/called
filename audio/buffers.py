"""Ring buffer for lock-free audio chunk storage.

The ring buffer uses a pre-allocated list with a fixed capacity,
preventing memory growth during long-running sessions. It supports
concurrent single-producer, single-consumer access.
"""

import logging
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("realtime-transcriber.buffers")


@dataclass
class AudioChunk:
    """Represents a captured audio chunk with metadata."""

    data: np.ndarray
    sample_rate: int
    timestamp: float
    source: str  # "mic" or "system"
    sequence_id: int = 0


class RingBuffer:
    """Lock-free ring buffer for audio chunks.

    Uses a fixed-size circular buffer that overwrites old data when full.
    This prevents unbounded memory growth during long-running sessions.

    Thread-safe for single-producer, single-consumer scenarios.
    """

    def __init__(self, capacity: int = 128) -> None:
        """Initialize ring buffer with fixed capacity.

        Args:
            capacity: Maximum number of chunks to store.
        """
        if capacity < 1:
            raise ValueError(f"Capacity must be >= 1, got {capacity}")

        self._capacity: int = capacity
        self._buffer: list[AudioChunk | None] = [None] * capacity
        self._head: int = 0  # Write position
        self._tail: int = 0  # Read position
        self._size: int = 0
        self._lock: threading.Lock = threading.Lock()
        self._sequence: int = 0
        self._overflow_count: int = 0

    @property
    def capacity(self) -> int:
        """Maximum number of chunks the buffer can hold."""
        return self._capacity

    @property
    def size(self) -> int:
        """Current number of chunks in the buffer."""
        with self._lock:
            return self._size

    @property
    def is_empty(self) -> bool:
        """Whether the buffer is empty."""
        return self.size == 0

    @property
    def is_full(self) -> bool:
        """Whether the buffer is full."""
        return self.size == self._capacity

    @property
    def overflow_count(self) -> int:
        """Number of times the buffer has overflowed (data overwritten)."""
        with self._lock:
            return self._overflow_count

    def put(self, chunk: AudioChunk) -> None:
        """Add a chunk to the buffer.

        If the buffer is full, the oldest chunk is overwritten.

        Args:
            chunk: Audio chunk to add.
        """
        chunk.sequence_id = self._sequence
        self._sequence += 1

        with self._lock:
            self._buffer[self._head] = chunk
            self._head = (self._head + 1) % self._capacity
            if self._size < self._capacity:
                self._size += 1
            else:
                # Buffer full: advance tail to match (overwrite oldest)
                self._overflow_count += 1
                self._tail = (self._tail + 1) % self._capacity
                if self._overflow_count % 100 == 1:
                    logger.warning(
                        "Ring buffer overflow #%d (capacity=%d). "
                        "Oldest audio chunks are being overwritten.",
                        self._overflow_count,
                        self._capacity,
                    )

    def get(self) -> AudioChunk | None:
        """Retrieve and remove the oldest chunk from the buffer.

        Returns:
            The oldest chunk, or None if the buffer is empty.
        """
        with self._lock:
            if self._size == 0:
                return None

            chunk = self._buffer[self._tail]
            self._buffer[self._tail] = None
            self._tail = (self._tail + 1) % self._capacity
            self._size -= 1
            return chunk

    def peek(self) -> AudioChunk | None:
        """View the oldest chunk without removing it.

        Returns:
            The oldest chunk, or None if the buffer is empty.
        """
        with self._lock:
            if self._size == 0:
                return None
            return self._buffer[self._tail]

    def clear(self) -> None:
        """Remove all chunks from the buffer and reset state."""
        with self._lock:
            self._buffer = [None] * self._capacity
            self._head = 0
            self._tail = 0
            self._size = 0
            self._overflow_count = 0

    def get_all(self) -> list[AudioChunk]:
        """Retrieve all chunks in order and clear the buffer.

        Returns:
            List of all chunks in FIFO order.
        """
        with self._lock:
            if self._size == 0:
                return []

            chunks: list[AudioChunk] = []
            idx = self._tail
            for _ in range(self._size):
                chunk = self._buffer[idx]
                if chunk is not None:
                    chunks.append(chunk)
                self._buffer[idx] = None
                idx = (idx + 1) % self._capacity

            self._head = 0
            self._tail = 0
            self._size = 0
            return chunks

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"RingBuffer(capacity={self._capacity}, size={self._size})"
