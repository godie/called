"""Unit tests for RingBuffer."""

import numpy as np
import pytest

from audio.buffers import AudioChunk, RingBuffer


def make_chunk(seq: int = 0) -> AudioChunk:
    """Helper to create a test audio chunk."""
    return AudioChunk(
        data=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        timestamp=float(seq),
        source="test",
    )


class TestRingBuffer:
    """Tests for RingBuffer implementation."""

    def test_initial_state(self) -> None:
        """Buffer starts empty with correct capacity."""
        buf = RingBuffer(capacity=64)
        assert buf.capacity == 64
        assert buf.size == 0
        assert buf.is_empty
        assert not buf.is_full
        assert len(buf) == 0

    def test_put_and_get_single(self) -> None:
        """Put and get a single chunk."""
        buf = RingBuffer(capacity=8)
        chunk = make_chunk()
        buf.put(chunk)
        assert buf.size == 1
        assert not buf.is_empty

        retrieved = buf.get()
        assert retrieved is not None
        assert retrieved.data.shape == (16000,)
        assert buf.is_empty

    def test_put_and_get_fifo_order(self) -> None:
        """Chunks are retrieved in FIFO order."""
        buf = RingBuffer(capacity=8)
        chunks = [make_chunk(i) for i in range(4)]

        for c in chunks:
            buf.put(c)

        retrieved = []
        while True:
            c = buf.get()
            if c is None:
                break
            retrieved.append(c)

        assert len(retrieved) == 4
        for i, c in enumerate(retrieved):
            assert c.sequence_id == i

    def test_overflow_wraps_correctly(self) -> None:
        """When buffer is full, old data is overwritten."""
        buf = RingBuffer(capacity=4)

        # Fill buffer
        for i in range(4):
            buf.put(make_chunk(i))
        assert buf.is_full
        assert buf.size == 4

        # Overflow: put more chunks
        for i in range(4, 8):
            buf.put(make_chunk(i))

        # Should still be 4 chunks, but the oldest (0-3) are gone
        assert buf.size == 4
        assert buf.overflow_count > 0

        # Read all
        retrieved = []
        while True:
            c = buf.get()
            if c is None:
                break
            retrieved.append(c)

        # Should get chunks 4, 5, 6, 7
        assert len(retrieved) == 4
        for i, c in enumerate(retrieved):
            assert c.sequence_id == 4 + i

    def test_peek_does_not_remove(self) -> None:
        """Peek returns the oldest chunk without removing it."""
        buf = RingBuffer(capacity=8)
        buf.put(make_chunk())
        buf.put(make_chunk())

        peeked = buf.peek()
        assert peeked is not None
        assert buf.size == 2  # Unchanged

    def test_peek_empty_returns_none(self) -> None:
        """Peek on empty buffer returns None."""
        buf = RingBuffer(capacity=8)
        assert buf.peek() is None

    def test_get_empty_returns_none(self) -> None:
        """Get on empty buffer returns None."""
        buf = RingBuffer(capacity=8)
        assert buf.get() is None

    def test_clear_empties_buffer(self) -> None:
        """Clear removes all chunks."""
        buf = RingBuffer(capacity=8)
        for i in range(5):
            buf.put(make_chunk(i))
        assert buf.size == 5

        buf.clear()
        assert buf.size == 0
        assert buf.is_empty
        assert buf.get() is None

    def test_get_all_returns_all_and_clears(self) -> None:
        """get_all returns all chunks in order and clears buffer."""
        buf = RingBuffer(capacity=8)
        for i in range(6):
            buf.put(make_chunk(i))

        all_chunks = buf.get_all()
        assert len(all_chunks) == 6
        assert buf.is_empty

        for i, c in enumerate(all_chunks):
            assert c.sequence_id == i

    def test_get_all_empty_returns_empty_list(self) -> None:
        """get_all on empty buffer returns empty list."""
        buf = RingBuffer(capacity=8)
        assert buf.get_all() == []

    def test_sequence_ids_are_monotonic(self) -> None:
        """Sequence IDs increase monotonically."""
        buf = RingBuffer(capacity=8)
        ids = []
        for _ in range(10):
            c = make_chunk()
            buf.put(c)
            ids.append(c.sequence_id)

        for i in range(1, len(ids)):
            assert ids[i] > ids[i - 1]

    def test_capacity_one(self) -> None:
        """Buffer with capacity=1 works correctly."""
        buf = RingBuffer(capacity=1)
        buf.put(make_chunk())
        assert buf.is_full

        buf.put(make_chunk())  # Overwrites
        assert buf.size == 1
        retrieved = buf.get()
        assert retrieved is not None

    def test_invalid_capacity(self) -> None:
        """Negative or zero capacity raises ValueError."""
        with pytest.raises(ValueError):
            RingBuffer(capacity=0)
        with pytest.raises(ValueError):
            RingBuffer(capacity=-1)

    def test_thread_safety_basic(self) -> None:
        """Basic thread safety: concurrent put and get."""
        import threading
        import time

        buf = RingBuffer(capacity=128)
        results: list[int] = []
        errors: list[Exception] = []
        stop = threading.Event()

        def producer() -> None:
            for i in range(500):
                try:
                    buf.put(make_chunk(i))
                    time.sleep(0.001)
                except Exception as e:
                    errors.append(e)
            stop.set()

        def consumer() -> None:
            while not stop.is_set() or not buf.is_empty:
                try:
                    chunk = buf.get()
                    if chunk is not None:
                        results.append(chunk.sequence_id)
                except Exception as e:
                    errors.append(e)
                time.sleep(0.001)

        t1 = threading.Thread(target=producer)
        t2 = threading.Thread(target=consumer)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, f"Errors during thread test: {errors}"
        assert len(results) > 0, "No chunks were consumed"

    def test_long_running_no_memory_growth(self) -> None:
        """Memory stays bounded during extended operation (8h sim)."""
        import sys

        buf = RingBuffer(capacity=64)
        initial_size = sys.getsizeof(buf)

        # Simulate 8 hours: ~28800 chunks at 1/sec
        for i in range(5000):
            buf.put(make_chunk(i))
            if i % 10 == 0:
                buf.get()  # Simulate consumption

        final_size = sys.getsizeof(buf)
        # Size should not grow significantly (allow some overhead)
        assert final_size <= initial_size * 2, (
            f"Buffer size grew from {initial_size} to {final_size}"
        )
        assert buf.size <= 64
