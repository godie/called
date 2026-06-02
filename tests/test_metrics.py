"""Unit tests for ProcessingMetrics."""

import time

from transcription.processor import ProcessingMetrics


class TestProcessingMetrics:
    """Tests for ProcessingMetrics collection."""

    def test_initial_state(self) -> None:
        """Metrics start at zero."""
        m = ProcessingMetrics()
        assert m.chunks_processed == 0
        assert m.chunks_dropped == 0
        assert m.total_processing_time == 0.0
        assert m.total_audio_duration == 0.0
        assert m.min_latency == float("inf")
        assert m.max_latency == 0.0
        assert m.avg_latency == 0.0
        assert m.latency_samples == []
        assert m.queue_size_samples == []

    def test_record_chunk_updates_metrics(self) -> None:
        """Recording a chunk updates all relevant metrics."""
        m = ProcessingMetrics()
        m.record_chunk(
            processing_time=0.5,
            audio_duration=2.0,
            queue_size=3,
        )

        assert m.chunks_processed == 1
        assert m.total_processing_time == 0.5
        assert m.total_audio_duration == 2.0
        assert m.min_latency == 0.5
        assert m.max_latency == 0.5
        assert m.avg_latency == 0.5
        assert m.latency_samples == [0.5]
        assert m.queue_size_samples == [3]

    def test_record_multiple_chunks(self) -> None:
        """Multiple chunks aggregate correctly."""
        m = ProcessingMetrics()

        m.record_chunk(processing_time=0.2, audio_duration=1.0, queue_size=5)
        m.record_chunk(processing_time=1.0, audio_duration=2.0, queue_size=2)
        m.record_chunk(processing_time=0.3, audio_duration=1.5, queue_size=8)

        assert m.chunks_processed == 3
        assert m.total_processing_time == 1.5
        assert m.total_audio_duration == 4.5
        assert m.min_latency == 0.2
        assert m.max_latency == 1.0
        assert abs(m.avg_latency - 0.5) < 1e-10
        assert m.latency_samples == [0.2, 1.0, 0.3]
        assert m.queue_size_samples == [5, 2, 8]

    def test_record_dropped(self) -> None:
        """Recording a dropped chunk increments counter."""
        m = ProcessingMetrics()
        m.record_dropped()
        m.record_dropped()
        assert m.chunks_dropped == 2
        assert m.chunks_processed == 0

    def test_get_summary(self) -> None:
        """get_summary returns correct aggregated values."""
        m = ProcessingMetrics()
        m.record_chunk(processing_time=0.1, audio_duration=1.0, queue_size=3)
        m.record_chunk(processing_time=0.3, audio_duration=1.0, queue_size=7)
        m.record_dropped()

        summary = m.get_summary()
        assert summary["chunks_processed"] == 2
        assert summary["chunks_dropped"] == 1
        assert summary["min_latency_ms"] == 100.0
        assert summary["max_latency_ms"] == 300.0
        assert summary["avg_latency_ms"] == 200.0
        assert summary["avg_queue_size"] == 5.0
        assert summary["realtime_factor"] == 0.2
        assert summary["total_audio_s"] == 2.0
        assert summary["total_processing_s"] == 0.4

    def test_latency_samples_capped_at_1000(self) -> None:
        """Latency samples are capped to prevent memory growth."""
        m = ProcessingMetrics()
        for i in range(1500):
            m.record_chunk(processing_time=0.1, audio_duration=1.0, queue_size=1)

        assert len(m.latency_samples) <= 1000
        assert len(m.queue_size_samples) <= 1000

    def test_empty_metrics_summary_no_div_by_zero(self) -> None:
        """Summary on empty metrics does not divide by zero."""
        m = ProcessingMetrics()
        summary = m.get_summary()
        assert summary["avg_latency_ms"] == 0.0
        assert summary["avg_queue_size"] == 0.0
        assert summary["realtime_factor"] == 0.0

    def test_min_latency_starts_at_inf(self) -> None:
        """Before any chunks, min latency is inf."""
        m = ProcessingMetrics()
        assert m.min_latency == float("inf")
