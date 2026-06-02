"""Unit tests for timing utilities."""

from utils.timers import format_duration, get_timestamp


class TestTimers:
    """Tests for timing utilities."""

    def test_get_timestamp_returns_string(self) -> None:
        """get_timestamp returns a non-empty string."""
        ts = get_timestamp()
        assert isinstance(ts, str)
        assert len(ts) > 0
        assert "T" in ts  # ISO-like separator

    def test_format_duration_zero(self) -> None:
        """Zero seconds formatted correctly."""
        assert format_duration(0.0) == "00:00:00.000"

    def test_format_duration_seconds(self) -> None:
        """Seconds formatted correctly."""
        assert format_duration(5.5) == "00:00:05.500"

    def test_format_duration_minutes(self) -> None:
        """Minutes formatted correctly."""
        assert format_duration(125.0) == "00:02:05.000"

    def test_format_duration_hours(self) -> None:
        """Hours formatted correctly."""
        assert format_duration(3661.0) == "01:01:01.000"

    def test_format_duration_8_hours(self) -> None:
        """8+ hours formatted correctly."""
        result = format_duration(28800.0)  # 8 hours
        assert result.startswith("08:00:00")
