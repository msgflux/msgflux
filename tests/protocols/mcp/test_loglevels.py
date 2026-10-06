"""Tests for MCP log levels."""

import pytest

from msgflux.protocols.mcp.loglevels import LogLevel


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_loglevel_all_values(self):
        """Test that all log levels are defined."""
        expected_levels = {
            "debug",
            "info",
            "notice",
            "warning",
            "error",
            "critical",
            "alert",
            "emergency",
        }
        actual_levels = {level.value for level in LogLevel}
        assert actual_levels == expected_levels

    def test_loglevel_is_enum(self):
        """Test that LogLevel is an enum."""
        from enum import Enum

        assert issubclass(LogLevel, Enum)
