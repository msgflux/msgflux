"""Tests for MCP exceptions."""

import pytest

from msgflux.protocols.mcp.exceptions import (
    MCPConnectionError,
    MCPError,
    MCPTimeoutError,
    MCPToolError,
)


class TestMCPError:
    """Tests for base MCPError exception."""

    def test_raise_base_exception(self):
        """Test raising base MCP exception."""
        with pytest.raises(MCPError) as exc_info:
            raise MCPError("Base error message")
        assert str(exc_info.value) == "Base error message"


class TestMCPTimeoutError:
    """Tests for MCPTimeoutError exception."""

    def test_raise_timeout_error(self):
        """Test raising timeout error."""
        with pytest.raises(MCPTimeoutError) as exc_info:
            raise MCPTimeoutError("Operation timed out after 30s")
        assert "timed out" in str(exc_info.value)


class TestMCPToolError:
    """Tests for MCPToolError exception."""

    def test_raise_tool_error(self):
        """Test raising tool error."""
        with pytest.raises(MCPToolError) as exc_info:
            raise MCPToolError("Tool execution failed")
        assert "Tool execution failed" in str(exc_info.value)


class TestMCPConnectionError:
    """Tests for MCPConnectionError exception."""

    def test_raise_connection_error(self):
        """Test raising connection error."""
        with pytest.raises(MCPConnectionError) as exc_info:
            raise MCPConnectionError("Failed to connect to server")
        assert "Failed to connect" in str(exc_info.value)


def test_mcp_error_subclasses_inherit_from_mcp_error():
    """Keep the public MCP error hierarchy consistent."""
    for error_type in (MCPTimeoutError, MCPToolError, MCPConnectionError):
        assert issubclass(error_type, MCPError)
