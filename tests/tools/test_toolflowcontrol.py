"""Unit tests for ToolFlowControl class."""

from msgflux.tools import ToolFlowControl


class TestToolFlowControl:
    """Test suite for ToolFlowControl base class."""

    def test_toolflowcontrol_instantiation(self):
        """Test that ToolFlowControl can be instantiated."""
        control = ToolFlowControl()
        assert isinstance(control, ToolFlowControl)
