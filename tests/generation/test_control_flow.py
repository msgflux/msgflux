"""Tests for control flow functionality."""

from typing import Any, List, Mapping
from unittest.mock import MagicMock

import pytest

from msgflux.generation.control_flow import ToolFlowControl, ToolFlowResult


class TestToolFlowResult:
    """Tests for ToolFlowResult dataclass."""

    def test_tool_flow_result_creation_complete(self):
        """Test creating a complete ToolFlowResult."""
        result = ToolFlowResult(
            is_complete=True,
            tool_calls=None,
            reasoning=None,
            final_response={"answer": "test"},
        )
        assert result.is_complete is True
        assert result.tool_calls is None
        assert result.reasoning is None
        assert result.final_response == {"answer": "test"}

    def test_tool_flow_result_creation_with_tool_calls(self):
        """Test creating a ToolFlowResult with tool calls."""
        tool_calls = [
            ("id1", "search", {"query": "test"}),
            ("id2", "calculate", {"a": 1, "b": 2}),
        ]
        result = ToolFlowResult(
            is_complete=False,
            tool_calls=tool_calls,
            reasoning="Need to search first",
            final_response=None,
        )
        assert result.is_complete is False
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0] == ("id1", "search", {"query": "test"})
        assert result.reasoning == "Need to search first"
        assert result.final_response is None


class TestToolFlowControl:
    """Tests for ToolFlowControl base class."""

    def test_tool_flow_control_can_be_inherited(self):
        """Test a concrete implementation of ToolFlowControl can be used."""

        class CustomControl(ToolFlowControl):
            @classmethod
            def extract_flow_result(
                cls, raw_response: Mapping[str, Any]
            ) -> ToolFlowResult:
                return ToolFlowResult(
                    is_complete=True,
                    tool_calls=None,
                    reasoning=None,
                    final_response=None,
                )

            @classmethod
            def inject_results(
                cls, raw_response: Mapping[str, Any], tool_results
            ) -> Mapping[str, Any]:
                return raw_response

            @classmethod
            def build_history(
                cls, raw_response: Mapping[str, Any], messages: List[Mapping[str, Any]]
            ) -> List[Mapping[str, Any]]:
                return messages

        control = CustomControl()
        result = control.extract_flow_result({})
        assert isinstance(control, ToolFlowControl)
        assert result.is_complete is True

    def test_tool_flow_control_class_attributes(self):
        """Test that ToolFlowControl has class attributes."""
        assert hasattr(ToolFlowControl, "system_prompt")
        assert hasattr(ToolFlowControl, "tools_template")
        assert ToolFlowControl.system_prompt is None
        assert ToolFlowControl.tools_template is None

    def test_tool_flow_control_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            ToolFlowControl.extract_flow_result({})

        with pytest.raises(NotImplementedError):
            ToolFlowControl.inject_results({}, MagicMock())

        with pytest.raises(NotImplementedError):
            ToolFlowControl.build_history({}, [])


class TestToolFlowControlAsync:
    """Tests for async methods of ToolFlowControl."""

    @pytest.mark.asyncio
    async def test_async_methods_default_to_sync(self):
        """Test that async methods default to calling sync versions."""

        class CustomControl(ToolFlowControl):
            sync_called = False

            @classmethod
            def extract_flow_result(
                cls, raw_response: Mapping[str, Any]
            ) -> ToolFlowResult:
                cls.sync_called = True
                return ToolFlowResult(
                    is_complete=True,
                    tool_calls=None,
                    reasoning=None,
                    final_response=None,
                )

            @classmethod
            def inject_results(
                cls, raw_response: Mapping[str, Any], tool_results
            ) -> Mapping[str, Any]:
                return raw_response

            @classmethod
            def build_history(
                cls, raw_response: Mapping[str, Any], messages: List[Mapping[str, Any]]
            ) -> List[Mapping[str, Any]]:
                return messages

        # Call async version
        result = await CustomControl.aextract_flow_result({})

        # Verify sync was called
        assert CustomControl.sync_called is True
        assert result.is_complete is True

    @pytest.mark.asyncio
    async def test_async_inject_results(self):
        """Test async inject_results defaults to sync."""

        class CustomControl(ToolFlowControl):
            inject_called = False

            @classmethod
            def extract_flow_result(
                cls, raw_response: Mapping[str, Any]
            ) -> ToolFlowResult:
                return ToolFlowResult(
                    is_complete=True,
                    tool_calls=None,
                    reasoning=None,
                    final_response=None,
                )

            @classmethod
            def inject_results(
                cls, raw_response: Mapping[str, Any], tool_results
            ) -> Mapping[str, Any]:
                cls.inject_called = True
                return raw_response

            @classmethod
            def build_history(
                cls, raw_response: Mapping[str, Any], messages: List[Mapping[str, Any]]
            ) -> List[Mapping[str, Any]]:
                return messages

        await CustomControl.ainject_results({}, MagicMock())
        assert CustomControl.inject_called is True

    @pytest.mark.asyncio
    async def test_async_build_history(self):
        """Test async build_history defaults to sync."""

        class CustomControl(ToolFlowControl):
            history_called = False

            @classmethod
            def extract_flow_result(
                cls, raw_response: Mapping[str, Any]
            ) -> ToolFlowResult:
                return ToolFlowResult(
                    is_complete=True,
                    tool_calls=None,
                    reasoning=None,
                    final_response=None,
                )

            @classmethod
            def inject_results(
                cls, raw_response: Mapping[str, Any], tool_results
            ) -> Mapping[str, Any]:
                return raw_response

            @classmethod
            def build_history(
                cls, raw_response: Mapping[str, Any], messages: List[Mapping[str, Any]]
            ) -> List[Mapping[str, Any]]:
                cls.history_called = True
                messages.append({"role": "test"})
                return messages

        result = await CustomControl.abuild_history({}, [])
        assert CustomControl.history_called is True
        assert len(result) == 1
