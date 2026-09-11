"""Tests for control flow functionality."""

from typing import Any, List, Mapping, Optional
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

    def test_tool_flow_control_is_class(self):
        """Test that ToolFlowControl exists and is a class."""
        assert isinstance(ToolFlowControl, type)

    def test_tool_flow_control_can_be_inherited(self):
        """Test that ToolFlowControl can be inherited with required methods."""

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

        assert issubclass(CustomControl, ToolFlowControl)

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


class TestCustomToolFlowControl:
    """Tests for custom ToolFlowControl implementation."""

    def test_simple_tool_loop(self):
        """Test a simple custom tool flow control."""

        class SimpleToolLoop(ToolFlowControl):
            """Simple tool loop without ReAct structure."""

            @classmethod
            def extract_flow_result(
                cls, raw_response: Mapping[str, Any]
            ) -> ToolFlowResult:
                if raw_response.get("done"):
                    return ToolFlowResult(
                        is_complete=True,
                        tool_calls=None,
                        reasoning=None,
                        final_response={"done": True},
                    )

                calls = raw_response.get("calls", [])
                if calls:
                    tool_calls = [
                        (f"id_{i}", call["name"], call["args"])
                        for i, call in enumerate(calls)
                    ]
                    return ToolFlowResult(
                        is_complete=False,
                        tool_calls=tool_calls,
                        reasoning="Processing calls",
                        final_response=None,
                    )

                return ToolFlowResult(
                    is_complete=True,
                    tool_calls=None,
                    reasoning=None,
                    final_response={"done": True, "no_calls": True},
                )

            @classmethod
            def inject_results(
                cls, raw_response: Mapping[str, Any], tool_results
            ) -> Mapping[str, Any]:
                calls = raw_response.get("calls", [])
                for i, call in enumerate(calls):
                    call_id = f"id_{i}"
                    result = tool_results.get_by_id(call_id)
                    if result:
                        call["result"] = result.result
                return raw_response

            @classmethod
            def build_history(
                cls, raw_response: Mapping[str, Any], messages: List[Mapping[str, Any]]
            ) -> List[Mapping[str, Any]]:
                messages.append({"role": "assistant", "content": str(raw_response)})
                return messages

        # Test completed state
        result = SimpleToolLoop.extract_flow_result({"done": True})
        assert result.is_complete is True

        # Test with pending calls
        raw = {"calls": [{"name": "search", "args": {"q": "test"}}]}
        result = SimpleToolLoop.extract_flow_result(raw)
        assert result.is_complete is False
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0][1] == "search"

        # Test build_history
        messages = []
        messages = SimpleToolLoop.build_history(raw, messages)
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"


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
