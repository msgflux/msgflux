"""Tests for reasoning strategies (CoT, ReAct, Self-Consistency)."""

from unittest.mock import MagicMock

import msgspec
import pytest
from msgspec import Struct

from msgflux.generation.control_flow import ToolFlowControl
from msgflux.generation.reasoning.cot import ChainOfThought
from msgflux.generation.reasoning.react import (
    REACT_SYSTEM_MESSAGE,
    REACT_TOOLS_TEMPLATE,
    Action,
    ReAct,
)


class TestChainOfThought:
    """Tests for Chain of Thought reasoning strategy."""

    def test_chain_of_thought_is_struct(self):
        """Test that ChainOfThought is a Struct."""
        assert issubclass(ChainOfThought, Struct)

    def test_chain_of_thought_has_required_fields(self):
        """Test that ChainOfThought has required fields."""
        cot = ChainOfThought(
            reasoning="First, we need to analyze the problem. Then we solve it.",
            final_answer="42",
        )
        assert (
            cot.reasoning == "First, we need to analyze the problem. Then we solve it."
        )
        assert cot.final_answer == "42"


class TestReActToolFlowControl:
    """Tests for ReAct ToolFlowControl interface implementation."""

    def test_react_inherits_tool_flow_control(self):
        """Test that ReAct inherits from ToolFlowControl."""
        assert issubclass(ReAct, ToolFlowControl)

    def test_react_has_class_attributes(self):
        """Test that ReAct has system_message and tools_template."""
        assert ReAct.system_message == REACT_SYSTEM_MESSAGE
        assert ReAct.tools_template == REACT_TOOLS_TEMPLATE

    def test_action_struct(self):
        """Test that Action struct holds name and arguments."""
        action = Action(
            name="search",
            arguments={"query": "test"},
        )
        assert action.name == "search"
        assert action.arguments["query"] == "test"

        # Without arguments
        action_no_args = Action(name="noop")
        assert action_no_args.arguments is None

    def test_extract_flow_result_with_final_answer(self):
        """Test extract_flow_result when final_answer is present."""
        raw_response = {
            "thought": None,
            "actions": None,
            "final_answer": "The answer is 42",
        }
        result = ReAct.extract_flow_result(raw_response)

        assert result.is_complete is True
        assert result.tool_calls is None
        assert result.reasoning is None
        assert result.final_response is raw_response

    def test_extract_flow_result_with_actions(self):
        """Test extract_flow_result when actions are present."""
        raw_response = {
            "thought": "I need to search for information",
            "actions": [
                {
                    "name": "search",
                    "arguments": {"query": "Python docs"},
                },
                {
                    "name": "calculate",
                    "arguments": {"a": 1, "b": 2},
                },
            ],
            "final_answer": None,
        }
        result = ReAct.extract_flow_result(raw_response)

        assert result.is_complete is False
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0][1] == "search"
        assert result.tool_calls[0][2] == {"query": "Python docs"}
        assert result.tool_calls[1][1] == "calculate"
        assert result.tool_calls[1][2] == {"a": 1, "b": 2}
        assert result.reasoning == "I need to search for information"
        assert result.final_response is None

        # Verify _id was assigned to actions (runtime field)
        actions = raw_response["actions"]
        assert actions[0].get("_id") is not None
        assert actions[1].get("_id") is not None

    def test_extract_flow_result_empty_state(self):
        """Test extract_flow_result with no actions and no final_answer."""
        raw_response = {"thought": None, "actions": None, "final_answer": None}
        result = ReAct.extract_flow_result(raw_response)

        assert result.is_complete is True
        assert result.tool_calls is None
        assert result.final_response is raw_response

    def test_inject_results(self):
        """Test inject_results adds observations."""
        raw_response = {
            "thought": "Testing",
            "actions": [
                {
                    "name": "search",
                    "arguments": {"q": "test"},
                    "_id": "id1",
                },
                {
                    "name": "calc",
                    "arguments": {"a": 1},
                    "_id": "id2",
                },
            ],
            "final_answer": None,
        }

        # Mock tool results
        mock_results = MagicMock()
        mock_result1 = MagicMock()
        mock_result1.result = "search result"
        mock_result1.error = None
        mock_result2 = MagicMock()
        mock_result2.result = None
        mock_result2.error = "calculation error"

        def get_by_id(tool_id):
            if tool_id == "id1":
                return mock_result1
            elif tool_id == "id2":
                return mock_result2
            return None

        mock_results.get_by_id = get_by_id

        raw_response = ReAct.inject_results(raw_response, mock_results)

        observations = raw_response["observations"]
        assert len(observations) == 2
        assert observations[0]["tool"] == "search"
        assert observations[0]["result"] == "search result"
        assert observations[1]["tool"] == "calc"
        assert observations[1]["result"] == "calculation error"

    def test_build_history_new_message(self):
        """Test build_history adds new assistant message."""
        raw_response = {
            "thought": "Testing",
            "actions": [
                {"name": "search", "arguments": {"q": "test"}},
            ],
            "observations": [{"tool": "search", "result": "found"}],
            "final_answer": None,
        }

        messages = []
        result = ReAct.build_history(raw_response, messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"

    def test_build_history_append_to_existing(self):
        """Test build_history appends to existing assistant message."""
        raw_response = {
            "thought": "Second step",
            "actions": [
                {"name": "calc", "arguments": {"a": 1}},
            ],
            "observations": [{"tool": "calc", "result": "42"}],
            "final_answer": None,
        }

        # Simulate existing ReAct state in messages
        existing_step = {
            "thought": "First step",
            "actions": [
                {"name": "search", "arguments": {"q": "test"}},
            ],
            "observations": [{"tool": "search", "result": "found"}],
        }
        existing_content = msgspec.json.encode([existing_step]).decode()
        messages = [{"role": "assistant", "content": existing_content}]

        result = ReAct.build_history(raw_response, messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_async_methods_work(self):
        """Test that async methods work (default to sync)."""
        raw_response = {"thought": None, "actions": None, "final_answer": "Done"}

        result = await ReAct.aextract_flow_result(raw_response)
        assert result.is_complete is True

        raw_response2 = await ReAct.ainject_results(raw_response, MagicMock())
        assert raw_response2 is raw_response

        history = await ReAct.abuild_history(raw_response, [])
        assert len(history) == 1

    def test_build_provider_response_format_flattens_tool_parameters(self):
        response_format = ReAct.build_provider_response_format(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "store_fields",
                        "description": "Store values",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "fields": {
                                    "type": "object",
                                    "properties": {"entries": {"type": "array"}},
                                    "required": ["entries"],
                                    "additionalProperties": False,
                                }
                            },
                            "required": ["fields"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    },
                }
            ]
        )

        action_schema = response_format["json_schema"]["schema"]["properties"]["actions"][
            "anyOf"
        ][0]["items"]
        assert action_schema["properties"]["name"]["enum"] == ["store_fields"]
        assert "fields" in action_schema["properties"]
        assert "arguments" not in action_schema["properties"]

    def test_normalize_provider_response_restores_action_arguments(self):
        raw_response = {
            "thought": "Store the fields",
            "actions": [
                {
                    "name": "store_fields",
                    "fields": {
                        "entries": [
                            {"key": "city", "value": "Austin"},
                            {"key": "country", "value": "USA"},
                        ]
                    },
                }
            ],
            "final_answer": None,
        }

        normalized = ReAct.normalize_provider_response(
            raw_response,
            tool_annotations={"store_fields": {"fields": dict[str, str]}},
        )

        assert normalized == {
            "thought": "Store the fields",
            "actions": [
                {
                    "name": "store_fields",
                    "arguments": {"fields": {"city": "Austin", "country": "USA"}},
                }
            ],
            "final_answer": None,
        }
