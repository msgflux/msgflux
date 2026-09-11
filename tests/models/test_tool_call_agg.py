"""Tests for msgflux.models.tool_call_agg module."""

import json

import pytest

from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.tools.runtime import ToolOutcome


class TestToolCallAggregatorBasics:
    """Test suite for basic ToolCallAggregator functionality."""

    def test_aggregator_initialization(self):
        """Test ToolCallAggregator initialization."""
        agg = ToolCallAggregator()
        assert agg.reasoning is None
        assert len(agg.tool_calls) == 0

    def test_aggregator_initialization_with_reasoning(self):
        """Test ToolCallAggregator initialization with reasoning."""
        agg = ToolCallAggregator(reasoning="Let me think about this...")
        assert agg.reasoning == "Let me think about this..."
        assert len(agg.tool_calls) == 0

    def test_process_single_tool_call(self):
        """Test processing a single complete tool call."""
        agg = ToolCallAggregator()
        agg.process(
            call_index=0,
            tool_id="call_123",
            name="get_weather",
            arguments='{"location": "NYC"}',
        )

        assert len(agg.tool_calls) == 1
        assert 0 in agg.tool_calls
        assert agg.tool_calls[0]["id"] == "call_123"
        assert agg.tool_calls[0]["name"] == "get_weather"
        assert agg.tool_calls[0]["arguments"] == '{"location": "NYC"}'

    def test_process_incremental_arguments(self):
        """Test processing tool call with incremental arguments (streaming)."""
        agg = ToolCallAggregator()

        # First chunk
        agg.process(call_index=0, tool_id="", name="calculate", arguments='{"op')

        # Second chunk
        agg.process(
            call_index=0, tool_id="call_456", name="", arguments='eration": "add'
        )

        # Third chunk
        agg.process(call_index=0, tool_id="", name="", arguments='", "a": 5, "b": 3}')

        assert len(agg.tool_calls) == 1
        assert agg.tool_calls[0]["id"] == "call_456"
        assert agg.tool_calls[0]["name"] == "calculate"
        assert agg.tool_calls[0]["arguments"] == '{"operation": "add", "a": 5, "b": 3}'

    def test_decodes_canonical_intents(self):
        agg = ToolCallAggregator()
        agg.process(0, "call_123", "get_weather", '{"location":"NYC"}')

        intents = agg.get_intents()

        assert len(intents) == 1
        assert intents[0].id == "call_123"
        assert intents[0].name == "get_weather"
        assert intents[0].arguments == {"location": "NYC"}

    def test_chat_completions_renders_outcomes_as_messages(self):
        agg = ToolCallAggregator()
        agg.process(0, "call_123", "get_weather", '{"location":"NYC"}')
        intent = agg.get_intents()[0]

        messages = agg.render_outcomes(
            [ToolOutcome.completed(intent, {"temperature": 21})]
        )

        assert messages[0]["role"] == "assistant"
        assert messages[0]["tool_calls"][0]["id"] == "call_123"
        assert messages[1] == {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": '{"temperature":21}',
        }

    def test_responses_renders_canonical_function_call_output(self):
        agg = ToolCallAggregator(api_mode="responses")
        agg.process(0, "call_123", "get_weather", '{"location":"NYC"}')
        intent = agg.get_intents()[0]

        items = agg.render_outcomes([ToolOutcome.completed(intent, "sunny")])

        assert items == [
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": "sunny",
            }
        ]

    def test_requires_one_outcome_per_call(self):
        agg = ToolCallAggregator(api_mode="responses")
        agg.process(0, "call_123", "get_weather", "{}")

        with pytest.raises(ValueError, match="call_123"):
            agg.render_outcomes([])
