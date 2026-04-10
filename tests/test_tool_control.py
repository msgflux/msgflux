"""Tests for tool_filter and max_tool_turns features."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from msgflux.core.message import Message
from msgflux.nn.modules.agent import Agent


def search(query: str) -> str:
    """Search tool."""
    return query


def browser(url: str) -> str:
    """Browser tool."""
    return url


class TestToolFilter:
    """Tests for _apply_tool_filter method."""

    def setup_method(self):
        """Create a minimal agent for testing."""

        class MockModel:
            pass

        # We'll test the _apply_tool_filter method directly
        self.mock_schemas = [
            {"function": {"name": "search", "description": "Search tool"}},
            {"function": {"name": "calculator", "description": "Calculator tool"}},
            {"function": {"name": "browser", "description": "Browser tool"}},
        ]

    def test_allow_filter(self):
        """Test that allow filter keeps only specified tools."""

        class MockModel:
            pass

        # Create agent with minimal config
        agent = Agent.__new__(Agent)
        agent.config = {}

        result = agent._apply_tool_filter(
            self.mock_schemas, {"allow": ["search", "calculator"]}
        )

        assert len(result) == 2
        names = [s["function"]["name"] for s in result]
        assert "search" in names
        assert "calculator" in names
        assert "browser" not in names

    def test_allow_filter_accepts_single_value(self):
        """Test that allow filter accepts a single tool name."""

        agent = Agent.__new__(Agent)
        agent.config = {}

        result = agent._apply_tool_filter(self.mock_schemas, {"allow": "search"})

        assert [schema["function"]["name"] for schema in result] == ["search"]

    def test_block_filter(self):
        """Test that block filter removes specified tools."""

        agent = Agent.__new__(Agent)
        agent.config = {}

        result = agent._apply_tool_filter(self.mock_schemas, {"block": ["browser"]})

        assert len(result) == 2
        names = [s["function"]["name"] for s in result]
        assert "search" in names
        assert "calculator" in names
        assert "browser" not in names

    def test_block_filter_accepts_single_value(self):
        """Test that block filter accepts a single tool name."""

        agent = Agent.__new__(Agent)
        agent.config = {}

        result = agent._apply_tool_filter(self.mock_schemas, {"block": "browser"})

        names = [s["function"]["name"] for s in result]
        assert "browser" not in names
        assert "search" in names
        assert "calculator" in names

    def test_empty_filter_raises(self):
        """Test that empty filter dict raises error."""

        agent = Agent.__new__(Agent)
        agent.config = {}

        with pytest.raises(ValueError, match="must contain 'allow' or 'block' key"):
            agent._apply_tool_filter(self.mock_schemas, {})

    def test_invalid_key_raises(self):
        """Test that invalid filter key raises error."""

        agent = Agent.__new__(Agent)
        agent.config = {}

        with pytest.raises(ValueError, match="invalid keys"):
            agent._apply_tool_filter(self.mock_schemas, {"invalid": ["search"]})

    def test_both_keys_raises(self):
        """Test that both allow and block raises error."""

        agent = Agent.__new__(Agent)
        agent.config = {}

        with pytest.raises(ValueError, match="only one key"):
            agent._apply_tool_filter(
                self.mock_schemas, {"allow": ["search"], "block": ["browser"]}
            )

    def test_non_dict_filter_raises(self):
        """Test that non-dict filter raises error."""

        agent = Agent.__new__(Agent)
        agent.config = {}

        with pytest.raises(ValueError, match="must be a dict"):
            agent._apply_tool_filter(self.mock_schemas, ["search"])

    def test_invalid_filter_value_raises(self):
        """Test that filter values must be strings or lists of strings."""

        agent = Agent.__new__(Agent)
        agent.config = {}

        with pytest.raises(ValueError, match="must be a string or list of strings"):
            agent._apply_tool_filter(self.mock_schemas, {"allow": {"search"}})

    def test_block_all_with_wildcard(self):
        """Test that block '*' removes all tools."""

        agent = Agent.__new__(Agent)
        agent.config = {}

        result = agent._apply_tool_filter(self.mock_schemas, {"block": "*"})

        assert result == []


class TestToolFilterIntegration:
    """Integration tests for tool_filter with Agent public APIs."""

    def _create_agent(self, **kwargs):
        mock_model = Mock()
        mock_model.model_type = "chat_completion"
        return Agent(name="agent", model=mock_model, tools=[search, browser], **kwargs)

    def test_inspect_model_execution_params_accepts_tool_filter(self):
        """tool_filter should work with inspect_model_execution_params."""

        agent = self._create_agent()

        params = agent.inspect_model_execution_params(
            "What is the weather?", tool_filter={"allow": "search"}
        )

        names = [schema["function"]["name"] for schema in params.tool_schemas]
        assert names == ["search"]

    def test_message_fields_tool_filter_is_applied(self):
        """tool_filter can be loaded from Message via message_fields."""

        agent = self._create_agent(
            message_fields={
                "task_inputs": "content",
                "tool_filter": "extra.tool_filter",
            }
        )
        message = Message(content="What is the weather?")
        message.set("extra.tool_filter", {"allow": "search"})

        params = agent.inspect_model_execution_params(message)

        names = [schema["function"]["name"] for schema in params.tool_schemas]
        assert names == ["search"]

    def test_runtime_tool_filter_overrides_message_field(self):
        """Explicit runtime tool_filter should override message_fields value."""

        agent = self._create_agent(
            message_fields={
                "task_inputs": "content",
                "tool_filter": "extra.tool_filter",
            }
        )
        message = Message(content="What is the weather?")
        message.set("extra.tool_filter", {"block": "search"})

        params = agent.inspect_model_execution_params(
            message, tool_filter={"allow": "search"}
        )

        names = [schema["function"]["name"] for schema in params.tool_schemas]
        assert names == ["search"]

    def test_tool_choice_is_reconciled_after_filtering(self):
        """tool_choice should not point to a filtered-out tool."""

        agent = self._create_agent(config={"tool_choice": "browser"})

        params = agent.inspect_model_execution_params(
            "What is the weather?", tool_filter={"allow": "search"}
        )

        assert [schema["function"]["name"] for schema in params.tool_schemas] == [
            "search"
        ]
        assert params.tool_choice == "auto"


class TestMaxToolTurnsConfig:
    """Tests for max_tool_turns config validation."""

    def _create_agent(self):
        """Create an agent with minimal initialization for testing."""
        agent = Agent.__new__(Agent)
        agent._buffers = {}
        agent._non_persistent_buffers_set = set()
        return agent

    def test_valid_max_tool_turns(self):
        """Test that valid max_tool_turns is accepted."""
        agent = self._create_agent()
        agent._set_config({"max_tool_turns": 5})
        assert agent.config["max_tool_turns"] == 5

    def test_invalid_max_tool_turns_zero(self):
        """Test that zero max_tool_turns raises error."""
        agent = self._create_agent()
        with pytest.raises(ValueError, match="positive integer"):
            agent._set_config({"max_tool_turns": 0})

    def test_invalid_max_tool_turns_negative(self):
        """Test that negative max_tool_turns raises error."""
        agent = self._create_agent()
        with pytest.raises(ValueError, match="positive integer"):
            agent._set_config({"max_tool_turns": -1})

    def test_invalid_max_tool_turns_string(self):
        """Test that string max_tool_turns raises error."""
        agent = self._create_agent()
        with pytest.raises(ValueError, match="positive integer"):
            agent._set_config({"max_tool_turns": "5"})


class TestMaxToolTurnsBehavior:
    """Tests for max_tool_turns execution behavior."""

    def test_second_tool_turn_is_blocked_before_execution(self):
        """After the limit is reached, tools are removed for a final answer turn."""
        agent = Agent.__new__(Agent)
        agent.name = "agent"
        agent.config = {"max_tool_turns": 1}

        processed_tool_turns = []
        execution_filters = []

        class ToolResults:
            def __init__(self):
                self.return_directly = False
                self.tool_calls = [SimpleNamespace(id="id", result="ok", error=None)]

        class RawResponse:
            def __init__(self, label: str):
                self.reasoning = None
                self.label = label

            def get_calls(self):
                return [("id", self.label, {})]

            def insert_results(self, id_results):
                self.id_results = id_results

            def get_messages(self):
                return []

        first = SimpleNamespace(response_type="tool_call", data=RawResponse("first"))
        second = SimpleNamespace(response_type="tool_call", data=RawResponse("second"))
        final = SimpleNamespace(response_type="text_generation", data="done")
        queued_responses = [second, final]

        def process_tool_call(tool_callings, messages, vars):
            processed_tool_turns.append(tool_callings[0][1])
            return ToolResults()

        def execute_model(**kwargs):
            execution_filters.append(kwargs.get("tool_filter"))
            return queued_responses.pop(0)

        agent._process_tool_call = process_tool_call
        agent._execute_model = execute_model

        result, messages = agent._process_tool_call_response(first, [], {}, None, None)

        assert processed_tool_turns == ["first"]
        assert execution_filters == [None, {"block": "*"}]
        assert result.response_type == "text_generation"
        assert messages == []
