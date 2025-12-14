"""Tests for tool_filter and max_tool_turns features."""

import pytest

from msgflux.nn.modules.agent import Agent


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

    def test_block_all_with_wildcard(self):
        """Test that block '*' removes all tools."""

        agent = Agent.__new__(Agent)
        agent.config = {}

        result = agent._apply_tool_filter(self.mock_schemas, {"block": "*"})

        assert result == []


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
