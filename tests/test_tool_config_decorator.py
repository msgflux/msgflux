"""
Test for tool_config decorator with classes and AutoParams.

This test validates that tool_config is properly accessible both as a class
attribute and from instances, especially when used with AutoParams.
"""

from typing import Literal
from unittest.mock import MagicMock

import msgflux as mf
from msgflux.nn import Agent


def create_mock_model():
    """Create a mock chat completion model."""
    model = MagicMock()
    model.model_type = "chat_completion"

    # Mock the model response
    mock_response = MagicMock()
    mock_response.response_type = "text_generation"
    mock_response.consume.return_value = "Mocked response"
    model.return_value = mock_response

    return model


def test_tool_config_accessible_from_class():
    """Test that tool_config is accessible as a class attribute."""
    mock_model = create_mock_model()

    @mf.tool_config(return_direct=True, spawn=False)
    class TestAgent(Agent):
        """Test agent."""

        pass

    # Create instance to test
    agent = TestAgent(name="TestAgent", model=mock_model)

    # Should be accessible from class
    assert hasattr(TestAgent, "tool_config")
    assert TestAgent.tool_config.return_direct is True
    assert TestAgent.tool_config.spawn is False

    # Also from instance
    assert hasattr(agent, "tool_config")
    assert agent.tool_config.return_direct is True

    print("✓ Test 1 passed: tool_config accessible from class")


def test_tool_config_accessible_from_instance():
    """Test that tool_config is accessible from instances."""
    mock_model = create_mock_model()

    @mf.tool_config(return_direct=True, call_as_response=False)
    class TestAgent(Agent):
        """Test agent."""

        pass

    instance = TestAgent(name="TestAgent", model=mock_model)

    # Should be accessible from instance
    assert hasattr(instance, "tool_config")
    assert instance.tool_config.return_direct is True
    assert instance.tool_config.call_as_response is False

    print("✓ Test 2 passed: tool_config accessible from instance")


def test_tool_config_with_agent_as_tool():
    """Test that tool_config works when Agent class is used as tool."""
    mock_model = create_mock_model()

    class Classify(mf.Signature):
        """Classify sentiment."""

        sentence: str = mf.InputField()
        sentiment: Literal["positive", "negative", "neutral"] = mf.OutputField()

    @mf.tool_config(return_direct=True)
    class SentimentClassifier(Agent):
        """Sentiment classifier agent."""

        name = "SentimentClassifier"
        model = mock_model
        signature = Classify

    # Create assistant with classifier as tool
    assistant = Agent(
        name="Assistant",
        model=mock_model,
        tools=[SentimentClassifier],
    )

    # Verify tool was added to library
    assert "SentimentClassifier" in assistant.tool_library.library

    # Verify tool_config was preserved
    tool = assistant.tool_library.library["SentimentClassifier"]
    assert hasattr(tool, "tool_config")
    assert tool.tool_config.return_direct is True

    print("✓ Test 3 passed: tool_config preserved when used as tool")


def test_tool_config_values_are_correct():
    """Test that all tool_config values are set correctly."""
    mock_model = create_mock_model()

    @mf.tool_config(
        return_direct=True,
        call_as_response=False,
        spawn=False,
        inject_messages=False,
        inject_vars=["var1", "var2"],
        handoff=False,
        name_override="CustomName",
    )
    class TestAgent(Agent):
        """Test agent."""

        name = "TestAgent"
        model = mock_model

    config = TestAgent.tool_config

    assert config.return_direct is True
    assert config.call_as_response is False
    assert config.spawn is False
    assert config.inject_messages is False
    assert config.inject_vars == ["var1", "var2"]
    assert config.handoff is False
    assert config.name_overridden == "CustomName"

    print("✓ Test 4 passed: All tool_config values set correctly")


def test_tool_config_handoff_sets_return_direct():
    """Test that handoff=True automatically sets return_direct=True."""
    mock_model = create_mock_model()

    @mf.tool_config(handoff=True)
    class HandoffAgent(Agent):
        """Handoff agent."""

        name = "HandoffAgent"
        model = mock_model

    # handoff should automatically enable return_direct and inject_messages
    assert HandoffAgent.tool_config.handoff is True
    assert HandoffAgent.tool_config.return_direct is True
    assert HandoffAgent.tool_config.inject_messages is True

    print("✓ Test 5 passed: handoff=True sets dependent flags")


def test_tool_config_with_autoparams():
    """Test that tool_config works with AutoParams class attributes."""
    mock_model = create_mock_model()

    @mf.tool_config(return_direct=True)
    class AutoParamsAgent(Agent):
        """Agent with AutoParams class attributes."""

        name = "AutoParamsAgent"
        model = mock_model
        system_message = "You are a helpful assistant"

    # Class-level access
    assert AutoParamsAgent.tool_config.return_direct is True

    # Instance-level access
    instance = AutoParamsAgent()
    assert instance.tool_config.return_direct is True

    # Verify system_message works (related bug fix)
    assert hasattr(instance.system_message, "data")
    assert instance.system_message.data == "You are a helpful assistant"

    print("✓ Test 6 passed: tool_config compatible with AutoParams")


def test_multiple_decorated_classes_dont_share_config():
    """Test that different decorated classes have independent tool_config."""
    mock_model = create_mock_model()

    @mf.tool_config(return_direct=True)
    class Agent1(Agent):
        """First agent."""

        name = "Agent1"
        model = mock_model

    @mf.tool_config(return_direct=False, spawn=True)
    class Agent2(Agent):
        """Second agent."""

        name = "Agent2"
        model = mock_model

    # Each should have its own config
    assert Agent1.tool_config.return_direct is True
    assert Agent1.tool_config.spawn is False

    assert Agent2.tool_config.return_direct is False
    assert Agent2.tool_config.spawn is True

    print("✓ Test 7 passed: Multiple classes have independent configs")


if __name__ == "__main__":
    test_tool_config_accessible_from_class()
    test_tool_config_accessible_from_instance()
    test_tool_config_with_agent_as_tool()
    test_tool_config_values_are_correct()
    test_tool_config_handoff_sets_return_direct()
    test_tool_config_with_autoparams()
    test_multiple_decorated_classes_dont_share_config()

    print("\n✅ All tests passed! tool_config decorator validated.")
    print("\nFeature summary:")
    print("  1. tool_config accessible as class attribute")
    print("  2. tool_config accessible from instances")
    print("  3. tool_config preserved when class used as tool")
    print("  4. All configuration values set correctly")
    print("  5. Automatic flag dependencies (handoff → return_direct)")
    print("  6. Compatible with AutoParams")
    print("  7. Independent configs for different classes")
