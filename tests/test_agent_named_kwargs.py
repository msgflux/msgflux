"""
Test for Agent named kwargs functionality.

This test validates that agents can accept named keyword arguments as task inputs
when a task template is configured, enabling better integration with tool calling
systems that require typed parameters.
"""

from unittest.mock import MagicMock

import pytest

from msgflux.core.message import Message
from msgflux.nn.modules.agent import Agent


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


def test_named_kwargs_with_task_template():
    """Test that named kwargs work when task template is configured."""
    model = create_mock_model()

    agent = Agent(
        name="greeter",
        model=model,
        templates={"task": "Hello {{name}}, you are {{age}} years old!"},
        annotations={"name": str, "age": int, "return": str},
    )

    # This should work - named kwargs with task template
    result = agent(name="João", age=27)

    assert result is not None
    print("✓ Test 1 passed: Named kwargs accepted with task template")


def test_named_kwargs_without_task_template_raises_error():
    """Test that named kwargs raise error when no task template is configured."""
    model = create_mock_model()

    agent = Agent(
        name="test_agent",
        model=model,
        annotations={"name": str, "age": int, "return": str},
    )

    # Named kwargs now normalize to a dict task, which requires a task template
    with pytest.raises(
        ValueError, match="Dict task requires a 'task' template to be configured"
    ):
        agent(name="João", age=27)

    print("✓ Test 2 passed: ValueError raised when using named kwargs without template")


def test_named_kwargs_with_message_raises_error():
    """Test that passing both message and named kwargs raises error."""
    model = create_mock_model()

    agent = Agent(
        name="greeter",
        model=model,
        templates={"task": "Hello {{name}}!"},
        annotations={"name": str, "return": str},
    )

    # Should raise ValueError when both message and named kwargs are provided
    with pytest.raises(
        ValueError, match="Cannot pass both 'message' argument and named task arguments"
    ):
        agent("some message", name="João")

    print(
        "✓ Test 3 passed: ValueError raised when both message and named kwargs provided"
    )


def test_reserved_kwargs_not_treated_as_task():
    """Test that reserved kwargs (vars, messages, etc.) are not treated as task inputs."""
    model = create_mock_model()

    agent = Agent(name="test_agent", model=model, templates={"task": "Test message"})

    # Reserved kwargs should not trigger the named kwargs logic
    # This should work without error
    result = agent("Hello", vars={"extra": "data"})

    assert result is not None
    print("✓ Test 4 passed: Reserved kwargs handled correctly")


def test_backward_compatibility_with_dict():
    """Test that passing dict as message still works (backward compatibility)."""
    model = create_mock_model()

    agent = Agent(
        name="greeter",
        model=model,
        templates={"task": "Hello {{name}}, you are {{age}} years old!"},
    )

    # Old way should still work
    result = agent({"name": "João", "age": 27})

    assert result is not None
    print("✓ Test 5 passed: Dict input still works (backward compatible)")


def test_backward_compatibility_with_string():
    """Test that passing string as message still works (backward compatibility)."""
    model = create_mock_model()

    agent = Agent(name="test_agent", model=model)

    # String input should still work
    result = agent("Hello world")

    assert result is not None
    print("✓ Test 6 passed: String input still works (backward compatible)")


def test_mixed_reserved_and_named_kwargs():
    """Test that mixing reserved kwargs with named task kwargs works correctly."""
    model = create_mock_model()

    agent = Agent(name="greeter", model=model, templates={"task": "Hello {{name}}!"})

    # Should work - reserved kwargs are extracted, named kwargs become task inputs
    result = agent(name="João", vars={"extra": "data"})

    assert result is not None
    print("✓ Test 7 passed: Mixed reserved and named kwargs handled correctly")


def test_message_envelope_plus_named_kwargs_works():
    """Test that Message envelope can coexist with named task kwargs."""
    model = create_mock_model()

    agent = Agent(
        name="greeter",
        model=model,
        templates={"task": "Hello {{name}}!"},
        message_fields={"vars": "vars"},
    )

    msg = Message()
    msg.set("vars.locale", "pt-BR")

    result = agent(msg, name="João")

    assert result is not None
    print("✓ Test 8 passed: Message envelope works with named task kwargs")


def test_message_envelope_plus_explicit_task_works():
    """Test that Message envelope can coexist with explicit task kwarg."""
    model = create_mock_model()

    agent = Agent(name="greeter", model=model, templates={"task": "{{ greeting }}"})

    msg = Message()
    result = agent(msg, task={"greeting": "Olá"})

    assert result is not None
    print("✓ Test 9 passed: Message envelope works with explicit task kwarg")


@pytest.mark.asyncio
async def test_async_named_kwargs():
    """Test that named kwargs work in async mode."""
    model = create_mock_model()

    # Mock async model call
    async def async_mock(*args, **kwargs):
        mock_response = MagicMock()
        mock_response.response_type = "text_generation"
        mock_response.consume.return_value = "Async mocked response"
        return mock_response

    model.acall = async_mock

    agent = Agent(
        name="greeter",
        model=model,
        templates={"task": "Hello {{name}}!"},
        annotations={"name": str, "return": str},
    )

    # Test async forward with named kwargs
    result = await agent.acall(name="João")

    assert result is not None
    print("✓ Test 10 passed: Named kwargs work in async mode")


def test_inspect_model_execution_params_with_named_kwargs():
    """Test that inspect_model_execution_params works with named kwargs."""
    model = create_mock_model()

    agent = Agent(
        name="classifier", model=model, templates={"task": "Classify: {{sentence}}"}
    )

    # Should work with named kwargs (no message arg)
    params = agent.inspect_model_execution_params(
        sentence="Hello world", confidence=0.99
    )

    assert params is not None
    assert "messages" in params
    print("✓ Test 11 passed: inspect_model_execution_params works with named kwargs")


def test_inspect_model_execution_params_message_plus_kwargs_raises_error():
    """Test that inspect_model_execution_params raises error with message + task kwargs."""
    model = create_mock_model()

    agent = Agent(
        name="classifier", model=model, templates={"task": "Classify: {{sentence}}"}
    )

    # Should raise error when passing message AND other task kwargs (ambiguous)
    with pytest.raises(
        ValueError, match="Cannot pass both 'message' argument and named task arguments"
    ):
        agent.inspect_model_execution_params(
            message="direct message", sentence="Hello world", confidence=0.99
        )

    print(
        "✓ Test 12 passed: inspect_model_execution_params rejects message + task kwargs"
    )


def test_inspect_model_execution_params_with_message_positional():
    """Test that inspect_model_execution_params works with message as positional arg."""
    model = create_mock_model()

    agent = Agent(name="test_agent", model=model)

    # Should work with message as positional
    params = agent.inspect_model_execution_params("Hello world")

    assert params is not None
    assert "messages" in params
    print(
        "✓ Test 13 passed: inspect_model_execution_params works with positional message"
    )


if __name__ == "__main__":
    import asyncio

    # Run sync tests
    test_named_kwargs_with_task_template()
    test_named_kwargs_without_task_template_raises_error()
    test_named_kwargs_with_message_raises_error()
    test_reserved_kwargs_not_treated_as_task()
    test_backward_compatibility_with_dict()
    test_backward_compatibility_with_string()
    test_mixed_reserved_and_named_kwargs()
    test_message_envelope_plus_named_kwargs_works()
    test_message_envelope_plus_explicit_task_works()
    test_inspect_model_execution_params_with_named_kwargs()
    test_inspect_model_execution_params_message_plus_kwargs_raises_error()
    test_inspect_model_execution_params_with_message_positional()

    # Run async test
    asyncio.run(test_async_named_kwargs())

    print("\n✅ All tests passed! Named kwargs feature validated successfully.")
    print("\nFeature summary:")
    print(
        "  1. Agents can accept named kwargs as task inputs when task template exists"
    )
    print("  2. Reserved kwargs (vars, messages, etc.) are handled separately")
    print("  3. Clear error messages when template is missing or args conflict")
    print("  4. Fully backward compatible with existing string/dict/Message inputs")
    print("  5. Works in both sync and async modes")
    print("  6. inspect_model_execution_params() supports all argument patterns")
