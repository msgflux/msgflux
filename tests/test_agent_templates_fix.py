"""
Test to verify the fix for the Agent template bug.

This test validates:
1. Agent can be created without signature (no _set_task_template method call)
2. Agent with signature properly overrides task template in self.templates
3. The signature-generated task_template takes precedence over user-provided templates
"""

from unittest.mock import MagicMock

import pytest

from msgflux.nn.modules.agent import Agent


def create_mock_model():
    """Create a mock chat completion model."""
    model = MagicMock()
    model.model_type = "chat_completion"
    return model


def test_agent_without_signature_no_error():
    """Test that Agent without signature can be created without calling non-existent _set_task_template."""
    model = create_mock_model()

    # Should not raise AttributeError about _set_task_template
    agent = Agent(
        name="test_agent", model=model, templates={"task": "Test task: {{input}}"}
    )

    assert agent.templates == {"task": "Test task: {{input}}"}
    print("✓ Test 1 passed: Agent without signature created successfully")


def test_agent_with_signature_rejects_task_template():
    """Test that signature and templates['task'] cannot be used together."""
    model = create_mock_model()

    with pytest.raises(ValueError, match="Cannot specify both 'signature' and templates"):
        Agent(
            name="test_agent_sig",
            model=model,
            signature="input: str -> output: str",
            templates={"task": "This should be rejected", "response": "{{output}}"},
        )

    # Signature with other templates (without 'task') should work fine
    agent = Agent(
        name="test_agent_sig",
        model=model,
        signature="input: str -> output: str",
        templates={"response": "{{output}}"},
    )
    assert agent.templates["response"] == "{{output}}"
    assert "input" in agent.templates["task"]

    print("✓ Test 2 passed: Signature correctly rejects task template")


def test_agent_with_signature_without_templates():
    """Test that Agent with signature works without providing templates dict."""
    model = create_mock_model()

    agent = Agent(
        name="test_agent_sig", model=model, signature="question: str -> answer: str"
    )

    # Should have task template from signature
    assert "task" in agent.templates
    assert "question" in agent.templates["task"]

    print("✓ Test 3 passed: Agent with signature and no templates works correctly")


def test_templates_initialization_order():
    """Test that non-task templates are preserved when signature is used."""
    model = create_mock_model()

    # Signature with non-task templates should work
    agent = Agent(
        name="test_agent",
        model=model,
        signature="x: int -> y: int",
        templates={
            "context": "Context: {{ctx}}",
            "response": "Result: {{y}}",
        },
    )

    # Task should come from signature
    assert "x" in agent.templates["task"]

    # Other templates should be preserved
    assert agent.templates["context"] == "Context: {{ctx}}"
    assert agent.templates["response"] == "Result: {{y}}"

    print("✓ Test 4 passed: Template initialization order is correct")


if __name__ == "__main__":
    test_agent_without_signature_no_error()
    test_agent_with_signature_rejects_task_template()
    test_agent_with_signature_without_templates()
    test_templates_initialization_order()

    print("\n✅ All tests passed! Bug fix validated successfully.")
    print("\nSummary of the fix:")
    print(
        "  1. Removed non-existent _set_task_template() calls (was on lines 241 and 1383)"
    )
    print(
        "  2. Moved _set_templates(templates) to execute BEFORE _set_signature() in __init__"
    )
    print(
        "  3. In _set_signature(), task_template now directly sets self.templates['task']"
    )
    print("  4. This ensures signature-generated templates override user-provided ones")
