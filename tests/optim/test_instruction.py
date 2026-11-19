"""Contract tests for the Agent task-instruction API used by optimizers."""

from unittest.mock import Mock

import msgspec
import pytest

from msgflux import nn
from msgflux.dsl.signature import InputField, OutputField, Signature


def chat_model():
    """Minimal chat-completion stand-in; the API never calls the model."""
    model = Mock()
    model.model_type = "chat_completion"
    return model


class Classify(Signature):
    """Classify the incoming request."""

    request: str = InputField(desc="user request")
    label: str = OutputField(desc="billing or technical")


def class_schema():
    class Schema(msgspec.Struct):
        system_prompt = "Schema level guidance."
        answer: str

    return Schema


@pytest.mark.parametrize("value", ["", "   ", "\n\t", None, 3, b"bytes"])
def test_set_rejects_non_string_or_blank_text(value):
    agent = nn.Agent(name="support", model=chat_model(), system_prompt="Base.")

    with pytest.raises(ValueError, match="non-empty task instruction"):
        agent.set_task_instruction(value)

    assert agent.get_task_instruction() == "Base."
    assert agent.system_prompt.data == "Base."


def test_set_preserves_signature_guidance_schema_and_expected_output():
    agent = nn.Agent(name="router", model=chat_model(), signature=Classify)
    composed_before = agent.system_prompt.data

    agent.set_task_instruction("Classify the request quickly.")

    composed_after = agent.system_prompt.data
    assert agent.get_task_instruction() == "Classify the request quickly."
    assert composed_after.startswith("Classify the request quickly.")
    # Everything the Signature contributes after the editable body is unchanged.
    tail = composed_after.split("Your task inputs are:", 1)[1]
    assert tail == composed_before.split("Your task inputs are:", 1)[1]
    assert "1. `request` (str): user request" in composed_after
    assert "1. `label` (str): billing or technical" in composed_after
    assert "Write an encoded JSON." in composed_after


def test_explicit_system_prompt_wins_over_the_signature_docstring():
    agent = nn.Agent(
        name="router",
        model=chat_model(),
        signature=Classify,
        system_prompt="Explicit operator guidance.",
    )

    assert agent.get_task_instruction() == "Explicit operator guidance."
    composed = agent.system_prompt.data
    assert composed.index("Explicit operator guidance.") < composed.index(
        "Classify the incoming request."
    )
    assert composed.index("Classify the incoming request.") < composed.index(
        "Your task outputs are:"
    )

    agent.set_task_instruction("Edited operator guidance.")

    assert agent.get_task_instruction() == "Edited operator guidance."
    assert "Edited operator guidance." in agent.system_prompt.data
    assert "Classify the incoming request." in agent.system_prompt.data
    assert "Your task outputs are:" in agent.system_prompt.data


def test_generation_schema_prompt_stays_a_prefix_across_instruction_updates():
    agent = nn.Agent(
        name="router",
        model=chat_model(),
        signature=Classify,
        generation_schema=class_schema(),
    )

    assert agent.get_task_instruction() == "Classify the incoming request."
    assert agent.system_prompt.data.startswith("Schema level guidance.")

    agent.set_task_instruction("Edited guidance.")

    composed = agent.system_prompt.data
    assert composed.startswith("Schema level guidance.")
    assert composed.index("Schema level guidance.") < composed.index("Edited guidance.")
    assert composed.index("Edited guidance.") < composed.index("Your task inputs are:")


def test_set_rejects_a_frozen_parameter():
    agent = nn.Agent(name="router", model=chat_model(), signature=Classify)
    composed_before = agent.system_prompt.data
    agent.system_prompt.requires_grad_(requires_grad=False)

    with pytest.raises(ValueError, match="frozen"):
        agent.set_task_instruction("Classify the request quickly.")

    assert agent.get_task_instruction() == "Classify the incoming request."
    assert agent.system_prompt.data == composed_before


def test_jinja_variables_must_be_preserved():
    agent = nn.Agent(
        name="support",
        model=chat_model(),
        system_prompt="Support {{ product }} customers.",
    )

    assert agent.get_task_instruction() == "Support {{ product }} customers."

    with pytest.raises(ValueError, match="preserve template variables"):
        agent.set_task_instruction("Support customers.")
    with pytest.raises(ValueError, match="preserve template variables"):
        agent.set_task_instruction("Support {{ product }} and {{ tier }} customers.")
    assert agent.system_prompt.data == "Support {{ product }} customers."

    agent.set_task_instruction("Assist {{ product }} customers politely.")

    assert agent.get_task_instruction() == "Assist {{ product }} customers politely."
    assert (
        agent.get_system_prompt(vars={"product": "Warehouse Cloud"})
        == "Assist Warehouse Cloud customers politely."
    )


def test_str_format_placeholders_must_be_preserved():
    agent = nn.Agent(
        name="support", model=chat_model(), system_prompt="Hello {}, welcome."
    )

    with pytest.raises(ValueError, match="preserve template variables"):
        agent.set_task_instruction("Hello, welcome.")

    agent.set_task_instruction("Greetings {}, welcome.")

    assert agent.system_prompt.data == "Greetings {}, welcome."


def test_declared_agent_subclass_instruction_is_editable():
    class Support(nn.Agent):
        model = chat_model()
        system_prompt = "Support {{ product }} customers."

    agent = Support()

    assert agent.get_task_instruction() == "Support {{ product }} customers."

    agent.set_task_instruction("Serve {{ product }} customers.")

    assert agent.get_task_instruction() == "Serve {{ product }} customers."
    assert agent.system_prompt.data == "Serve {{ product }} customers."
