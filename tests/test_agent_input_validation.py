from typing import Optional
from unittest.mock import MagicMock

import pytest

import msgflux as mf
from msgflux import Message
from msgflux.dsl.signature import Signature
from msgflux.nn.modules.agent import Agent


def create_mock_model():
    model = MagicMock()
    model.model_type = "chat_completion"
    return model


def test_validate_inputs_accepts_valid_named_kwargs_and_rejects_type():
    agent = Agent(
        name="test",
        model=create_mock_model(),
        signature="query: str, max_results: int -> answer: str",
        config={"validate_inputs": True},
    )

    agent._prepare_inputs(query="python", max_results=3)

    with pytest.raises(ValueError, match="max_results"):
        agent._prepare_inputs(query="python", max_results="many")


def test_validate_inputs_rejects_missing_required_field():
    agent = Agent(
        name="test",
        model=create_mock_model(),
        signature="query: str, max_results: int -> answer: str",
        config={"validate_inputs": True},
    )

    with pytest.raises(ValueError, match="max_results"):
        agent._prepare_inputs(query="python")


def test_validate_inputs_allows_missing_optional_field():
    class QA(Signature):
        question: str = mf.InputField()
        context: Optional[str] = mf.InputField()
        answer: str = mf.OutputField()

    agent = Agent(
        name="test",
        model=create_mock_model(),
        signature=QA,
        config={"validate_inputs": True},
    )

    agent._prepare_inputs(question="What is Python?")
    agent._prepare_inputs(question="What is Python?", context=None)


def test_validate_inputs_allows_missing_pep604_optional_field():
    class QA(Signature):
        question: str = mf.InputField()
        context: str | None = mf.InputField()
        answer: str = mf.OutputField()

    agent = Agent(
        name="test",
        model=create_mock_model(),
        signature=QA,
        config={"validate_inputs": True},
    )

    agent._prepare_inputs(question="What is Python?")
    agent._prepare_inputs(question="What is Python?", context=None)


def test_validate_inputs_rejects_invalid_single_scalar_task():
    agent = Agent(
        name="test",
        model=create_mock_model(),
        signature="count: int -> answer: str",
        config={"validate_inputs": True},
    )

    agent._prepare_inputs(3)

    with pytest.raises(ValueError, match="count"):
        agent._prepare_inputs("many")


def test_validate_inputs_uses_task_extracted_from_message_fields():
    agent = Agent(
        name="test",
        model=create_mock_model(),
        signature="query: str -> answer: str",
        message_fields={"task": "content"},
        config={"validate_inputs": True},
    )

    agent._prepare_inputs(Message(content={"query": "python"}))

    with pytest.raises(ValueError, match="query"):
        agent._prepare_inputs(Message(content={"query": 123}))


def test_validate_inputs_config_must_be_bool():
    with pytest.raises(TypeError, match="validate_inputs"):
        Agent(
            name="test",
            model=create_mock_model(),
            signature="query: str -> answer: str",
            config={"validate_inputs": "yes"},
        )
