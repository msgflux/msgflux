"""Tests for the Groq OpenAI-compatible model provider."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from msgflux.chat_messages import ChatMessages
from tests.models._chat_transport import EndpointMockTransport


@pytest.fixture(autouse=True)
def groq_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")


@pytest.fixture
def mock_openai_client():
    from msgflux.models.providers.groq import GroqChatCompletion

    client = MagicMock()
    async_client = MagicMock()
    transport = EndpointMockTransport(client.return_value, async_client.return_value)
    with patch.object(GroqChatCompletion, "chat_transport", transport):
        yield client


def test_groq_defaults_to_direct_chat_transport():
    from msgflux.models.chat_transport import HTTPChatTransport
    from msgflux.models.providers.groq import GroqChatCompletion

    model = GroqChatCompletion(model_id="openai/gpt-oss-20b")

    assert isinstance(model.chat_transport, HTTPChatTransport)


def test_chat_completions_extracts_but_does_not_replay_reasoning(
    mock_openai_client,
):

    from msgflux.models.providers.groq import GroqChatCompletion

    model = GroqChatCompletion(model_id="openai/gpt-oss-20b")
    messages = ChatMessages()
    messages.add_reasoning("private chain")
    messages.add_assistant("first answer")
    messages.add_user("follow up")

    params = model._build_generation_params(
        messages,
        system_prompt=None,
        prefilling=None,
        tool_catalog=None,
    )

    assert params["messages"] == [
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "follow up"},
    ]


def test_responses_mode_uses_clear_text_reasoning_contract(mock_openai_client):

    from msgflux.models.providers.groq import GroqChatCompletion

    reasoning_item = {
        "type": "reasoning",
        "id": "rs_1",
        "status": "completed",
        "content": [{"type": "reasoning_text", "text": "Checked both values."}],
        "summary": [],
    }
    mock_openai_client.return_value.responses.create.return_value = SimpleNamespace(
        id="resp_1",
        status="completed",
        incomplete_details=None,
        usage=None,
        output=[
            reasoning_item,
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "They match."}],
            },
        ],
    )
    model = GroqChatCompletion(
        model_id="openai/gpt-oss-20b",
        api_mode="responses",
        reasoning_effort="low",
    )

    response = model("Compare 17 and 17.")

    request = mock_openai_client.return_value.responses.create.call_args.kwargs
    assert request["reasoning"] == {"effort": "low"}
    assert "include" not in request
    assert response.consume() == "They match."
    assert response.reasoning == "Checked both values."
    assert response.history_items == [
        {
            "type": "reasoning",
            "role": "assistant",
            "text": "Checked both values.",
            "provider_state": {
                "provider": "groq",
                "api_mode": "responses",
                "codec": "responses_reasoning_text",
                "data": {
                    "type": "reasoning",
                    "id": "rs_1",
                    "status": "completed",
                },
            },
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "They match."}],
            "provider_state": {
                "provider": "groq",
                "api_mode": "responses",
                "data": {},
            },
        },
    ]
    assert ChatMessages(response.history_items).to_responses_input(
        provider="groq",
        reasoning_codec=model.reasoning_codec,
    ) == [
        reasoning_item,
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "They match."}],
        },
    ]
