"""Tests for the Vercel AI Gateway OpenAI-compatible provider."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from msgflux.chat_messages import ChatMessages
from tests.models._chat_transport import EndpointMockTransport


@pytest.fixture(autouse=True)
def vercel_env(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key")
    monkeypatch.setenv("AI_GATEWAY_BASE_URL", "https://ai-gateway.vercel.sh/v1")


@pytest.fixture
def mock_vercel_client():
    from msgflux.models.providers.vercel import VercelChatCompletion

    client = MagicMock()
    async_client = MagicMock()
    transport = EndpointMockTransport(client.return_value, async_client.return_value)
    with patch.object(VercelChatCompletion, "chat_transport", transport):
        yield client


def test_vercel_defaults_to_chat_completions():
    from msgflux.models.providers.vercel import VercelChatCompletion

    model = VercelChatCompletion(model_id="anthropic/claude-opus-5")

    assert model.provider == "vercel"
    assert model.api_mode == "chat_completions"
    assert model.supported_api_modes == ("chat_completions", "responses")


def test_vercel_reads_base_url_and_api_key():
    from msgflux.models.providers.vercel import VercelChatCompletion

    model = VercelChatCompletion(model_id="anthropic/claude-opus-5")

    assert model._get_base_url() == "https://ai-gateway.vercel.sh/v1"
    assert model._get_api_key() == "test-key"


def test_vercel_missing_api_key_raises(monkeypatch):
    from msgflux.models.providers.vercel import VercelChatCompletion

    monkeypatch.delenv("AI_GATEWAY_API_KEY")

    with pytest.raises(ValueError, match="AI_GATEWAY_API_KEY"):
        VercelChatCompletion(model_id="anthropic/claude-opus-5")


def test_vercel_models_registered():
    from msgflux.models.registry import model_registry

    assert "vercel" in model_registry.get("chat_completion", {})


def test_vercel_resolves_nested_model_id_through_factory():
    import msgflux as mf

    model = mf.Model.chat_completion("vercel/anthropic/claude-opus-5")

    assert model.provider == "vercel"
    assert model.model_id == "anthropic/claude-opus-5"


def test_vercel_responses_reasoning_summary_shape(mock_vercel_client):
    from msgflux.models.providers.vercel import VercelChatCompletion

    reasoning_item = {
        "type": "reasoning",
        "id": "rs_1",
        "status": "completed",
        "summary": [{"type": "summary_text", "text": "Checking the request."}],
    }
    mock_vercel_client.return_value.responses.create.return_value = SimpleNamespace(
        id="resp_1",
        status="completed",
        incomplete_details=None,
        usage=None,
        output=[
            reasoning_item,
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "OK"}],
            },
        ],
    )
    model = VercelChatCompletion(model_id="openai/gpt-6-astra", api_mode="responses")
    response = model("Reply with exactly: OK")

    assert response.consume() == "OK"
    assert response.reasoning_summary == "Checking the request."


def test_vercel_reasoning_round_trip(mock_vercel_client):
    from msgflux.models.providers.vercel import VercelChatCompletion

    details = [
        {
            "type": "reasoning.text",
            "text": "Checking the request.",
            "signature": "gateway-sig",
            "format": "anthropic-claude-v1",
            "index": 0,
        }
    ]
    mock_vercel_client.return_value.chat.completions.create.return_value = (
        SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content="OK",
                        reasoning="Checking the request.",
                        reasoning_details=details,
                        tool_calls=None,
                        audio=None,
                        annotations=None,
                    ),
                )
            ],
        )
    )
    model = VercelChatCompletion(
        model_id="anthropic/claude-opus-5", reasoning_effort="medium"
    )
    response = model("Reply with exactly: OK")

    assert response.consume() == "OK"
    assert response.reasoning == "Checking the request."
    assert response.history_items[0]["provider_state"] == {
        "provider": "vercel",
        "api_mode": "chat_completions",
        "codec": "vercel_reasoning_details",
        "data": details,
    }

    request = mock_vercel_client.return_value.chat.completions.create.call_args.kwargs
    assert request["reasoning_effort"] == "medium"


def test_vercel_replays_reasoning_and_details():
    from msgflux.models.providers.vercel import VercelChatCompletion

    details = [{"type": "reasoning.encrypted", "data": "opaque"}]
    model = VercelChatCompletion(model_id="openai/gpt-6-astra")
    messages = ChatMessages()
    messages.add_reasoning(
        "summary",
        provider="vercel",
        provider_state=details,
    )
    messages.add_assistant("answer")
    messages.add_user("follow up")

    params = model._build_generation_params(
        messages,
        system_prompt=None,
        prefilling=None,
        tool_catalog=None,
    )

    assert params["messages"] == [
        {
            "role": "assistant",
            "content": "answer",
            "reasoning": "summary",
            "reasoning_details": details,
        },
        {"role": "user", "content": "follow up"},
    ]


def test_vercel_stream_extracts_reasoning_text_and_state():
    from msgflux.models.providers.vercel import VercelChatCompletion

    model = VercelChatCompletion(model_id="openai/gpt-6-astra")
    delta = SimpleNamespace(
        content=None,
        reasoning="Checking.",
        reasoning_details=[{"type": "reasoning.text", "text": "Checking."}],
        tool_calls=None,
    )

    assert model._extract_reasoning(delta) == "Checking."
    assert model._extract_reasoning_state(delta) == [
        {"type": "reasoning.text", "text": "Checking."}
    ]


def test_vercel_state_does_not_replay_for_other_providers(monkeypatch):
    from msgflux.models.providers.openai import OpenAIChatCompletion
    from msgflux.models.providers.vercel import VercelChatCompletion

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    vercel = VercelChatCompletion(model_id="openai/gpt-6-astra")
    other = OpenAIChatCompletion(model_id="gpt-6-astra", api_mode="chat_completions")
    messages = ChatMessages(
        [
            {
                "type": "reasoning",
                "role": "assistant",
                "text": "summary",
                "provider_state": {
                    "provider": "vercel",
                    "api_mode": "chat_completions",
                    "codec": "vercel_reasoning_details",
                    "data": [{"type": "reasoning.encrypted", "data": "opaque"}],
                },
            },
            {"role": "assistant", "content": "answer"},
        ]
    )

    replay = messages.to_chatml(
        provider="openai",
        api_mode="chat_completions",
        reasoning_codec=other.reasoning_codec,
    )

    assert replay == [{"role": "assistant", "content": "answer"}]
    assert vercel.reasoning_codec.name == "vercel_reasoning_details"


def test_vercel_api_key_env_override(monkeypatch):
    from msgflux.models.providers.vercel import VercelChatCompletion

    monkeypatch.setenv("VERCEL_ACME_KEY", "vercel-acme-key")
    model = VercelChatCompletion(
        model_id="anthropic/claude-opus-5", api_key_env="VERCEL_ACME_KEY"
    )

    assert model._get_api_key() == "vercel-acme-key"
