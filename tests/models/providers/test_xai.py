"""Tests for the xAI Grok OpenAI-compatible provider."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from msgflux.runtime.context import thread_context
from tests.models._chat_transport import EndpointMockTransport


@pytest.fixture(autouse=True)
def xai_env(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    monkeypatch.setenv("XAI_BASE_URL", "https://api.x.ai/v1")


@pytest.fixture
def mock_xai_client():
    from msgflux.models.providers.xai import XAIChatCompletion

    client = MagicMock()
    async_client = MagicMock()
    transport = EndpointMockTransport(client.return_value, async_client.return_value)
    with patch.object(XAIChatCompletion, "chat_transport", transport):
        yield client


def test_xai_defaults_to_responses_api_mode():
    from msgflux.models.providers.xai import XAIChatCompletion

    model = XAIChatCompletion(model_id="grok-4.6")

    assert model.provider == "xai"
    assert model.api_mode == "responses"


def test_xai_defaults_to_direct_chat_transport():
    from msgflux.models.chat_transport import HTTPChatTransport
    from msgflux.models.providers.xai import XAIChatCompletion

    model = XAIChatCompletion(model_id="grok-4.6")

    assert isinstance(model.chat_transport, HTTPChatTransport)


def test_xai_reads_base_url_and_api_key():
    from msgflux.models.providers.xai import XAIChatCompletion

    model = XAIChatCompletion(model_id="grok-4.6")

    assert model._get_base_url() == "https://api.x.ai/v1"
    assert model._get_api_key() == "test-key"


def test_xai_missing_api_key_raises(monkeypatch):
    from msgflux.models.providers.xai import XAIChatCompletion

    monkeypatch.delenv("XAI_API_KEY")

    with pytest.raises(ValueError, match="XAI_API_KEY"):
        XAIChatCompletion(model_id="grok-4.6")


def test_xai_models_registered():
    from msgflux.models.registry import model_registry

    assert "xai" in model_registry.get("chat_completion", {})


def test_xai_resolves_through_model_factory():
    import msgflux as mf

    model = mf.Model.chat_completion("xai/grok-4.6")

    assert model.provider == "xai"
    assert model.api_mode == "responses"


def test_xai_chat_completions_sends_reasoning_effort(mock_xai_client):
    from msgflux.models.providers.xai import XAIChatCompletion

    mock_xai_client.return_value.chat.completions.create.return_value = SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    content="done",
                    tool_calls=None,
                    audio=None,
                    annotations=None,
                ),
            )
        ],
    )
    model = XAIChatCompletion(
        model_id="grok-4.6",
        api_mode="chat_completions",
        reasoning_effort="high",
    )
    response = model("Prove that the square root of 2 is irrational.")

    request = mock_xai_client.return_value.chat.completions.create.call_args.kwargs
    assert request["reasoning_effort"] == "high"
    assert response.consume() == "done"


def test_xai_responses_nests_reasoning_effort(mock_xai_client):
    from msgflux.models.providers.xai import XAIChatCompletion

    mock_xai_client.return_value.responses.create.return_value = SimpleNamespace(
        id="resp_1",
        status="completed",
        incomplete_details=None,
        usage=None,
        output=[
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "They match."}],
            },
        ],
    )
    model = XAIChatCompletion(
        model_id="grok-4.6",
        api_mode="responses",
        reasoning_effort="low",
    )
    response = model("Compare 17 and 17.")

    request = mock_xai_client.return_value.responses.create.call_args.kwargs
    assert request["reasoning"] == {"effort": "low", "summary": "auto"}
    assert response.consume() == "They match."


def _chat_completion_ok(client):
    client.return_value.chat.completions.create.return_value = SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    content="done",
                    tool_calls=None,
                    audio=None,
                    annotations=None,
                ),
            )
        ],
    )


def _responses_ok(client):
    client.return_value.responses.create.return_value = SimpleNamespace(
        id="resp_1",
        status="completed",
        incomplete_details=None,
        usage=None,
        output=[
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done"}],
            },
        ],
    )


def test_chat_sends_conv_id_from_active_thread(mock_xai_client):
    from msgflux.models.providers.xai import XAIChatCompletion

    _chat_completion_ok(mock_xai_client)

    with thread_context(thread_id="thread_1"):
        XAIChatCompletion(model_id="grok-4.6", api_mode="chat_completions")("Hello")

    headers = mock_xai_client.return_value.chat.completions.create.call_args.kwargs[
        "extra_headers"
    ]
    assert headers == {"User-Agent": "msgflux", "x-grok-conv-id": "thread_1"}


def test_responses_sends_cache_key_from_active_thread(mock_xai_client):
    from msgflux.models.providers.xai import XAIChatCompletion

    _responses_ok(mock_xai_client)

    with thread_context(thread_id="thread_1"):
        XAIChatCompletion(model_id="grok-4.6", api_mode="responses")("Hello")

    request = mock_xai_client.return_value.responses.create.call_args.kwargs
    assert request["prompt_cache_key"] == "thread_1"
    assert request["extra_headers"] == {"User-Agent": "msgflux"}


def test_no_conv_id_without_active_thread(mock_xai_client):
    from msgflux.models.providers.xai import XAIChatCompletion

    _chat_completion_ok(mock_xai_client)

    XAIChatCompletion(model_id="grok-4.6", api_mode="chat_completions")("Hello")
    headers = mock_xai_client.return_value.chat.completions.create.call_args.kwargs[
        "extra_headers"
    ]
    assert headers == {"User-Agent": "msgflux"}
