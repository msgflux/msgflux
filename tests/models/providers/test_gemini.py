"""Tests for the Google Gemini OpenAI-compatible provider."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from msgflux.chat_messages import ChatMessages
from tests.models._chat_transport import EndpointMockTransport


@pytest.fixture(autouse=True)
def gemini_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv(
        "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"
    )


@pytest.fixture
def mock_gemini_client():
    from msgflux.models.providers.gemini import GeminiChatCompletion

    client = MagicMock()
    async_client = MagicMock()
    transport = EndpointMockTransport(client.return_value, async_client.return_value)
    with patch.object(GeminiChatCompletion, "chat_transport", transport):
        yield client


def _message(content="OK", signature="sig-1", thought=False):
    extra = {"google": {"thought_signature": signature}}
    if thought:
        extra["google"]["thought"] = True
    return SimpleNamespace(
        content=content,
        extra_content=SimpleNamespace(
            google=SimpleNamespace(thought_signature=signature, thought=thought)
            if thought
            else SimpleNamespace(thought_signature=signature)
        ),
        tool_calls=None,
        audio=None,
        annotations=None,
    )


def test_gemini_defaults_to_chat_completions():
    from msgflux.models.providers.gemini import GeminiChatCompletion

    model = GeminiChatCompletion(model_id="gemini-3.6-flash")

    assert model.provider == "gemini"
    assert model.api_mode == "chat_completions"
    assert model.supported_api_modes == ("chat_completions",)


def test_gemini_reads_base_url_and_api_key():
    from msgflux.models.providers.gemini import GeminiChatCompletion

    model = GeminiChatCompletion(model_id="gemini-3.6-flash")

    assert (
        model._get_base_url()
        == "https://generativelanguage.googleapis.com/v1beta/openai"
    )
    assert model._get_api_key() == "test-key"


def test_gemini_missing_api_key_raises(monkeypatch):
    from msgflux.models.providers.gemini import GeminiChatCompletion

    monkeypatch.delenv("GEMINI_API_KEY")

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiChatCompletion(model_id="gemini-3.6-flash")


def test_gemini_models_registered():
    from msgflux.models.registry import model_registry

    assert "gemini" in model_registry.get("chat_completion", {})


def test_gemini_resolves_through_model_factory():
    import msgflux as mf

    model = mf.Model.chat_completion("gemini/gemini-3.6-flash")

    assert model.provider == "gemini"
    assert model.model_id == "gemini-3.6-flash"


def test_gemini_rejects_responses_api_mode():
    from msgflux.models.providers.gemini import GeminiChatCompletion

    with pytest.raises(ValueError, match="responses"):
        GeminiChatCompletion(model_id="gemini-3.6-flash", api_mode="responses")


def test_gemini_round_trip_keeps_signature_state(mock_gemini_client):
    from msgflux.models.providers.gemini import GeminiChatCompletion

    mock_gemini_client.return_value.chat.completions.create.return_value = (
        SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=_message(content="OK", signature="sig-1"),
                )
            ],
        )
    )
    model = GeminiChatCompletion(model_id="gemini-3.6-flash", reasoning_effort="low")
    response = model("Reply with exactly: OK")

    assert response.consume() == "OK"
    assert response.reasoning is None
    assert response.history_items == [
        {
            "type": "reasoning",
            "role": "assistant",
            "provider_state": {
                "provider": "gemini",
                "api_mode": "chat_completions",
                "codec": "gemini_thought_signature",
                "data": "sig-1",
            },
        }
    ]

    request = mock_gemini_client.return_value.chat.completions.create.call_args.kwargs
    assert request["reasoning_effort"] == "low"
    assert request["extra_headers"]["x-goog-api-client"].startswith("msgflux-oai/")


def test_gemini_splits_inline_thought_summary(mock_gemini_client):
    from msgflux.models.providers.gemini import GeminiChatCompletion

    mock_gemini_client.return_value.chat.completions.create.return_value = (
        SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=_message(
                        content="<thought>Checking digits.</thought>OK",
                        signature="sig-2",
                    ),
                )
            ],
        )
    )
    model = GeminiChatCompletion(model_id="gemini-3.6-flash")
    response = model("Reply with exactly: OK")

    assert response.consume() == "OK"
    assert response.reasoning == "Checking digits."
    assert response.history_items[0]["text"] == "Checking digits."
    assert response.history_items[0]["provider_state"]["data"] == "sig-2"


def test_gemini_replays_signature_on_assistant_message():
    from msgflux.models.providers.gemini import GeminiChatCompletion

    model = GeminiChatCompletion(model_id="gemini-3.6-flash")
    messages = ChatMessages(
        [
            {
                "type": "reasoning",
                "role": "assistant",
                "provider_state": {
                    "provider": "gemini",
                    "api_mode": "chat_completions",
                    "codec": "gemini_thought_signature",
                    "data": "sig-1",
                },
            },
            {"role": "assistant", "content": "OK"},
            {"role": "user", "content": "follow up"},
        ]
    )

    params = model._build_generation_params(
        messages,
        system_prompt=None,
        prefilling=None,
        tool_catalog=None,
    )

    assert params["messages"] == [
        {
            "role": "assistant",
            "content": "OK",
            "extra_content": {"google": {"thought_signature": "sig-1"}},
        },
        {"role": "user", "content": "follow up"},
    ]


def test_gemini_tool_calls_keep_per_call_signatures(mock_gemini_client):
    from msgflux.models.providers.gemini import GeminiChatCompletion

    tool_call = SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name="get_weather", arguments='{"city":"Paris"}'),
        extra_content=SimpleNamespace(
            google=SimpleNamespace(thought_signature="sig-call-1")
        ),
    )
    mock_gemini_client.return_value.chat.completions.create.return_value = (
        SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="tool_calls",
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[tool_call],
                        audio=None,
                        annotations=None,
                    ),
                )
            ],
        )
    )
    model = GeminiChatCompletion(model_id="gemini-3.6-flash", reasoning_effort="low")
    response = model("What is the weather in Paris?")

    assert response.response_type == "tool_call"
    (history_call,) = response.history_items
    assert history_call["type"] == "function_call"
    assert history_call["call_id"] == "call_1"

    replay = ChatMessages(response.history_items).to_chatml(
        provider="gemini",
        api_mode="chat_completions",
        reasoning_codec=model.reasoning_codec,
    )
    assert replay == [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "extra_content": {"google": {"thought_signature": "sig-call-1"}},
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"Paris"}',
                    },
                }
            ],
        }
    ]


def test_gemini_stream_splits_thought_tags():
    from msgflux.models.providers.gemini import GeminiChatCompletion

    model = GeminiChatCompletion(model_id="gemini-3.6-flash")
    state = {"in_thought": False}
    thought_delta = SimpleNamespace(
        content="<thought>Checking.",
        extra_content=SimpleNamespace(google=SimpleNamespace(thought=True)),
        tool_calls=None,
    )
    chunk = SimpleNamespace(choices=[SimpleNamespace(delta=thought_delta)])

    (expanded,) = model._split_gemini_chunk(chunk, state)

    assert state["in_thought"] is True
    assert expanded.choices[0].delta.content is None
    assert expanded.choices[0].delta.reasoning_content == "Checking."

    boundary = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content="</thought>OK", tool_calls=None)
            )
        ]
    )
    expanded = model._split_gemini_chunk(boundary, state)

    assert state["in_thought"] is False
    assert len(expanded) == 1
    assert expanded[0].choices[0].delta.content == "OK"


def test_gemini_stream_signature_delta_passes_through():
    from msgflux.models.providers.gemini import GeminiChatCompletion

    model = GeminiChatCompletion(model_id="gemini-3.6-flash")
    chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=None,
                    extra_content=SimpleNamespace(
                        google=SimpleNamespace(thought_signature="sig-end")
                    ),
                    tool_calls=None,
                )
            )
        ]
    )

    assert (
        model.reasoning_codec.extract_state(
            chunk.choices[0].delta,
            serialize=lambda value: value,
        )
        == "sig-end"
    )
    assert model._split_gemini_chunk(chunk, {"in_thought": False}) == [chunk]


def test_gemini_surfaces_provider_cache_hits(mock_gemini_client):
    from msgflux.models.providers.gemini import GeminiChatCompletion

    mock_gemini_client.return_value.chat.completions.create.return_value = (
        SimpleNamespace(
            usage={
                "prompt_tokens": 2000,
                "completion_tokens": 5,
                "total_tokens": 2005,
                "prompt_tokens_details": {"cached_tokens": 1500},
            },
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=_message(content="OK", signature="sig-1"),
                )
            ],
        )
    )
    model = GeminiChatCompletion(model_id="gemini-3.6-flash")
    response = model("Reply with exactly: OK")

    assert response.consume() == "OK"
    assert response.metadata.usage.input_tokens_details.cached_tokens == 1500
    assert response.metadata.usage.cache_hit_percentage == pytest.approx(75.0)


def test_gemini_api_key_env_override(monkeypatch):
    from msgflux.models.providers.gemini import GeminiChatCompletion

    monkeypatch.setenv("GEMINI_ACME_KEY", "gemini-acme-key")
    model = GeminiChatCompletion(
        model_id="gemini-3.6-flash", api_key_env="GEMINI_ACME_KEY"
    )

    assert model._get_api_key() == "gemini-acme-key"
