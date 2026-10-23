"""Tests for the Meta Model API OpenAI-compatible provider."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.models._chat_transport import EndpointMockTransport


@pytest.fixture(autouse=True)
def meta_env(monkeypatch):
    monkeypatch.setenv("META_API_KEY", "test-key")
    monkeypatch.setenv("META_BASE_URL", "https://api.meta.ai/v1")


@pytest.fixture
def mock_meta_client():
    from msgflux.models.providers.meta import MetaChatCompletion

    client = MagicMock()
    async_client = MagicMock()
    transport = EndpointMockTransport(client.return_value, async_client.return_value)
    with patch.object(MetaChatCompletion, "chat_transport", transport):
        yield client


def test_meta_defaults_to_responses_api_mode():
    from msgflux.models.providers.meta import MetaChatCompletion

    model = MetaChatCompletion(model_id="muse-spark-1.3")

    assert model.provider == "meta"
    assert model.api_mode == "responses"


def test_meta_defaults_to_direct_chat_transport():
    from msgflux.models.chat_transport import HTTPChatTransport
    from msgflux.models.providers.meta import MetaChatCompletion

    model = MetaChatCompletion(model_id="muse-spark-1.3")

    assert isinstance(model.chat_transport, HTTPChatTransport)


def test_meta_reads_base_url_and_api_key():
    from msgflux.models.providers.meta import MetaChatCompletion

    model = MetaChatCompletion(model_id="muse-spark-1.3")

    assert model._get_base_url() == "https://api.meta.ai/v1"
    assert model._get_api_key() == "test-key"


def test_meta_missing_api_key_raises(monkeypatch):
    from msgflux.models.providers.meta import MetaChatCompletion

    monkeypatch.delenv("META_API_KEY")

    with pytest.raises(ValueError, match="META_API_KEY"):
        MetaChatCompletion(model_id="muse-spark-1.3")


def test_meta_models_registered():
    from msgflux.models.registry import model_registry

    assert "meta" in model_registry.get("chat_completion", {})


def test_meta_chat_completions_sends_reasoning_effort(mock_meta_client):
    from msgflux.models.providers.meta import MetaChatCompletion

    mock_meta_client.return_value.chat.completions.create.return_value = (
        SimpleNamespace(
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
    )
    model = MetaChatCompletion(
        model_id="muse-spark-1.3",
        api_mode="chat_completions",
        reasoning_effort="high",
    )
    response = model("Prove that the square root of 2 is irrational.")

    request = mock_meta_client.return_value.chat.completions.create.call_args.kwargs
    assert request["reasoning_effort"] == "high"
    assert response.consume() == "done"


def test_meta_responses_nests_reasoning_effort(mock_meta_client):
    from msgflux.models.providers.meta import MetaChatCompletion

    mock_meta_client.return_value.responses.create.return_value = SimpleNamespace(
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
    model = MetaChatCompletion(
        model_id="muse-spark-1.3",
        api_mode="responses",
        reasoning_effort="low",
    )
    response = model("Compare 17 and 17.")

    request = mock_meta_client.return_value.responses.create.call_args.kwargs
    assert request["reasoning"] == {"effort": "low", "summary": "auto"}
    assert response.consume() == "They match."


def test_meta_rejects_parallel_sampling():
    from msgflux.models.providers.meta import MetaChatCompletion

    model = MetaChatCompletion(model_id="muse-spark-1.3", api_mode="chat_completions")

    with pytest.raises(ValueError, match="`n=1`"):
        model._adapt_params({"model": "muse-spark-1.3", "n": 2})


def test_meta_rejects_audio_modalities():
    from msgflux.models.providers.meta import MetaChatCompletion

    model = MetaChatCompletion(model_id="muse-spark-1.3", api_mode="chat_completions")

    with pytest.raises(ValueError, match="`modalities`"):
        model._adapt_params({"model": "muse-spark-1.3", "modalities": ["text"]})
    with pytest.raises(ValueError, match="`audio`"):
        model._adapt_params({"model": "muse-spark-1.3", "audio": {"voice": "x"}})


def test_api_key_env_override(monkeypatch):
    from msgflux.models.providers.meta import MetaChatCompletion

    monkeypatch.setenv("META_ACME_KEY", "meta-acme-key")
    model = MetaChatCompletion(model_id="muse-spark-1.3", api_key_env="META_ACME_KEY")

    assert model.api_key_env == "META_ACME_KEY"
    assert model._get_api_key() == "meta-acme-key"
