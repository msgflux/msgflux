"""Tests for the Morph OpenAI-compatible provider."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from msgflux.runtime.context import thread_context
from tests.models._chat_transport import EndpointMockTransport


@pytest.fixture(autouse=True)
def morph_env(monkeypatch):
    monkeypatch.setenv("MORPH_API_KEY", "test-key")
    monkeypatch.setenv("MORPH_BASE_URL", "https://api.morphllm.com/v1")


@pytest.fixture
def mock_morph_client():
    from msgflux.models.providers.morph import MorphChatCompletion

    client = MagicMock()
    async_client = MagicMock()
    transport = EndpointMockTransport(client.return_value, async_client.return_value)
    with patch.object(MorphChatCompletion, "chat_transport", transport):
        yield client


def _ok(client):
    client.return_value.chat.completions.create.return_value = SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    content="OK",
                    reasoning_content="Checking the request.",
                    tool_calls=None,
                    audio=None,
                    annotations=None,
                ),
            )
        ],
    )


def test_morph_defaults_to_chat_completions():
    from msgflux.models.providers.morph import MorphChatCompletion

    model = MorphChatCompletion(model_id="morph-kimik3")

    assert model.provider == "morph"
    assert model.api_mode == "chat_completions"


def test_morph_reads_base_url_and_api_key():
    from msgflux.models.providers.morph import MorphChatCompletion

    model = MorphChatCompletion(model_id="morph-kimik3")

    assert model._get_base_url() == "https://api.morphllm.com/v1"
    assert model._get_api_key() == "test-key"


def test_morph_missing_api_key_raises(monkeypatch):
    from msgflux.models.providers.morph import MorphChatCompletion

    monkeypatch.delenv("MORPH_API_KEY")

    with pytest.raises(ValueError, match="MORPH_API_KEY"):
        MorphChatCompletion(model_id="morph-kimik3")


def test_morph_models_registered():
    from msgflux.models.registry import model_registry

    assert "morph" in model_registry.get("chat_completion", {})


def test_morph_resolves_through_model_factory():
    import msgflux as mf

    model = mf.Model.chat_completion("morph/morph-kimik3")

    assert model.provider == "morph"
    assert model.model_id == "morph-kimik3"


def test_morph_chat_round_trip(mock_morph_client):
    from msgflux.models.providers.morph import MorphChatCompletion

    _ok(mock_morph_client)
    model = MorphChatCompletion(model_id="morph-kimik3")
    response = model("Reply with exactly: OK")

    assert response.consume() == "OK"
    assert response.reasoning == "Checking the request."


def test_morph_sends_cache_key_from_active_thread(mock_morph_client):
    from msgflux.models.providers.morph import MorphChatCompletion

    _ok(mock_morph_client)

    with thread_context(thread_id="thread_1"):
        MorphChatCompletion(model_id="morph-kimik3")("Hello")

    request = mock_morph_client.return_value.chat.completions.create.call_args.kwargs
    assert request["prompt_cache_key"] == "thread_1"


def test_morph_omits_cache_key_without_active_thread(mock_morph_client):
    from msgflux.models.providers.morph import MorphChatCompletion

    _ok(mock_morph_client)
    MorphChatCompletion(model_id="morph-kimik3")("Hello")

    request = mock_morph_client.return_value.chat.completions.create.call_args.kwargs
    assert "prompt_cache_key" not in request


def test_morph_api_key_env_override(monkeypatch):
    from msgflux.models.providers.morph import MorphChatCompletion

    monkeypatch.setenv("MORPH_ACME_KEY", "morph-acme-key")
    model = MorphChatCompletion(model_id="morph-kimik3", api_key_env="MORPH_ACME_KEY")

    assert model._get_api_key() == "morph-acme-key"
