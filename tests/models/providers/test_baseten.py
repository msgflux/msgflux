"""Tests for the Baseten OpenAI-compatible provider."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from msgflux.runtime.context import thread_context
from tests.models._chat_transport import EndpointMockTransport


@pytest.fixture(autouse=True)
def baseten_env(monkeypatch):
    monkeypatch.setenv("BASETEN_API_KEY", "test-key")
    monkeypatch.setenv("BASETEN_BASE_URL", "https://inference.baseten.co/v1")


@pytest.fixture
def mock_baseten_client():
    from msgflux.models.providers.baseten import BasetenChatCompletion

    client = MagicMock()
    async_client = MagicMock()
    transport = EndpointMockTransport(client.return_value, async_client.return_value)
    with patch.object(BasetenChatCompletion, "chat_transport", transport):
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


def test_baseten_defaults_to_chat_completions():
    from msgflux.models.providers.baseten import BasetenChatCompletion

    model = BasetenChatCompletion(model_id="zai-org/GLM-5.2")

    assert model.provider == "baseten"
    assert model.api_mode == "chat_completions"


def test_baseten_reads_base_url_and_api_key():
    from msgflux.models.providers.baseten import BasetenChatCompletion

    model = BasetenChatCompletion(model_id="zai-org/GLM-5.2")

    assert model._get_base_url() == "https://inference.baseten.co/v1"
    assert model._get_api_key() == "test-key"


def test_baseten_missing_api_key_raises(monkeypatch):
    from msgflux.models.providers.baseten import BasetenChatCompletion

    monkeypatch.delenv("BASETEN_API_KEY")

    with pytest.raises(ValueError, match="BASETEN_API_KEY"):
        BasetenChatCompletion(model_id="zai-org/GLM-5.2")


def test_baseten_models_registered():
    from msgflux.models.registry import model_registry

    assert "baseten" in model_registry.get("chat_completion", {})


def test_baseten_resolves_through_model_factory():
    import msgflux as mf

    model = mf.Model.chat_completion("baseten/zai-org/GLM-5.2")

    assert model.provider == "baseten"
    assert model.model_id == "zai-org/GLM-5.2"


def test_baseten_chat_round_trip(mock_baseten_client):
    from msgflux.models.providers.baseten import BasetenChatCompletion

    _ok(mock_baseten_client)
    model = BasetenChatCompletion(model_id="zai-org/GLM-5.2")
    response = model("Reply with exactly: OK")

    assert response.consume() == "OK"
    assert response.reasoning == "Checking the request."


def test_baseten_sends_session_affinity_from_active_thread(mock_baseten_client):
    from msgflux.models.providers.baseten import BasetenChatCompletion

    _ok(mock_baseten_client)

    with thread_context(thread_id="thread_1"):
        BasetenChatCompletion(model_id="zai-org/GLM-5.2")("Hello")

    headers = mock_baseten_client.return_value.chat.completions.create.call_args.kwargs[
        "extra_headers"
    ]
    assert headers == {"User-Agent": "msgflux", "x-session-affinity": "thread_1"}


def test_baseten_omits_affinity_without_active_thread(mock_baseten_client):
    from msgflux.models.providers.baseten import BasetenChatCompletion

    _ok(mock_baseten_client)
    BasetenChatCompletion(model_id="zai-org/GLM-5.2")("Hello")

    headers = mock_baseten_client.return_value.chat.completions.create.call_args.kwargs[
        "extra_headers"
    ]
    assert headers == {"User-Agent": "msgflux"}


def test_baseten_api_key_env_override(monkeypatch):
    from msgflux.models.providers.baseten import BasetenChatCompletion

    monkeypatch.setenv("BASETEN_ACME_KEY", "baseten-acme-key")
    model = BasetenChatCompletion(
        model_id="zai-org/GLM-5.2", api_key_env="BASETEN_ACME_KEY"
    )

    assert model._get_api_key() == "baseten-acme-key"
