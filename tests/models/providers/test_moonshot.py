"""Tests for the Moonshot Kimi OpenAI-compatible provider."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.models._chat_transport import EndpointMockTransport


@pytest.fixture(autouse=True)
def moonshot_env(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    monkeypatch.setenv("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1")


@pytest.fixture
def mock_moonshot_client():
    from msgflux.models.providers.moonshot import MoonshotChatCompletion

    client = MagicMock()
    async_client = MagicMock()
    transport = EndpointMockTransport(client.return_value, async_client.return_value)
    with patch.object(MoonshotChatCompletion, "chat_transport", transport):
        yield client


def test_moonshot_defaults_to_chat_completions():
    from msgflux.models.providers.moonshot import MoonshotChatCompletion

    model = MoonshotChatCompletion(model_id="kimi-k2.6")

    assert model.provider == "moonshot"
    assert model.api_mode == "chat_completions"


def test_moonshot_reads_base_url_and_api_key():
    from msgflux.models.providers.moonshot import MoonshotChatCompletion

    model = MoonshotChatCompletion(model_id="kimi-k2.6")

    assert model._get_base_url() == "https://api.moonshot.ai/v1"
    assert model._get_api_key() == "test-key"


def test_moonshot_missing_api_key_raises(monkeypatch):
    from msgflux.models.providers.moonshot import MoonshotChatCompletion

    monkeypatch.delenv("MOONSHOT_API_KEY")

    with pytest.raises(ValueError, match="MOONSHOT_API_KEY"):
        MoonshotChatCompletion(model_id="kimi-k2.6")


def test_moonshot_models_registered():
    from msgflux.models.registry import model_registry

    assert "moonshot" in model_registry.get("chat_completion", {})


def test_moonshot_resolves_through_model_factory():
    import msgflux as mf

    model = mf.Model.chat_completion("moonshot/kimi-k2.6")

    assert model.provider == "moonshot"
    assert model.model_id == "kimi-k2.6"


def test_moonshot_chat_round_trip(mock_moonshot_client):
    from msgflux.models.providers.moonshot import MoonshotChatCompletion

    mock_moonshot_client.return_value.chat.completions.create.return_value = (
        SimpleNamespace(
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
    )
    model = MoonshotChatCompletion(model_id="kimi-k2.6")
    response = model("Reply with exactly: OK")

    assert response.consume() == "OK"
    assert response.reasoning == "Checking the request."


def test_moonshot_api_key_env_override(monkeypatch):
    from msgflux.models.providers.moonshot import MoonshotChatCompletion

    monkeypatch.setenv("MOONSHOT_ACME_KEY", "moonshot-acme-key")
    model = MoonshotChatCompletion(
        model_id="kimi-k2.6", api_key_env="MOONSHOT_ACME_KEY"
    )

    assert model._get_api_key() == "moonshot-acme-key"
