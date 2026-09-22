"""Tests for the Fireworks OpenAI-compatible provider."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.models._chat_transport import EndpointMockTransport


@pytest.fixture(autouse=True)
def fireworks_env(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")


@pytest.fixture
def mock_fireworks_client():
    from msgflux.models.providers.fireworks import FireworksChatCompletion

    client = MagicMock()
    async_client = MagicMock()
    transport = EndpointMockTransport(client.return_value, async_client.return_value)
    with patch.object(FireworksChatCompletion, "chat_transport", transport):
        yield client


def test_fireworks_defaults_to_chat_completions():
    from msgflux.models.providers.fireworks import FireworksChatCompletion

    model = FireworksChatCompletion(model_id="accounts/fireworks/models/gpt-oss-120b")

    assert model.provider == "fireworks"
    assert model.api_mode == "chat_completions"


def test_fireworks_reads_base_url_and_api_key():
    from msgflux.models.providers.fireworks import FireworksChatCompletion

    model = FireworksChatCompletion(model_id="accounts/fireworks/models/gpt-oss-120b")

    assert model._get_base_url() == "https://api.fireworks.ai/inference/v1"
    assert model._get_api_key() == "test-key"


def test_fireworks_missing_api_key_raises(monkeypatch):
    from msgflux.models.providers.fireworks import FireworksChatCompletion

    monkeypatch.delenv("FIREWORKS_API_KEY")

    with pytest.raises(ValueError, match="FIREWORKS_API_KEY"):
        FireworksChatCompletion(model_id="accounts/fireworks/models/gpt-oss-120b")


def test_fireworks_models_registered():
    from msgflux.models.registry import model_registry

    assert "fireworks" in model_registry.get("chat_completion", {})


def test_fireworks_resolves_through_model_factory():
    import msgflux as mf

    model = mf.Model.chat_completion("fireworks/accounts/fireworks/models/gpt-oss-120b")

    assert model.provider == "fireworks"
    assert model.model_id == "accounts/fireworks/models/gpt-oss-120b"


def test_fireworks_chat_round_trip(mock_fireworks_client):
    from msgflux.models.providers.fireworks import FireworksChatCompletion

    mock_fireworks_client.return_value.chat.completions.create.return_value = (
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
    model = FireworksChatCompletion(model_id="accounts/fireworks/models/gpt-oss-120b")
    response = model("Reply with exactly: OK")

    assert response.consume() == "OK"
    assert response.reasoning == "Checking the request."


def test_fireworks_api_key_env_override(monkeypatch):
    from msgflux.models.providers.fireworks import FireworksChatCompletion

    monkeypatch.setenv("FIREWORKS_ACME_KEY", "fireworks-acme-key")
    model = FireworksChatCompletion(
        model_id="accounts/fireworks/models/gpt-oss-120b",
        api_key_env="FIREWORKS_ACME_KEY",
    )

    assert model._get_api_key() == "fireworks-acme-key"
