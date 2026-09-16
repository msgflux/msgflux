"""Tests for the DeepInfra OpenAI-compatible provider."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.models._chat_transport import EndpointMockTransport


@pytest.fixture(autouse=True)
def deepinfra_env(monkeypatch):
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-key")
    monkeypatch.setenv("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai")


@pytest.fixture
def mock_deepinfra_client():
    from msgflux.models.providers.deepinfra import DeepInfraChatCompletion

    client = MagicMock()
    async_client = MagicMock()
    transport = EndpointMockTransport(client.return_value, async_client.return_value)
    with patch.object(DeepInfraChatCompletion, "chat_transport", transport):
        yield client


def test_deepinfra_defaults_to_chat_completions():
    from msgflux.models.providers.deepinfra import DeepInfraChatCompletion

    model = DeepInfraChatCompletion(model_id="deepseek-ai/DeepSeek-V4-Flash-0731")

    assert model.provider == "deepinfra"
    assert model.api_mode == "chat_completions"


def test_deepinfra_reads_base_url_and_api_key():
    from msgflux.models.providers.deepinfra import DeepInfraChatCompletion

    model = DeepInfraChatCompletion(model_id="deepseek-ai/DeepSeek-V4-Flash-0731")

    assert model._get_base_url() == "https://api.deepinfra.com/v1/openai"
    assert model._get_api_key() == "test-key"


def test_deepinfra_missing_api_key_raises(monkeypatch):
    from msgflux.models.providers.deepinfra import DeepInfraChatCompletion

    monkeypatch.delenv("DEEPINFRA_API_KEY")

    with pytest.raises(ValueError, match="DEEPINFRA_API_KEY"):
        DeepInfraChatCompletion(model_id="deepseek-ai/DeepSeek-V4-Flash-0731")


def test_deepinfra_models_registered():
    from msgflux.models.registry import model_registry

    assert "deepinfra" in model_registry.get("chat_completion", {})


def test_deepinfra_resolves_through_model_factory():
    import msgflux as mf

    model = mf.Model.chat_completion("deepinfra/deepseek-ai/DeepSeek-V4-Flash-0731")

    assert model.provider == "deepinfra"
    assert model.model_id == "deepseek-ai/DeepSeek-V4-Flash-0731"


def test_deepinfra_chat_round_trip(mock_deepinfra_client):
    from msgflux.models.providers.deepinfra import DeepInfraChatCompletion

    mock_deepinfra_client.return_value.chat.completions.create.return_value = (
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
    model = DeepInfraChatCompletion(model_id="deepseek-ai/DeepSeek-V4-Flash-0731")
    response = model("Reply with exactly: OK")

    assert response.consume() == "OK"
    assert response.reasoning == "Checking the request."


def test_deepinfra_api_key_env_override(monkeypatch):
    from msgflux.models.providers.deepinfra import DeepInfraChatCompletion

    monkeypatch.setenv("DEEPINFRA_ACME_KEY", "deepinfra-acme-key")
    model = DeepInfraChatCompletion(
        model_id="deepseek-ai/DeepSeek-V4-Flash-0731",
        api_key_env="DEEPINFRA_ACME_KEY",
    )

    assert model._get_api_key() == "deepinfra-acme-key"
