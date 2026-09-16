"""Tests for the W&B Serverless Inference OpenAI-compatible provider."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.models._chat_transport import EndpointMockTransport


@pytest.fixture(autouse=True)
def wandb_env(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setenv("WANDB_BASE_URL", "https://api.inference.wandb.ai/v1")


@pytest.fixture
def mock_wandb_client():
    from msgflux.models.providers.wandb import WandBChatCompletion

    client = MagicMock()
    async_client = MagicMock()
    transport = EndpointMockTransport(client.return_value, async_client.return_value)
    with patch.object(WandBChatCompletion, "chat_transport", transport):
        yield client


def test_wandb_defaults_to_chat_completions():
    from msgflux.models.providers.wandb import WandBChatCompletion

    model = WandBChatCompletion(model_id="openai/gpt-oss-120b")

    assert model.provider == "wandb"
    assert model.api_mode == "chat_completions"


def test_wandb_reads_base_url_and_api_key():
    from msgflux.models.providers.wandb import WandBChatCompletion

    model = WandBChatCompletion(model_id="openai/gpt-oss-120b")

    assert model._get_base_url() == "https://api.inference.wandb.ai/v1"
    assert model._get_api_key() == "test-key"


def test_wandb_missing_api_key_raises(monkeypatch):
    from msgflux.models.providers.wandb import WandBChatCompletion

    monkeypatch.delenv("WANDB_API_KEY")

    with pytest.raises(ValueError, match="WANDB_API_KEY"):
        WandBChatCompletion(model_id="openai/gpt-oss-120b")


def test_wandb_models_registered():
    from msgflux.models.registry import model_registry

    assert "wandb" in model_registry.get("chat_completion", {})


def test_wandb_resolves_through_model_factory():
    import msgflux as mf

    model = mf.Model.chat_completion("wandb/openai/gpt-oss-120b")

    assert model.provider == "wandb"
    assert model.model_id == "openai/gpt-oss-120b"


def test_wandb_chat_extracts_reasoning_field(mock_wandb_client):
    from msgflux.models.providers.wandb import WandBChatCompletion

    mock_wandb_client.return_value.chat.completions.create.return_value = (
        SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content="OK",
                        reasoning="Checking the request.",
                        tool_calls=None,
                        audio=None,
                        annotations=None,
                    ),
                )
            ],
        )
    )
    model = WandBChatCompletion(model_id="openai/gpt-oss-120b")
    response = model("Reply with exactly: OK")

    assert response.consume() == "OK"
    assert response.reasoning == "Checking the request."


def test_wandb_api_key_env_override(monkeypatch):
    from msgflux.models.providers.wandb import WandBChatCompletion

    monkeypatch.setenv("WANDB_ACME_KEY", "wandb-acme-key")
    model = WandBChatCompletion(
        model_id="openai/gpt-oss-120b", api_key_env="WANDB_ACME_KEY"
    )

    assert model._get_api_key() == "wandb-acme-key"
