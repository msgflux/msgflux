"""Tests for the InferenceNet OpenAI-compatible provider."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.models._chat_transport import EndpointMockTransport


@pytest.fixture(autouse=True)
def inference_env(monkeypatch):
    monkeypatch.setenv("INFERENCE_API_KEY", "test-key")
    monkeypatch.setenv("INFERENCE_BASE_URL", "https://api.inference.net/v1")


@pytest.fixture
def mock_inference_client():
    from msgflux.models.providers.inference_net import InferenceNetChatCompletion

    client = MagicMock()
    async_client = MagicMock()
    transport = EndpointMockTransport(client.return_value, async_client.return_value)
    with patch.object(InferenceNetChatCompletion, "chat_transport", transport):
        yield client


def test_inference_defaults_to_chat_completions():
    from msgflux.models.providers.inference_net import InferenceNetChatCompletion

    model = InferenceNetChatCompletion(model_id="glm-5.2")

    assert model.provider == "inference-net"
    assert model.api_mode == "chat_completions"


def test_inference_reads_base_url_and_api_key():
    from msgflux.models.providers.inference_net import InferenceNetChatCompletion

    model = InferenceNetChatCompletion(model_id="glm-5.2")

    assert model._get_base_url() == "https://api.inference.net/v1"
    assert model._get_api_key() == "test-key"


def test_inference_missing_api_key_raises(monkeypatch):
    from msgflux.models.providers.inference_net import InferenceNetChatCompletion

    monkeypatch.delenv("INFERENCE_API_KEY")

    with pytest.raises(ValueError, match="INFERENCE_API_KEY"):
        InferenceNetChatCompletion(model_id="glm-5.2")


def test_inference_models_registered():
    from msgflux.models.registry import model_registry

    assert "inference-net" in model_registry.get("chat_completion", {})


def test_inference_resolves_through_model_factory():
    import msgflux as mf

    model = mf.Model.chat_completion("inference-net/glm-5.2")

    assert model.provider == "inference-net"
    assert model.api_mode == "chat_completions"


def test_inference_chat_round_trip(mock_inference_client):
    from msgflux.models.providers.inference_net import InferenceNetChatCompletion

    mock_inference_client.return_value.chat.completions.create.return_value = (
        SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content="42",
                        reasoning_content=None,
                        tool_calls=None,
                        audio=None,
                        annotations=None,
                    ),
                )
            ],
        )
    )
    model = InferenceNetChatCompletion(model_id="glm-5.2")
    response = model("What is the meaning of life?")

    request = (
        mock_inference_client.return_value.chat.completions.create.call_args.kwargs
    )
    assert request["model"] == "glm-5.2"
    assert response.consume() == "42"


def test_inference_api_key_env_override(monkeypatch):
    from msgflux.models.providers.inference_net import InferenceNetChatCompletion

    monkeypatch.setenv("INFERENCE_ACME_KEY", "inference-acme-key")
    model = InferenceNetChatCompletion(
        model_id="glm-5.2", api_key_env="INFERENCE_ACME_KEY"
    )

    assert model._get_api_key() == "inference-acme-key"
