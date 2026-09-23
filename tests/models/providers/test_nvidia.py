"""Tests for the NVIDIA NIM OpenAI-compatible provider."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.models._chat_transport import EndpointMockTransport


@pytest.fixture(autouse=True)
def nvidia_env(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")


@pytest.fixture
def mock_nvidia_client():
    from msgflux.models.providers.nvidia import NVIDIAChatCompletion

    client = MagicMock()
    async_client = MagicMock()
    transport = EndpointMockTransport(client.return_value, async_client.return_value)
    with patch.object(NVIDIAChatCompletion, "chat_transport", transport):
        yield client


def test_nvidia_defaults_to_chat_completions():
    from msgflux.models.providers.nvidia import NVIDIAChatCompletion

    model = NVIDIAChatCompletion(model_id="z-ai/glm-5.3")

    assert model.provider == "nvidia"
    assert model.api_mode == "chat_completions"


def test_nvidia_reads_base_url_and_api_key():
    from msgflux.models.providers.nvidia import NVIDIAChatCompletion

    model = NVIDIAChatCompletion(model_id="z-ai/glm-5.3")

    assert model._get_base_url() == "https://integrate.api.nvidia.com/v1"
    assert model._get_api_key() == "test-key"


def test_nvidia_missing_api_key_raises(monkeypatch):
    from msgflux.models.providers.nvidia import NVIDIAChatCompletion

    monkeypatch.delenv("NVIDIA_API_KEY")

    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        NVIDIAChatCompletion(model_id="z-ai/glm-5.3")


def test_nvidia_models_registered():
    from msgflux.models.registry import model_registry

    assert "nvidia" in model_registry.get("chat_completion", {})


def test_nvidia_resolves_through_model_factory():
    import msgflux as mf

    model = mf.Model.chat_completion("nvidia/z-ai/glm-5.3")

    assert model.provider == "nvidia"
    assert model.api_mode == "chat_completions"


def test_nvidia_extracts_reasoning_content(mock_nvidia_client):
    from msgflux.models.providers.nvidia import NVIDIAChatCompletion

    mock_nvidia_client.return_value.chat.completions.create.return_value = (
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
    model = NVIDIAChatCompletion(model_id="z-ai/glm-5.3")
    response = model("Reply with exactly: OK")

    assert response.consume() == "OK"
    assert response.reasoning == "Checking the request."


def test_nvidia_api_key_env_override(monkeypatch):
    from msgflux.models.providers.nvidia import NVIDIAChatCompletion

    monkeypatch.setenv("NVIDIA_ACME_KEY", "nvidia-acme-key")
    model = NVIDIAChatCompletion(model_id="z-ai/glm-5.3", api_key_env="NVIDIA_ACME_KEY")

    assert model._get_api_key() == "nvidia-acme-key"
