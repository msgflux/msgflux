"""Tests for the GMI Cloud OpenAI-compatible provider."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.models._chat_transport import EndpointMockTransport


@pytest.fixture(autouse=True)
def gmicloud_env(monkeypatch):
    monkeypatch.setenv("GMICLOUD_API_KEY", "test-key")
    monkeypatch.setenv("GMICLOUD_BASE_URL", "https://api.gmi-serving.com/v1")


@pytest.fixture
def mock_gmicloud_client():
    from msgflux.models.providers.gmicloud import GMICloudChatCompletion

    client = MagicMock()
    async_client = MagicMock()
    transport = EndpointMockTransport(client.return_value, async_client.return_value)
    with patch.object(GMICloudChatCompletion, "chat_transport", transport):
        yield client


def test_gmicloud_defaults_to_chat_completions():
    from msgflux.models.providers.gmicloud import GMICloudChatCompletion

    model = GMICloudChatCompletion(model_id="MiniMaxAI/MiniMax-M2.7")

    assert model.provider == "gmicloud"
    assert model.api_mode == "chat_completions"


def test_gmicloud_reads_base_url_and_api_key():
    from msgflux.models.providers.gmicloud import GMICloudChatCompletion

    model = GMICloudChatCompletion(model_id="MiniMaxAI/MiniMax-M2.7")

    assert model._get_base_url() == "https://api.gmi-serving.com/v1"
    assert model._get_api_key() == "test-key"


def test_gmicloud_missing_api_key_raises(monkeypatch):
    from msgflux.models.providers.gmicloud import GMICloudChatCompletion

    monkeypatch.delenv("GMICLOUD_API_KEY")

    with pytest.raises(ValueError, match="GMICLOUD_API_KEY"):
        GMICloudChatCompletion(model_id="MiniMaxAI/MiniMax-M2.7")


def test_gmicloud_models_registered():
    from msgflux.models.registry import model_registry

    assert "gmicloud" in model_registry.get("chat_completion", {})


def test_gmicloud_resolves_through_model_factory():
    import msgflux as mf

    model = mf.Model.chat_completion("gmicloud/MiniMaxAI/MiniMax-M2.7")

    assert model.provider == "gmicloud"
    assert model.model_id == "MiniMaxAI/MiniMax-M2.7"


def test_gmicloud_chat_round_trip(mock_gmicloud_client):
    from msgflux.models.providers.gmicloud import GMICloudChatCompletion

    mock_gmicloud_client.return_value.chat.completions.create.return_value = (
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
    model = GMICloudChatCompletion(model_id="MiniMaxAI/MiniMax-M2.7")
    response = model("Reply with exactly: OK")

    assert response.consume() == "OK"
    assert response.reasoning == "Checking the request."


def test_gmicloud_api_key_env_override(monkeypatch):
    from msgflux.models.providers.gmicloud import GMICloudChatCompletion

    monkeypatch.setenv("GMICLOUD_ACME_KEY", "gmicloud-acme-key")
    model = GMICloudChatCompletion(
        model_id="MiniMaxAI/MiniMax-M2.7", api_key_env="GMICLOUD_ACME_KEY"
    )

    assert model._get_api_key() == "gmicloud-acme-key"
