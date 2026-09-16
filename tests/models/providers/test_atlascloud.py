"""Tests for the Atlas Cloud OpenAI-compatible provider."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.models._chat_transport import EndpointMockTransport


@pytest.fixture(autouse=True)
def atlascloud_env(monkeypatch):
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "test-key")
    monkeypatch.setenv("ATLASCLOUD_BASE_URL", "https://api.atlascloud.ai/v1")


@pytest.fixture
def mock_atlascloud_client():
    from msgflux.models.providers.atlascloud import AtlasCloudChatCompletion

    client = MagicMock()
    async_client = MagicMock()
    transport = EndpointMockTransport(client.return_value, async_client.return_value)
    with patch.object(AtlasCloudChatCompletion, "chat_transport", transport):
        yield client


def test_atlascloud_defaults_to_chat_completions():
    from msgflux.models.providers.atlascloud import AtlasCloudChatCompletion

    model = AtlasCloudChatCompletion(model_id="deepseek-ai/DeepSeek-V3.1")

    assert model.provider == "atlascloud"
    assert model.api_mode == "chat_completions"


def test_atlascloud_reads_base_url_and_api_key():
    from msgflux.models.providers.atlascloud import AtlasCloudChatCompletion

    model = AtlasCloudChatCompletion(model_id="deepseek-ai/DeepSeek-V3.1")

    assert model._get_base_url() == "https://api.atlascloud.ai/v1"
    assert model._get_api_key() == "test-key"


def test_atlascloud_missing_api_key_raises(monkeypatch):
    from msgflux.models.providers.atlascloud import AtlasCloudChatCompletion

    monkeypatch.delenv("ATLASCLOUD_API_KEY")

    with pytest.raises(ValueError, match="ATLASCLOUD_API_KEY"):
        AtlasCloudChatCompletion(model_id="deepseek-ai/DeepSeek-V3.1")


def test_atlascloud_models_registered():
    from msgflux.models.registry import model_registry

    assert "atlascloud" in model_registry.get("chat_completion", {})


def test_atlascloud_resolves_slashed_model_id():
    import msgflux as mf

    model = mf.Model.chat_completion("atlascloud/deepseek-ai/DeepSeek-V3.1")

    assert model.provider == "atlascloud"
    assert model.model_id == "deepseek-ai/DeepSeek-V3.1"


def test_atlascloud_chat_round_trip(mock_atlascloud_client):
    from msgflux.models.providers.atlascloud import AtlasCloudChatCompletion

    mock_atlascloud_client.return_value.chat.completions.create.return_value = (
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
    model = AtlasCloudChatCompletion(model_id="deepseek-ai/DeepSeek-V3.1")
    response = model("Reply with exactly: OK")

    assert response.consume() == "OK"
    assert response.reasoning == "Checking the request."


def test_atlascloud_api_key_env_override(monkeypatch):
    from msgflux.models.providers.atlascloud import AtlasCloudChatCompletion

    monkeypatch.setenv("ATLASCLOUD_ACME_KEY", "atlascloud-acme-key")
    model = AtlasCloudChatCompletion(
        model_id="deepseek-ai/DeepSeek-V3.1", api_key_env="ATLASCLOUD_ACME_KEY"
    )

    assert model._get_api_key() == "atlascloud-acme-key"
