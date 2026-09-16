"""Tests for the Tencent Cloud MaaS OpenAI-compatible provider."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.models._chat_transport import EndpointMockTransport


@pytest.fixture(autouse=True)
def tencentcloud_env(monkeypatch):
    monkeypatch.setenv("TENCENTCLOUD_API_KEY", "test-key")
    monkeypatch.setenv(
        "TENCENTCLOUD_BASE_URL", "https://tokenhub-intl.tencentcloudmaas.com/v1"
    )


@pytest.fixture
def mock_tencentcloud_client():
    from msgflux.models.providers.tencentcloud import TencentCloudChatCompletion

    client = MagicMock()
    async_client = MagicMock()
    transport = EndpointMockTransport(client.return_value, async_client.return_value)
    with patch.object(TencentCloudChatCompletion, "chat_transport", transport):
        yield client


def test_tencentcloud_defaults_to_chat_completions():
    from msgflux.models.providers.tencentcloud import TencentCloudChatCompletion

    model = TencentCloudChatCompletion(model_id="glm-5.3-flash")

    assert model.provider == "tencentcloud"
    assert model.api_mode == "chat_completions"


def test_tencentcloud_reads_base_url_and_api_key():
    from msgflux.models.providers.tencentcloud import TencentCloudChatCompletion

    model = TencentCloudChatCompletion(model_id="glm-5.3-flash")

    assert model._get_base_url() == "https://tokenhub-intl.tencentcloudmaas.com/v1"
    assert model._get_api_key() == "test-key"


def test_tencentcloud_missing_api_key_raises(monkeypatch):
    from msgflux.models.providers.tencentcloud import TencentCloudChatCompletion

    monkeypatch.delenv("TENCENTCLOUD_API_KEY")

    with pytest.raises(ValueError, match="TENCENTCLOUD_API_KEY"):
        TencentCloudChatCompletion(model_id="glm-5.3-flash")


def test_tencentcloud_models_registered():
    from msgflux.models.registry import model_registry

    assert "tencentcloud" in model_registry.get("chat_completion", {})


def test_tencentcloud_resolves_through_model_factory():
    import msgflux as mf

    model = mf.Model.chat_completion("tencentcloud/glm-5.3-flash")

    assert model.provider == "tencentcloud"
    assert model.api_mode == "chat_completions"


def test_tencentcloud_chat_round_trip(mock_tencentcloud_client):
    from msgflux.models.providers.tencentcloud import TencentCloudChatCompletion

    mock_tencentcloud_client.return_value.chat.completions.create.return_value = (
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
    model = TencentCloudChatCompletion(model_id="glm-5.3-flash")
    response = model("Reply with exactly: OK")

    assert response.consume() == "OK"
    assert response.reasoning == "Checking the request."


def test_tencentcloud_api_key_env_override(monkeypatch):
    from msgflux.models.providers.tencentcloud import TencentCloudChatCompletion

    monkeypatch.setenv("TENCENTCLOUD_ACME_KEY", "tencentcloud-acme-key")
    model = TencentCloudChatCompletion(
        model_id="glm-5.3-flash", api_key_env="TENCENTCLOUD_ACME_KEY"
    )

    assert model._get_api_key() == "tencentcloud-acme-key"
