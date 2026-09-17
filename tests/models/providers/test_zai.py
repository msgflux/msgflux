"""Tests for the Z.AI and ZhipuAI BigModel providers."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.models._chat_transport import EndpointMockTransport


@pytest.fixture(autouse=True)
def zai_env(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test-key")
    monkeypatch.setenv("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4")
    monkeypatch.setenv("ZAI_CODE_API_KEY", "test-code-key")
    monkeypatch.setenv("ZAI_CODE_BASE_URL", "https://api.z.ai/api/coding/paas/v4")


@pytest.fixture(params=["zai", "zai-code"])
def mock_zai_client(request):
    from msgflux.models.providers.zai import (
        ZAICodeChatCompletion,
        ZAIChatCompletion,
    )

    cls = ZAIChatCompletion if request.param == "zai" else ZAICodeChatCompletion
    client = MagicMock()
    async_client = MagicMock()
    transport = EndpointMockTransport(client.return_value, async_client.return_value)
    with patch.object(cls, "chat_transport", transport):
        yield request.param, client


def test_zai_providers_default_to_chat_completions():
    from msgflux.models.providers.zai import (
        ZAICodeChatCompletion,
        ZAIChatCompletion,
    )

    paygo = ZAIChatCompletion(model_id="glm-5.3")
    code = ZAICodeChatCompletion(model_id="glm-5.3")

    assert (paygo.provider, paygo.api_mode) == ("zai", "chat_completions")
    assert (code.provider, code.api_mode) == ("zai-code", "chat_completions")


def test_zai_providers_use_gateway_base_urls_and_own_keys():
    from msgflux.models.providers.zai import (
        ZAICodeChatCompletion,
        ZAIChatCompletion,
    )

    paygo = ZAIChatCompletion(model_id="glm-5.3")
    code = ZAICodeChatCompletion(model_id="glm-5.3")

    assert paygo._get_base_url() == "https://api.z.ai/api/paas/v4"
    assert code._get_base_url() == "https://api.z.ai/api/coding/paas/v4"
    assert paygo._get_api_key() == "test-key"
    assert code._get_api_key() == "test-code-key"


def test_zai_missing_api_key_raises(monkeypatch):
    from msgflux.models.providers.zai import (
        ZAICodeChatCompletion,
        ZAIChatCompletion,
    )

    monkeypatch.delenv("ZAI_API_KEY")
    monkeypatch.delenv("ZAI_CODE_API_KEY")

    with pytest.raises(ValueError, match="ZAI_API_KEY"):
        ZAIChatCompletion(model_id="glm-5.3")
    with pytest.raises(ValueError, match="ZAI_CODE_API_KEY"):
        ZAICodeChatCompletion(model_id="glm-5.3")


def test_zai_models_registered():
    from msgflux.models.registry import model_registry

    assert "zai" in model_registry.get("chat_completion", {})
    assert "zai-code" in model_registry.get("chat_completion", {})


def test_zai_gateways_resolve_through_model_factory():
    import msgflux as mf

    paygo = mf.Model.chat_completion("zai/glm-5.3")
    code = mf.Model.chat_completion("zai-code/glm-5.3")

    assert (paygo.provider, paygo.api_mode) == ("zai", "chat_completions")
    assert (code.provider, code.api_mode) == ("zai-code", "chat_completions")


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


def test_zai_chat_round_trip(mock_zai_client):
    from msgflux.models.providers.zai import (
        ZAICodeChatCompletion,
        ZAIChatCompletion,
    )

    tier, client = mock_zai_client
    cls = ZAIChatCompletion if tier == "zai" else ZAICodeChatCompletion
    _ok(client)
    response = cls(model_id="glm-5.3")("Reply with exactly: OK")

    assert response.consume() == "OK"
    assert response.reasoning == "Checking the request."


def test_zhipu_gateways_use_bigmodel_bases(monkeypatch):
    from msgflux.models.providers.zai import (
        ZhipuChatCompletion,
        ZhipuCodeChatCompletion,
    )

    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
    monkeypatch.setenv("ZHIPU_CODE_API_KEY", "test-code-key")

    paygo = ZhipuChatCompletion(model_id="glm-5.2")
    code = ZhipuCodeChatCompletion(model_id="glm-5.2")

    assert (paygo.provider, paygo.api_mode) == ("zhipu", "chat_completions")
    assert (code.provider, code.api_mode) == ("zhipu-code", "chat_completions")
    assert paygo._get_base_url() == "https://open.bigmodel.cn/api/paas/v4"
    assert code._get_base_url() == "https://open.bigmodel.cn/api/coding/paas/v4"
    assert paygo._get_api_key() == "test-key"
    assert code._get_api_key() == "test-code-key"


def test_zhipu_models_registered_and_resolvable(monkeypatch):
    import msgflux as mf
    from msgflux.models.registry import model_registry

    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
    monkeypatch.setenv("ZHIPU_CODE_API_KEY", "test-code-key")

    assert "zhipu" in model_registry.get("chat_completion", {})
    assert "zhipu-code" in model_registry.get("chat_completion", {})

    paygo = mf.Model.chat_completion("zhipu/glm-5.2")
    code = mf.Model.chat_completion("zhipu-code/glm-5.2")

    assert paygo.model_id == "glm-5.2"
    assert code.model_id == "glm-5.2"
