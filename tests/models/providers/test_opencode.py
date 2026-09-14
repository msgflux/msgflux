"""Tests for the OpenCode Zen/Go OpenAI-compatible providers."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.models._chat_transport import EndpointMockTransport


@pytest.fixture(autouse=True)
def opencode_env(monkeypatch):
    monkeypatch.setenv("OPENCODE_API_KEY", "test-key")
    monkeypatch.setenv("OPENCODE_BASE_URL", "https://opencode.ai/zen/v1")
    monkeypatch.setenv("OPENCODE_GO_BASE_URL", "https://opencode.ai/zen/go/v1")


@pytest.fixture(params=["zen", "go"])
def mock_opencode_client(request):
    from msgflux.models.providers.opencode import (
        OpenCodeChatCompletion,
        OpenCodeGoChatCompletion,
    )

    cls = OpenCodeChatCompletion if request.param == "zen" else OpenCodeGoChatCompletion
    client = MagicMock()
    async_client = MagicMock()
    transport = EndpointMockTransport(client.return_value, async_client.return_value)
    with patch.object(cls, "chat_transport", transport):
        yield request.param, client


def test_zen_and_go_default_to_responses_api_mode():
    from msgflux.models.providers.opencode import (
        OpenCodeChatCompletion,
        OpenCodeGoChatCompletion,
    )

    zen = OpenCodeChatCompletion(model_id="muse-spark-1.3-contributor-free")
    go = OpenCodeGoChatCompletion(model_id="kimi-k3")

    assert (zen.provider, zen.api_mode) == ("opencode", "responses")
    assert (go.provider, go.api_mode) == ("opencode-go", "responses")


def test_zen_and_go_use_gateway_base_urls():
    from msgflux.models.providers.opencode import (
        OpenCodeChatCompletion,
        OpenCodeGoChatCompletion,
    )

    zen = OpenCodeChatCompletion(model_id="muse-spark-1.3-contributor-free")
    go = OpenCodeGoChatCompletion(model_id="kimi-k3")

    assert zen._get_base_url() == "https://opencode.ai/zen/v1"
    assert go._get_base_url() == "https://opencode.ai/zen/go/v1"
    assert zen._get_api_key() == go._get_api_key() == "test-key"


def test_missing_api_key_raises(monkeypatch):
    from msgflux.models.providers.opencode import (
        OpenCodeChatCompletion,
        OpenCodeGoChatCompletion,
    )

    monkeypatch.delenv("OPENCODE_API_KEY")

    with pytest.raises(ValueError, match="OPENCODE_API_KEY"):
        OpenCodeChatCompletion(model_id="muse-spark-1.3-contributor-free")
    with pytest.raises(ValueError, match="OPENCODE_API_KEY"):
        OpenCodeGoChatCompletion(model_id="kimi-k3")


def test_zen_and_go_models_registered():
    from msgflux.models.registry import model_registry

    assert "opencode" in model_registry.get("chat_completion", {})
    assert "opencode-go" in model_registry.get("chat_completion", {})


def test_gateways_resolve_through_model_factory():
    import msgflux as mf

    zen = mf.Model.chat_completion("opencode/muse-spark-1.3-contributor-free")
    go = mf.Model.chat_completion("opencode-go/kimi-k3")

    assert (zen.provider, zen.api_mode) == ("opencode", "responses")
    assert (go.provider, go.api_mode) == ("opencode-go", "responses")


def test_chat_completions_sends_reasoning_effort(mock_opencode_client):
    from msgflux.models.providers.opencode import (
        OpenCodeChatCompletion,
        OpenCodeGoChatCompletion,
    )

    tier, client = mock_opencode_client
    cls = OpenCodeChatCompletion if tier == "zen" else OpenCodeGoChatCompletion
    client.return_value.chat.completions.create.return_value = SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    content="done",
                    tool_calls=None,
                    audio=None,
                    annotations=None,
                ),
            )
        ],
    )
    model = cls(
        model_id="kimi-k3",
        api_mode="chat_completions",
        reasoning_effort="high",
    )
    response = model("Hello")

    request = client.return_value.chat.completions.create.call_args.kwargs
    assert request["reasoning_effort"] == "high"
    assert response.consume() == "done"


def test_responses_nests_reasoning_effort(mock_opencode_client):
    from msgflux.models.providers.opencode import (
        OpenCodeChatCompletion,
        OpenCodeGoChatCompletion,
    )

    tier, client = mock_opencode_client
    cls = OpenCodeChatCompletion if tier == "zen" else OpenCodeGoChatCompletion
    client.return_value.responses.create.return_value = SimpleNamespace(
        id="resp_1",
        status="completed",
        incomplete_details=None,
        usage=None,
        output=[
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "They match."}],
            },
        ],
    )
    model = cls(model_id="kimi-k3", reasoning_effort="low")
    response = model("Compare 17 and 17.")

    request = client.return_value.responses.create.call_args.kwargs
    assert request["reasoning"] == {"effort": "low", "summary": "auto"}
    assert response.consume() == "They match."
