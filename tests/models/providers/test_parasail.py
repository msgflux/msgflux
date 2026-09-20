"""Tests for the Parasail OpenAI-compatible provider."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.models._chat_transport import EndpointMockTransport


@pytest.fixture(autouse=True)
def parasail_env(monkeypatch):
    monkeypatch.setenv("PARASAIL_API_KEY", "test-key")
    monkeypatch.setenv("PARASAIL_BASE_URL", "https://api.parasail.io/v1")


@pytest.fixture
def mock_parasail_client():
    from msgflux.models.providers.parasail import ParasailChatCompletion

    client = MagicMock()
    async_client = MagicMock()
    transport = EndpointMockTransport(client.return_value, async_client.return_value)
    with patch.object(ParasailChatCompletion, "chat_transport", transport):
        yield client


def test_parasail_defaults_to_chat_completions():
    from msgflux.models.providers.parasail import ParasailChatCompletion

    model = ParasailChatCompletion(model_id="deepseek-ai/DeepSeek-V4-Flash-0731")

    assert model.provider == "parasail"
    assert model.api_mode == "chat_completions"


def test_parasail_reads_base_url_and_api_key():
    from msgflux.models.providers.parasail import ParasailChatCompletion

    model = ParasailChatCompletion(model_id="deepseek-ai/DeepSeek-V4-Flash-0731")

    assert model._get_base_url() == "https://api.parasail.io/v1"
    assert model._get_api_key() == "test-key"


def test_parasail_missing_api_key_raises(monkeypatch):
    from msgflux.models.providers.parasail import ParasailChatCompletion

    monkeypatch.delenv("PARASAIL_API_KEY")

    with pytest.raises(ValueError, match="PARASAIL_API_KEY"):
        ParasailChatCompletion(model_id="deepseek-ai/DeepSeek-V4-Flash-0731")


def test_parasail_models_registered():
    from msgflux.models.registry import model_registry

    assert "parasail" in model_registry.get("chat_completion", {})


def test_parasail_resolves_through_model_factory():
    import msgflux as mf

    model = mf.Model.chat_completion("parasail/deepseek-ai/DeepSeek-V4-Flash-0731")

    assert model.provider == "parasail"
    assert model.model_id == "deepseek-ai/DeepSeek-V4-Flash-0731"


def test_parasail_chat_round_trip(mock_parasail_client):
    from msgflux.models.providers.parasail import ParasailChatCompletion

    mock_parasail_client.return_value.chat.completions.create.return_value = (
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
    model = ParasailChatCompletion(model_id="deepseek-ai/DeepSeek-V4-Flash-0731")
    response = model("Reply with exactly: OK")

    assert response.consume() == "OK"
    assert response.reasoning == "Checking the request."


def test_parasail_responses_forces_store_false(mock_parasail_client):
    from msgflux.models.providers.parasail import ParasailChatCompletion

    mock_parasail_client.return_value.responses.create.return_value = SimpleNamespace(
        id="resp_1",
        status="completed",
        incomplete_details=None,
        usage=None,
        output=[
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "OK"}],
            },
        ],
    )
    model = ParasailChatCompletion(
        model_id="deepseek-ai/DeepSeek-V4-Flash-0731", api_mode="responses"
    )
    response = model("Reply with exactly: OK")

    request = mock_parasail_client.return_value.responses.create.call_args.kwargs
    assert request["store"] is False
    assert response.consume() == "OK"


def test_parasail_api_key_env_override(monkeypatch):
    from msgflux.models.providers.parasail import ParasailChatCompletion

    monkeypatch.setenv("PARASAIL_ACME_KEY", "parasail-acme-key")
    model = ParasailChatCompletion(
        model_id="deepseek-ai/DeepSeek-V4-Flash-0731",
        api_key_env="PARASAIL_ACME_KEY",
    )

    assert model._get_api_key() == "parasail-acme-key"
