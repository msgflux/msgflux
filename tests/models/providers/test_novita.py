"""Tests for the Novita OpenAI-compatible provider."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.models._chat_transport import EndpointMockTransport


@pytest.fixture(autouse=True)
def novita_env(monkeypatch):
    monkeypatch.setenv("NOVITA_API_KEY", "test-key")
    monkeypatch.setenv("NOVITA_BASE_URL", "https://api.novita.ai/openai/v1")


@pytest.fixture
def mock_novita_client():
    from msgflux.models.providers.novita import NovitaChatCompletion

    client = MagicMock()
    async_client = MagicMock()
    transport = EndpointMockTransport(client.return_value, async_client.return_value)
    with patch.object(NovitaChatCompletion, "chat_transport", transport):
        yield client


def test_novita_defaults_to_chat_completions():
    from msgflux.models.providers.novita import NovitaChatCompletion

    model = NovitaChatCompletion(model_id="zai-org/glm-5.3-flash")

    assert model.provider == "novita"
    assert model.api_mode == "chat_completions"


def test_novita_reads_base_url_and_api_key():
    from msgflux.models.providers.novita import NovitaChatCompletion

    model = NovitaChatCompletion(model_id="zai-org/glm-5.3-flash")

    assert model._get_base_url() == "https://api.novita.ai/openai/v1"
    assert model._get_api_key() == "test-key"


def test_novita_missing_api_key_raises(monkeypatch):
    from msgflux.models.providers.novita import NovitaChatCompletion

    monkeypatch.delenv("NOVITA_API_KEY")

    with pytest.raises(ValueError, match="NOVITA_API_KEY"):
        NovitaChatCompletion(model_id="zai-org/glm-5.3-flash")


def test_novita_models_registered():
    from msgflux.models.registry import model_registry

    assert "novita" in model_registry.get("chat_completion", {})


def test_novita_resolves_through_model_factory():
    import msgflux as mf

    model = mf.Model.chat_completion("novita/zai-org/glm-5.3-flash")

    assert model.provider == "novita"
    assert model.api_mode == "chat_completions"


def test_novita_chat_extracts_reasoning_content(mock_novita_client):
    from msgflux.models.providers.novita import NovitaChatCompletion

    mock_novita_client.return_value.chat.completions.create.return_value = (
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
    model = NovitaChatCompletion(model_id="zai-org/glm-5.3-flash")
    response = model("Reply with exactly: OK")

    assert response.consume() == "OK"
    assert response.reasoning == "Checking the request."


def test_novita_responses_uses_clear_text_reasoning(mock_novita_client):
    from msgflux.models.providers.novita import NovitaChatCompletion

    reasoning_item = {
        "type": "reasoning",
        "id": "rs_1",
        "status": "completed",
        "summary": [{"type": "summary_text", "text": "Checking the request."}],
    }
    mock_novita_client.return_value.responses.create.return_value = SimpleNamespace(
        id="resp_1",
        status="completed",
        incomplete_details=None,
        usage=None,
        output=[
            reasoning_item,
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "OK"}],
            },
        ],
    )
    model = NovitaChatCompletion(model_id="zai-org/glm-5.3-flash", api_mode="responses")
    response = model("Reply with exactly: OK")

    assert response.consume() == "OK"
    assert response.reasoning_summary == "Checking the request."


def test_novita_api_key_env_override(monkeypatch):
    from msgflux.models.providers.novita import NovitaChatCompletion

    monkeypatch.setenv("NOVITA_ACME_KEY", "novita-acme-key")
    model = NovitaChatCompletion(
        model_id="zai-org/glm-5.3-flash", api_key_env="NOVITA_ACME_KEY"
    )

    assert model._get_api_key() == "novita-acme-key"
