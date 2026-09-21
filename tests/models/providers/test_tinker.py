"""Tests for the Thinking Machines Tinker OpenAI-compatible provider."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.models._chat_transport import EndpointMockTransport

MODEL_ID = (
    "tinker://0034d8c9-0a88-52a9-b2b7-bce7cb1e6fef:train:0/sampler_weights/000080"
)


@pytest.fixture(autouse=True)
def tinker_env(monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    monkeypatch.setenv(
        "TINKER_BASE_URL",
        "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1",
    )


@pytest.fixture
def mock_tinker_client():
    from msgflux.models.providers.tinker import TinkerChatCompletion

    client = MagicMock()
    async_client = MagicMock()
    transport = EndpointMockTransport(client.return_value, async_client.return_value)
    with patch.object(TinkerChatCompletion, "chat_transport", transport):
        yield client


def test_tinker_defaults_to_chat_completions():
    from msgflux.models.providers.tinker import TinkerChatCompletion

    model = TinkerChatCompletion(model_id=MODEL_ID)

    assert model.provider == "tinker"
    assert model.api_mode == "chat_completions"


def test_tinker_reads_base_url_and_api_key():
    from msgflux.models.providers.tinker import TinkerChatCompletion

    model = TinkerChatCompletion(model_id=MODEL_ID)

    assert (
        model._get_base_url()
        == "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
    )
    assert model._get_api_key() == "test-key"


def test_tinker_missing_api_key_raises(monkeypatch):
    from msgflux.models.providers.tinker import TinkerChatCompletion

    monkeypatch.delenv("TINKER_API_KEY")

    with pytest.raises(ValueError, match="TINKER_API_KEY"):
        TinkerChatCompletion(model_id=MODEL_ID)


def test_tinker_models_registered():
    from msgflux.models.registry import model_registry

    assert "tinker" in model_registry.get("chat_completion", {})


def test_tinker_resolves_checkpoint_path_through_factory():
    import msgflux as mf

    model = mf.Model.chat_completion(f"tinker/{MODEL_ID}")

    assert model.provider == "tinker"
    assert model.model_id == MODEL_ID


def test_tinker_chat_round_trip(mock_tinker_client):
    from msgflux.models.providers.tinker import TinkerChatCompletion

    mock_tinker_client.return_value.chat.completions.create.return_value = (
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
    model = TinkerChatCompletion(model_id=MODEL_ID)
    response = model("Reply with exactly: OK")

    assert response.consume() == "OK"
    assert response.reasoning == "Checking the request."


def test_tinker_api_key_env_override(monkeypatch):
    from msgflux.models.providers.tinker import TinkerChatCompletion

    monkeypatch.setenv("TINKER_ACME_KEY", "tinker-acme-key")
    model = TinkerChatCompletion(model_id=MODEL_ID, api_key_env="TINKER_ACME_KEY")

    assert model._get_api_key() == "tinker-acme-key"
