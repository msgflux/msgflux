"""Tests for the RunInfra OpenAI-compatible provider."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.models._chat_transport import EndpointMockTransport


@pytest.fixture(autouse=True)
def runinfra_env(monkeypatch):
    monkeypatch.setenv("RUNINFRA_API_KEY", "test-key")
    monkeypatch.setenv("RUNINFRA_BASE_URL", "https://api.runinfra.ai/v1")


@pytest.fixture
def mock_runinfra_client():
    from msgflux.models.providers.runinfra import RunInfraChatCompletion

    client = MagicMock()
    async_client = MagicMock()
    transport = EndpointMockTransport(client.return_value, async_client.return_value)
    with patch.object(RunInfraChatCompletion, "chat_transport", transport):
        yield client


def test_runinfra_defaults_to_chat_completions():
    from msgflux.models.providers.runinfra import RunInfraChatCompletion

    model = RunInfraChatCompletion(model_id="glm-5-3-flash")

    assert model.provider == "runinfra"
    assert model.api_mode == "chat_completions"


def test_runinfra_reads_base_url_and_api_key():
    from msgflux.models.providers.runinfra import RunInfraChatCompletion

    model = RunInfraChatCompletion(model_id="glm-5-3-flash")

    assert model._get_base_url() == "https://api.runinfra.ai/v1"
    assert model._get_api_key() == "test-key"


def test_runinfra_missing_api_key_raises(monkeypatch):
    from msgflux.models.providers.runinfra import RunInfraChatCompletion

    monkeypatch.delenv("RUNINFRA_API_KEY")

    with pytest.raises(ValueError, match="RUNINFRA_API_KEY"):
        RunInfraChatCompletion(model_id="glm-5-3-flash")


def test_runinfra_models_registered():
    from msgflux.models.registry import model_registry

    assert "runinfra" in model_registry.get("chat_completion", {})


def test_runinfra_resolves_through_model_factory():
    import msgflux as mf

    model = mf.Model.chat_completion("runinfra/glm-5-3-flash")

    assert model.provider == "runinfra"
    assert model.model_id == "glm-5-3-flash"


def test_runinfra_chat_round_trip(mock_runinfra_client):
    from msgflux.models.providers.runinfra import RunInfraChatCompletion

    mock_runinfra_client.return_value.chat.completions.create.return_value = (
        SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content="OK",
                        reasoning="Checking the request.",
                        tool_calls=None,
                        audio=None,
                        annotations=None,
                    ),
                )
            ],
        )
    )
    model = RunInfraChatCompletion(model_id="glm-5-3-flash")
    response = model("Reply with exactly: OK")

    assert response.consume() == "OK"
    assert response.reasoning == "Checking the request."


def test_runinfra_api_key_env_override(monkeypatch):
    from msgflux.models.providers.runinfra import RunInfraChatCompletion

    monkeypatch.setenv("RUNINFRA_ACME_KEY", "runinfra-acme-key")
    model = RunInfraChatCompletion(
        model_id="glm-5-3-flash", api_key_env="RUNINFRA_ACME_KEY"
    )

    assert model._get_api_key() == "runinfra-acme-key"
