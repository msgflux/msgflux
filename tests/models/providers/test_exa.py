from unittest.mock import MagicMock, patch

import pytest

from msgflux.models.providers.exa import ExaChatCompletion


@pytest.fixture
def mock_openai_client():
    mock_openai = MagicMock(DEFAULT_MAX_RETRIES=2)
    mock_httpx = MagicMock()

    with (
        patch("msgflux.models.providers.openai.openai", mock_openai),
        patch("msgflux.models.providers.openai.httpx", mock_httpx),
        patch("msgflux.models.providers.openai.OpenAI") as mock_client,
        patch("msgflux.models.providers.openai.AsyncOpenAI") as mock_async_client,
    ):
        yield mock_client, mock_async_client


def test_init_raises_without_api_key(mock_openai_client):
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError):
            ExaChatCompletion(model_id="exa")


def test_config(mock_openai_client):
    with patch.dict("os.environ", {"EXA_API_KEY": "test_key"}):
        model = ExaChatCompletion(model_id="exa")
        assert model.provider == "exa"
        assert model.sampling_params["base_url"] == "https://api.exa.ai"
        assert model.client is not None


def test_config_with_research_model(mock_openai_client):
    with patch.dict("os.environ", {"EXA_API_KEY": "test_key"}):
        model = ExaChatCompletion(model_id="exa-research")
        assert model.provider == "exa"
        assert model.model_id == "exa-research"


def test_custom_base_url(mock_openai_client):
    with patch.dict(
        "os.environ",
        {"EXA_API_KEY": "test_key", "EXA_BASE_URL": "https://custom.exa.ai"},
    ):
        model = ExaChatCompletion(model_id="exa")
        assert model.sampling_params["base_url"] == "https://custom.exa.ai"
