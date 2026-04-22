from unittest.mock import MagicMock, patch

import pytest

from msgflux.models.providers.brave import BraveChatCompletion


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
            BraveChatCompletion(model_id="brave")


def test_config(mock_openai_client):
    with patch.dict("os.environ", {"BRAVE_SEARCH_API_KEY": "test_key"}):
        model = BraveChatCompletion(model_id="brave")
        assert model.provider == "brave"
        assert (
            model.sampling_params["base_url"] == "https://api.search.brave.com/res/v1"
        )
        assert model.client is not None
