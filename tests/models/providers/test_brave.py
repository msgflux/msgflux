from unittest.mock import patch

import pytest

from msgflux.models.providers.brave import BraveChatCompletion


def test_init_raises_without_api_key():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError):
            BraveChatCompletion(model_id="brave")


def test_config():
    with patch.dict("os.environ", {"BRAVE_SEARCH_API_KEY": "test_key"}):
        model = BraveChatCompletion(model_id="brave")
        assert model.provider == "brave"
        assert (
            model.sampling_params["base_url"] == "https://api.search.brave.com/res/v1"
        )
        assert model.chat_transport is not None
