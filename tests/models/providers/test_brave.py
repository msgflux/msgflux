import pytest
import os
from unittest.mock import patch
from msgflux.models.providers.brave import BraveChatCompletion

def test_init_raises_without_api_key():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError):
            BraveChatCompletion(model_id="test-model")

def test_config():
    # Chat completion still uses BRAVE_API_KEY in implementation, 
    # ensuring consistency with user's previous request or standard unless changed.
    # The user complaint was about self retention in retriever.
    with patch.dict("os.environ", {"BRAVE_SEARCH_API_KEY": "test_key"}):
        model = BraveChatCompletion(model_id="test-model")
        assert model.provider == "brave"
        assert model.sampling_params["base_url"] == "https://api.search.brave.com/res/v1"
        assert model.client is not None
