import pytest
from unittest.mock import patch

from msgflux.models.providers.exa import ExaChatCompletion


def test_init_raises_without_api_key():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError):
            ExaChatCompletion(model_id="exa")


def test_config():
    with patch.dict("os.environ", {"EXA_API_KEY": "test_key"}):
        model = ExaChatCompletion(model_id="exa")
        assert model.provider == "exa"
        assert model.sampling_params["base_url"] == "https://api.exa.ai"
        assert model.client is not None


def test_config_with_research_model():
    with patch.dict("os.environ", {"EXA_API_KEY": "test_key"}):
        model = ExaChatCompletion(model_id="exa-research")
        assert model.provider == "exa"
        assert model.model_id == "exa-research"


def test_custom_base_url():
    with patch.dict(
        "os.environ",
        {"EXA_API_KEY": "test_key", "EXA_BASE_URL": "https://custom.exa.ai"},
    ):
        model = ExaChatCompletion(model_id="exa")
        assert model.sampling_params["base_url"] == "https://custom.exa.ai"
