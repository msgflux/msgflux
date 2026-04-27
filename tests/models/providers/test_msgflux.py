"""Tests for msgflux.models.providers.msgflux module."""

from unittest.mock import patch

import pytest


class TestMsgFluxProviderImport:
    """Test msgFlux provider import and initialization."""

    def test_msgflux_import_available(self):
        """Test that msgFlux provider imports correctly."""
        pytest.importorskip("openai")

        from msgflux.models.providers.msgflux import MsgFluxChatCompletion

        assert MsgFluxChatCompletion.provider == "msgflux"

    def test_msgflux_models_registered(self):
        """Test that msgFlux models are registered with @register_model."""
        pytest.importorskip("openai")

        from msgflux.models.registry import model_registry

        assert "msgflux" in model_registry.get("chat_completion", {})


class TestMsgFluxChatCompletion:
    """Test suite for MsgFluxChatCompletion."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Setup environment variables for tests."""
        monkeypatch.setenv("MSGFLUX_API_KEY", "test-key")
        monkeypatch.setenv("MSGFLUX_BASE_URL", "http://127.0.0.1:8010/v1")

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client."""
        with (
            patch("msgflux.models.providers.openai.OpenAI") as mock_client,
            patch("msgflux.models.providers.openai.AsyncOpenAI") as mock_async_client,
        ):
            yield mock_client, mock_async_client

    def test_chat_completion_initialization(self, mock_openai_client):
        """Test MsgFluxChatCompletion initialization."""
        pytest.importorskip("openai")

        from msgflux.models.providers.msgflux import MsgFluxChatCompletion

        model = MsgFluxChatCompletion(model_id="support")

        assert model.model_id == "support"
        assert model.provider == "msgflux"
        assert model.model_type == "chat_completion"
        assert model.sampling_params["base_url"] == "http://127.0.0.1:8010/v1"

    def test_chat_completion_injects_run_config(self, mock_openai_client):
        """Test run_config is forwarded through OpenAI-compatible extra_body."""
        pytest.importorskip("openai")

        from msgflux.models.providers.msgflux import MsgFluxChatCompletion

        model = MsgFluxChatCompletion(
            model_id="support",
            run_config={"model_preference": "fast"},
            variables={"tenant": "acme"},
        )

        adapted = model._adapt_params({"messages": [], "model": "support"})

        assert adapted["extra_body"]["run_config"] == {
            "model_preference": "fast",
            "vars": {"tenant": "acme"},
        }

    def test_chat_completion_merges_request_run_config(self, mock_openai_client):
        """Test request extra_body run_config overrides model defaults."""
        pytest.importorskip("openai")

        from msgflux.models.providers.msgflux import MsgFluxChatCompletion

        model = MsgFluxChatCompletion(
            model_id="support",
            variables={"tenant": "acme", "tier": "standard"},
        )

        adapted = model._adapt_params(
            {
                "messages": [],
                "model": "support",
                "extra_body": {
                    "run_config": {
                        "vars": {"tier": "priority"},
                        "model_preference": "fast",
                    }
                },
            }
        )

        assert adapted["extra_body"]["run_config"] == {
            "vars": {"tenant": "acme", "tier": "priority"},
            "model_preference": "fast",
        }
