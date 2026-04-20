"""Tests for msgflux.models.providers.openrouter module."""

from unittest.mock import patch

import pytest


class TestOpenRouterChatCompletion:
    """Test suite for OpenRouterChatCompletion."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Setup environment variables for tests."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-12345")
        monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client."""
        with (
            patch("msgflux.models.providers.openai.OpenAI") as mock_client,
            patch("msgflux.models.providers.openai.AsyncOpenAI") as mock_async_client,
        ):
            yield mock_client, mock_async_client

    def test_chat_completion_with_reasoning_max_tokens(self, mock_openai_client):
        """Test OpenRouter forwards reasoning_max_tokens as reasoning.max_tokens."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openrouter import OpenRouterChatCompletion

        model = OpenRouterChatCompletion(
            model_id="openrouter/anthropic/claude-sonnet-4.5",
            reasoning_max_tokens=2000,
        )

        assert model.sampling_run_params["reasoning_max_tokens"] == 2000

        params = {
            "messages": [],
            "model": model.model_id,
            "tool_choice": None,
            "tools": None,
            "web_search_options": None,
            "extra_body": {},
            "extra_headers": {},
            **model.sampling_run_params,
        }

        adapted = model._adapt_params(params)

        assert adapted["extra_body"]["reasoning"]["max_tokens"] == 2000

    def test_chat_completion_with_reasoning_effort(self, mock_openai_client):
        """Test OpenRouter still forwards reasoning_effort."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openrouter import OpenRouterChatCompletion

        model = OpenRouterChatCompletion(
            model_id="openrouter/anthropic/claude-sonnet-4.5",
            reasoning_effort="high",
        )

        params = {
            "messages": [],
            "model": model.model_id,
            "tool_choice": None,
            "tools": None,
            "web_search_options": None,
            "extra_body": {},
            "extra_headers": {},
            **model.sampling_run_params,
        }

        adapted = model._adapt_params(params)

        assert adapted["extra_body"]["reasoning"]["effort"] == "high"

    def test_chat_completion_rejects_reasoning_effort_with_max_tokens(
        self, mock_openai_client
    ):
        """Test OpenRouter rejects reasoning_effort and reasoning_max_tokens together."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openrouter import OpenRouterChatCompletion

        with pytest.raises(
            ValueError,
            match="`reasoning_max_tokens` cannot be used together with",
        ):
            OpenRouterChatCompletion(
                model_id="openrouter/anthropic/claude-sonnet-4.5",
                reasoning_effort="high",
                reasoning_max_tokens=2000,
            )
