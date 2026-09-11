"""Tests for msgflux.models.providers.openrouter module."""

from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import pytest

from msgflux.chat_messages import ChatMessages
from tests.models._chat_transport import EndpointMockTransport


class TestOpenRouterChatCompletion:
    """Test suite for OpenRouterChatCompletion."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Setup environment variables for tests."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-12345")
        monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    @pytest.fixture
    def mock_openai_client(self):
        """Mock provider chat endpoints."""
        from msgflux.models.providers.openrouter import OpenRouterChatCompletion

        mock_client = MagicMock()
        mock_async_client = MagicMock()
        transport = EndpointMockTransport(
            mock_client.return_value,
            mock_async_client.return_value,
        )
        with patch.object(OpenRouterChatCompletion, "chat_transport", transport):
            yield mock_client, mock_async_client

    def test_openrouter_defaults_to_direct_chat_transport(self):
        from msgflux.models.chat_transport import HTTPChatTransport
        from msgflux.models.providers.openrouter import OpenRouterChatCompletion

        model = OpenRouterChatCompletion(model_id="nvidia/nemotron-3.5-lightning:free")

        assert isinstance(model.chat_transport, HTTPChatTransport)

    def test_chat_completion_with_reasoning_max_tokens(self, mock_openai_client):
        """Test OpenRouter forwards reasoning_max_tokens as reasoning.max_tokens."""

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

    def test_set_reasoning_effort_replaces_reasoning_token_budget(
        self, mock_openai_client
    ):
        from msgflux.models.providers.openrouter import OpenRouterChatCompletion

        model = OpenRouterChatCompletion(
            model_id="openai/gpt-oss-120b",
            reasoning_max_tokens=2000,
        )

        model.set_reasoning_effort("high")

        assert model.reasoning_max_tokens is None
        assert "reasoning_max_tokens" not in model.sampling_run_params
        assert model.sampling_run_params["reasoning_effort"] == "high"

    def test_fast_speed_uses_native_openrouter_speed_for_claude(
        self, mock_openai_client
    ):
        from msgflux.models.providers.openrouter import OpenRouterChatCompletion

        mock_client, _ = mock_openai_client
        model = OpenRouterChatCompletion(
            model_id="anthropic/claude-opus-4.6",
            speed="fast",
        )

        model._execute_model(model=model.model_id, messages=[])

        request = mock_client.return_value.chat.completions.create.call_args.kwargs
        assert request["model"] == "anthropic/claude-opus-4.6"
        assert request["speed"] == "fast"

    @pytest.mark.parametrize("speed", ["fast", "nitro"])
    def test_openrouter_routes_non_claude_speed_through_nitro(
        self, mock_openai_client, speed
    ):
        from msgflux.models.providers.openrouter import OpenRouterChatCompletion

        mock_client, _ = mock_openai_client
        model = OpenRouterChatCompletion(
            model_id="openai/gpt-oss-120b",
            speed=speed,
        )

        model._execute_model(model=model.model_id, messages=[])

        request = mock_client.return_value.chat.completions.create.call_args.kwargs
        assert request["model"] == "openai/gpt-oss-120b:nitro"
        assert "speed" not in request

    @pytest.mark.parametrize(
        ("model_id", "speed"),
        [
            ("openai/gpt-oss-120b:free", "nitro"),
            ("openai/gpt-oss-120b", "ultrafast"),
        ],
    )
    def test_openrouter_warns_for_incompatible_speed(
        self, mock_openai_client, model_id, speed
    ):
        from msgflux.models.providers.openrouter import OpenRouterChatCompletion

        with pytest.warns(UserWarning, match="does not support"):
            model = OpenRouterChatCompletion(model_id=model_id, speed=speed)

        assert model.chat_settings == {}

    def test_openrouter_metadata_reads_effective_speed_from_usage(
        self, mock_openai_client
    ):
        from msgflux.models.providers.openrouter import OpenRouterChatCompletion

        model = OpenRouterChatCompletion(
            model_id="anthropic/claude-opus-4.6",
            speed="fast",
        )

        metadata = model._build_response_metadata(
            SimpleNamespace(usage=SimpleNamespace(speed="fast"))
        )

        assert metadata.model.requested_speed == "fast"
        assert metadata.model.effective_speed == "fast"

    def test_adapt_params_accepts_requests_without_tool_keys(self, mock_openai_client):

        from msgflux.models.providers.openrouter import OpenRouterChatCompletion

        model = OpenRouterChatCompletion(model_id="nvidia/test-model")
        params = model._adapt_params(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": model.model_id,
            }
        )

        assert params["tool_choice"] == "none"
        assert params["extra_body"] == {}

    @pytest.mark.parametrize(
        ("store", "zdr"),
        [(False, True), (True, False)],
    )
    def test_store_maps_to_openrouter_zdr(self, mock_openai_client, store, zdr):

        from msgflux.models.providers.openrouter import OpenRouterChatCompletion

        model = OpenRouterChatCompletion(
            model_id="nvidia/test-model",
            store=store,
            extra_body={"provider": {"allow_fallbacks": False}},
        )
        params = model._adapt_params(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": model.model_id,
                **model.sampling_run_params,
            }
        )

        assert "store" not in params
        assert params["extra_body"]["provider"] == {
            "allow_fallbacks": False,
            "zdr": zdr,
        }

    def test_openrouter_omits_zdr_when_store_is_not_configured(
        self, mock_openai_client
    ):

        from msgflux.models.providers.openrouter import OpenRouterChatCompletion

        model = OpenRouterChatCompletion(model_id="nvidia/test-model")
        params = model._adapt_params(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": model.model_id,
                **model.sampling_run_params,
            }
        )

        assert "provider" not in params["extra_body"]

    def test_responses_api_mode_is_not_inherited_implicitly(self, mock_openai_client):

        from msgflux.models.providers.openrouter import OpenRouterChatCompletion

        with pytest.raises(ValueError, match="does not support"):
            OpenRouterChatCompletion(
                model_id="openai/gpt-oss-120b",
                api_mode="responses",
            )

    def test_chat_completion_with_reasoning_effort(self, mock_openai_client):
        """Test OpenRouter still forwards reasoning_effort."""

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

    def test_model_converts_canonical_messages_for_selected_provider(
        self, mock_openai_client
    ):

        from msgflux.models.providers.openrouter import OpenRouterChatCompletion

        model = OpenRouterChatCompletion(model_id="openai/gpt-oss-120b")
        assert model.api_mode == "chat_completions"
        assert model.reasoning_codec.name == "openrouter_reasoning_details"
        details = [{"type": "reasoning.encrypted", "data": "opaque"}]
        messages = ChatMessages()
        messages.add_reasoning(
            "summary",
            provider="openrouter",
            provider_state=details,
        )
        messages.add_assistant("answer")

        params = model._build_generation_params(
            messages=messages,
            system_prompt=None,
            prefilling=None,
            tool_catalog=None,
        )

        assert params["messages"] == [
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "summary",
                "reasoning_details": details,
            }
        ]

    def test_response_state_records_provider_api_and_codec(self, mock_openai_client):

        from msgflux.models.providers.openrouter import OpenRouterChatCompletion

        model = OpenRouterChatCompletion(model_id="openai/gpt-oss-120b")
        details = [{"type": "reasoning.encrypted", "data": "opaque"}]
        output = SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    logprobs=None,
                    message=SimpleNamespace(
                        content="answer",
                        reasoning_content="summary",
                        reasoning_details=details,
                        tool_calls=None,
                        audio=None,
                        annotations=None,
                    ),
                )
            ],
        )

        response = model._process_completion_model_output(output)

        assert response.history_items[0]["provider_state"] == {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "codec": "openrouter_reasoning_details",
            "data": details,
        }

    def test_chat_completion_rejects_reasoning_effort_with_max_tokens(
        self, mock_openai_client
    ):
        """Test OpenRouter rejects reasoning_effort and reasoning_max_tokens together."""

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
