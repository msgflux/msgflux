"""Tests for msgflux.models.providers.openrouter module."""

from unittest.mock import AsyncMock, MagicMock, patch
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

    def test_responses_mode_is_explicit_and_keeps_commentary_untrusted(
        self, mock_openai_client
    ):

        from msgflux.models.providers.openrouter import OpenRouterChatCompletion

        model = OpenRouterChatCompletion(
            model_id="meta/muse-spark-1.3-contributor",
            api_mode="responses",
        )

        assert model.api_mode == "responses"
        assert model.api_mode_capabilities.assistant_commentary is False
        assert model.reasoning_codec.name == "openrouter_responses"

    def test_responses_params_preserve_zdr_and_reasoning_budget(
        self, mock_openai_client
    ):
        from msgflux.models.providers.openrouter import OpenRouterChatCompletion

        model = OpenRouterChatCompletion(
            model_id="meta/muse-spark-1.3-contributor",
            api_mode="responses",
            store=False,
            reasoning_max_tokens=200,
        )
        params = model._adapt_responses_params(
            {
                "model": model.model_id,
                "input": [{"role": "user", "content": "Hi"}],
                "extra_body": {"provider": {"allow_fallbacks": False}},
                **model.sampling_run_params,
            }
        )

        assert "store" not in params
        assert "reasoning_max_tokens" not in params
        assert params["extra_body"]["provider"] == {
            "allow_fallbacks": False,
            "zdr": True,
        }
        assert params["extra_body"]["reasoning"] == {"max_tokens": 200}

    @pytest.mark.parametrize("api_mode", ["chat_completions", "responses"])
    def test_openrouter_uses_thread_for_sticky_cache_routing(
        self, mock_openai_client, api_mode
    ):
        from msgflux.models.providers.openrouter import OpenRouterChatCompletion
        from msgflux.runtime.context import thread_context

        model = OpenRouterChatCompletion(
            model_id="z-ai/glm-5.3-flash", api_mode=api_mode
        )
        with thread_context(thread_id="thread_1"):
            params = (
                model._adapt_params({"model": model.model_id, "max_tokens": 20})
                if api_mode == "chat_completions"
                else model._adapt_responses_params(
                    {"model": model.model_id, "max_tokens": 20}
                )
            )
        assert params["extra_headers"]["x-session-id"] == "thread_1"

        with thread_context(thread_id="thread_1"):
            explicit = model._adapt_responses_params(
                {"extra_body": {"session_id": "explicit_session"}}
            )
        assert "x-session-id" not in explicit["extra_headers"]
        assert explicit["extra_body"]["session_id"] == "explicit_session"

    @pytest.mark.parametrize("api_mode", ["chat_completions", "responses"])
    def test_chat_messages_thread_keeps_sticky_routing_without_runtime_context(
        self, mock_openai_client, api_mode
    ):
        from msgflux.models.providers.openrouter import OpenRouterChatCompletion

        model = OpenRouterChatCompletion(
            model_id="z-ai/glm-5.3-flash", api_mode=api_mode
        )
        messages = ChatMessages(
            [{"role": "user", "content": "Check inventory."}],
            thread_id="inventory-thread-42",
        )
        params = model._build_generation_params(messages, None, None, None)
        adapted = (
            model._adapt_params(params)
            if api_mode == "chat_completions"
            else model._adapt_responses_params(params)
        )
        assert adapted["extra_headers"]["x-session-id"] == "inventory-thread-42"

    def test_responses_rejects_reasoning_effort_with_budget(self, mock_openai_client):
        from msgflux.models.providers.openrouter import OpenRouterChatCompletion

        model = OpenRouterChatCompletion(
            model_id="meta/muse-spark-1.3-contributor", api_mode="responses"
        )
        with pytest.raises(ValueError, match="cannot be used together"):
            model._adapt_responses_params(
                {"reasoning_effort": "high", "reasoning_max_tokens": 200}
            )

    def test_responses_encrypted_reasoning_round_trips_without_commentary(
        self, mock_openai_client
    ):
        from msgflux.models.providers.openrouter import OpenRouterChatCompletion

        model = OpenRouterChatCompletion(model_id="x-ai/grok-4.7", api_mode="responses")
        response = model._process_responses_model_output(
            {
                "status": "completed",
                "output": [
                    {
                        "type": "reasoning",
                        "id": "rs_1",
                        "encrypted_content": "opaque",
                        "summary": [],
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "phase": "commentary",
                        "content": [{"type": "output_text", "text": "internal text"}],
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "phase": "final_answer",
                        "content": [{"type": "output_text", "text": "Done."}],
                    },
                ],
            }
        )

        assert response.data == "Done."
        assert response.commentary == []
        assert response.reasoning is None
        replay = ChatMessages(response.history_items).to_responses_input(
            provider="openrouter",
            api_mode="responses",
            reasoning_codec=model.reasoning_codec,
        )
        assert replay[0]["encrypted_content"] == "opaque"
        assert replay[0] == {
            "type": "reasoning",
            "id": "rs_1",
            "encrypted_content": "opaque",
            "summary": [],
        }
        assert [item.get("phase") for item in replay[1:]] == [
            "commentary",
            "final_answer",
        ]

    @pytest.mark.asyncio
    async def test_responses_stream_does_not_publish_untrusted_commentary(
        self, mock_openai_client
    ):
        from msgflux.models.providers.openrouter import OpenRouterChatCompletion
        from msgflux.models.response import ModelStreamResponse
        from msgflux.models.tool_call_agg import ToolCallAggregator

        model = OpenRouterChatCompletion(model_id="x-ai/grok-4.7", api_mode="responses")
        response = ModelStreamResponse()
        state = model._new_responses_stream_state(MagicMock())
        aggregator = ToolCallAggregator(api_mode="responses")
        model._handle_responses_stream_event(
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "message", "role": "assistant", "phase": "commentary"},
            },
            response,
            aggregator,
            state,
        )
        model._handle_responses_stream_event(
            {
                "type": "response.output_text.delta",
                "output_index": 0,
                "delta": "private thought",
            },
            response,
            aggregator,
            state,
        )
        response.finish()

        assert response.data is None
        assert response.commentary == []
        assert [event async for event in response.consume_events()] == []
        assert response.chat_accumulator.snapshot()[0]["phase"] == "commentary"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("calls_tool", [False, True])
    async def test_unphased_text_waits_for_tool_decision(
        self, mock_openai_client, calls_tool
    ):
        from msgflux.models.providers.openrouter import OpenRouterChatCompletion
        from msgflux.models.response import ModelStreamResponse
        from msgflux.models.tool_call_agg import ToolCallAggregator

        model = OpenRouterChatCompletion(
            model_id="deepseek/deepseek-v4-flash", api_mode="responses"
        )
        response = ModelStreamResponse()
        state = model._new_responses_stream_state(MagicMock())
        state["has_tools"] = True
        aggregator = ToolCallAggregator(api_mode="responses")
        model._handle_responses_stream_event(
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "message", "role": "assistant"},
            },
            response,
            aggregator,
            state,
        )
        model._handle_responses_stream_event(
            {
                "type": "response.output_text.delta",
                "output_index": 0,
                "delta": "Checking inventory.",
            },
            response,
            aggregator,
            state,
        )
        assert response.data is None
        if calls_tool:
            aggregator.process(1, "call_1", "lookup_inventory", "{}")
        model._finish_responses_stream(response, aggregator, state)
        response.finish()

        if calls_tool:
            assert response.data != "Checking inventory."
        else:
            assert response.data == "Checking inventory."
        events = [event async for event in response.consume_events()]
        assert [(event.type, event.data) for event in events] == (
            [("commentary.delta", "Checking inventory.")]
            if calls_tool
            else [("output.delta", "Checking inventory.")]
        )
        if calls_tool:
            assert response.chat_accumulator.snapshot()[0]["visibility"] == "commentary"
            assert "phase" not in response.chat_accumulator.snapshot()[0]

    @pytest.mark.asyncio
    async def test_responses_stream_classifies_only_messages_before_tool(
        self, mock_openai_client
    ):
        from msgflux.models.providers.openrouter import OpenRouterChatCompletion
        from msgflux.models.response import ModelStreamResponse
        from msgflux.models.tool_call_agg import ToolCallAggregator

        model = OpenRouterChatCompletion(
            model_id="deepseek/deepseek-v3.2", api_mode="responses"
        )
        response = ModelStreamResponse()
        state = model._new_responses_stream_state(MagicMock())
        state["has_tools"] = True
        aggregator = ToolCallAggregator(api_mode="responses")
        for index, phase, content in (
            (0, "commentary", "Checking inventory."),
            (2, None, "Post-tool text."),
        ):
            model._handle_responses_stream_event(
                {
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": {"type": "message", "role": "assistant", "phase": phase},
                },
                response,
                aggregator,
                state,
            )
            model._handle_responses_stream_event(
                {
                    "type": "response.output_text.delta",
                    "output_index": index,
                    "delta": content,
                },
                response,
                aggregator,
                state,
            )
        model._handle_responses_stream_event(
            {
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup_inventory",
                    "arguments": "{}",
                },
            },
            response,
            aggregator,
            state,
        )
        model._finish_responses_stream(response, aggregator, state)
        response.finish()

        assert response.commentary == ["Checking inventory."]
        assert [
            (event.type, event.data) async for event in response.consume_events()
        ] == [("commentary.delta", "Checking inventory.")]
        history = response.chat_accumulator.snapshot()
        assert history[0]["visibility"] == "commentary"
        assert "visibility" not in history[1]

    def test_recent_unphased_tool_preamble_replays_without_synthetic_phase(
        self, mock_openai_client
    ):
        from msgflux.models.providers.openrouter import OpenRouterChatCompletion

        model = OpenRouterChatCompletion(
            model_id="deepseek/deepseek-v4-flash", api_mode="responses"
        )
        message = {
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Checking inventory."}],
        }
        response = model._process_responses_model_output(
            {
                "status": "completed",
                "output": [
                    message,
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "lookup_inventory",
                        "arguments": '{"sku":"42"}',
                    },
                ],
            }
        )

        assert response.commentary == ["Checking inventory."]
        assert response.history_items[0]["visibility"] == "commentary"
        replay = ChatMessages(response.history_items).to_responses_input(
            provider="openrouter",
            api_mode="responses",
            reasoning_codec=model.reasoning_codec,
        )
        assert replay[0] == message
        messages = ChatMessages([{"role": "user", "content": "Check inventory."}])
        initial = model._build_generation_params(
            messages, "Stable instructions", None, None
        )
        messages.extend(response.history_items)
        messages.add_tool("call_1", "3 units")
        continued = model._build_generation_params(
            messages, "Stable instructions", None, None
        )
        assert continued["input"][: len(initial["input"])] == initial["input"]

    def test_all_responses_models_infer_pretool_commentary(self, mock_openai_client):
        from msgflux.models.providers.openrouter import OpenRouterChatCompletion
        from msgflux.models import ChatModelCapabilities

        model = OpenRouterChatCompletion(
            model_id="deepseek/deepseek-v3.2", api_mode="responses"
        )
        assert model.supports_unphased_tool_commentary() is True
        disabled = OpenRouterChatCompletion(
            model_id="deepseek/deepseek-v3.2",
            api_mode="responses",
            model_capabilities=ChatModelCapabilities(unphased_tool_commentary=False),
        )
        assert disabled.supports_unphased_tool_commentary() is False

    def test_other_responses_provider_uses_pretool_commentary(self, monkeypatch):
        from msgflux.models.providers.groq import GroqChatCompletion

        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        model = GroqChatCompletion(model_id="custom-model", api_mode="responses")
        response = model._process_responses_model_output(
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "Checking inventory."}
                        ],
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "lookup_inventory",
                        "arguments": "{}",
                    },
                ],
            }
        )
        assert response.commentary == ["Checking inventory."]

    def test_phased_pretool_message_is_commentary_without_phase_guarantee(
        self, mock_openai_client
    ):
        from msgflux.models.providers.openrouter import OpenRouterChatCompletion

        model = OpenRouterChatCompletion(model_id="x-ai/grok-4.7", api_mode="responses")
        response = model._process_responses_model_output(
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "phase": "commentary",
                        "content": [
                            {"type": "output_text", "text": "Checking inventory."}
                        ],
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "lookup_inventory",
                        "arguments": "{}",
                    },
                ],
            }
        )
        assert response.commentary == ["Checking inventory."]
        assert response.history_items[0]["phase"] == "commentary"

    def test_responses_codec_preserves_clear_text_reasoning_item(
        self, mock_openai_client
    ):
        from msgflux.models.providers.openrouter import OpenRouterChatCompletion

        model = OpenRouterChatCompletion(
            model_id="deepseek/deepseek-v4-flash", api_mode="responses"
        )
        reasoning_item = {
            "type": "reasoning",
            "id": "rs_2",
            "content": [{"type": "reasoning_text", "text": "private thought"}],
        }
        response = model._process_responses_model_output(
            {
                "status": "completed",
                "output": [
                    reasoning_item,
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Done."}],
                    },
                ],
            }
        )

        assert response.reasoning == "private thought"
        assert response.reasoning_summary is None
        replay = ChatMessages(response.history_items).to_responses_input(
            provider="openrouter",
            api_mode="responses",
            reasoning_codec=model.reasoning_codec,
        )
        assert replay[0] == reasoning_item

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

    def test_chat_completion_tool_content_is_commentary_and_cache_hit(
        self, mock_openai_client
    ):
        from msgflux.models.providers.openrouter import OpenRouterChatCompletion

        client, _ = mock_openai_client
        client.return_value.chat.completions.create.return_value = SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="tool_calls",
                    logprobs=None,
                    message=SimpleNamespace(
                        content="Checking inventory.",
                        reasoning_content="private reasoning",
                        reasoning_details=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call_1",
                                function=SimpleNamespace(
                                    name="lookup_inventory",
                                    arguments='{"sku":"42"}',
                                ),
                            )
                        ],
                        audio=None,
                        annotations=None,
                    ),
                )
            ],
        )
        model = OpenRouterChatCompletion(
            model_id="z-ai/glm-5.3-flash", enable_cache=True
        )

        first = model("Check SKU-42")
        cached = model("Check SKU-42")

        assert client.return_value.chat.completions.create.call_count == 1
        assert first.metadata.timing.source == "provider"
        assert cached.metadata.timing.source == "cache"
        assert first.commentary == cached.commentary == ["Checking inventory."]
        assert first.reasoning == cached.reasoning == "private reasoning"
        assert first.history_items == cached.history_items
        from msgflux.nn.events import emit_model_response_events
        from msgflux.runtime.events import _CURRENT_EVENT_SINK, _EventSink

        events = []
        token = _CURRENT_EVENT_SINK.set(_EventSink(events.append))
        try:
            emit_model_response_events(cached)
        finally:
            _CURRENT_EVENT_SINK.reset(token)
        assert [(event.type, event.data.get("delta")) for event in events] == [
            ("commentary.delta", "Checking inventory."),
            ("reasoning.delta", "private reasoning"),
            ("model.response", None),
        ]
        replay = ChatMessages(cached.history_items).to_chatml(
            provider="openrouter",
            api_mode="chat_completions",
            reasoning_codec=model.reasoning_codec,
        )
        assert replay == [
            {
                "role": "assistant",
                "content": "Checking inventory.",
                "reasoning_content": "private reasoning",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup_inventory",
                            "arguments": '{"sku":"42"}',
                        },
                    }
                ],
            }
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("asynchronous", [False, True])
    async def test_chat_stream_tool_content_is_commentary(
        self, mock_openai_client, asynchronous
    ):
        from msgflux.models.providers.openrouter import OpenRouterChatCompletion
        from msgflux.models.response import ModelStreamResponse

        model = OpenRouterChatCompletion(model_id="z-ai/glm-5.3-flash")
        chunk = SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="tool_calls",
                    delta=SimpleNamespace(
                        content="Checking inventory.",
                        reasoning_content="private reasoning",
                        reasoning_details=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="call_1",
                                function=SimpleNamespace(
                                    name="lookup_inventory",
                                    arguments='{"sku":"42"}',
                                ),
                            )
                        ],
                        annotations=None,
                    ),
                )
            ],
        )
        response = ModelStreamResponse(mode="async" if asynchronous else "sync")
        if asynchronous:

            async def stream():
                yield chunk

            model._aexecute_model = AsyncMock(return_value=stream())
            await model._astream_chat_completions_generate(
                stream_response=response, tools=[{}]
            )
        else:
            model._execute_model = MagicMock(return_value=iter([chunk]))
            model._stream_chat_completions_generate(
                stream_response=response, tools=[{}]
            )

        assert response.error is None
        assert response.response_type == "tool_call"
        assert [event.type async for event in response.consume_events()] == [
            "reasoning.delta",
            "commentary.delta",
        ]
        history = response.chat_accumulator.snapshot()
        assert history[1]["visibility"] == "commentary"
        assert (
            ChatMessages(history).to_chatml(
                provider="openrouter",
                api_mode="chat_completions",
                reasoning_codec=model.reasoning_codec,
            )[0]["content"]
            == "Checking inventory."
        )

    def test_chat_stream_plain_content_with_tools_stays_answer(
        self, mock_openai_client
    ):
        from msgflux.models.providers.openrouter import OpenRouterChatCompletion
        from msgflux.models.response import ModelStreamResponse

        model = OpenRouterChatCompletion(model_id="z-ai/glm-5.3-flash")
        chunk = SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    delta=SimpleNamespace(
                        content="In stock.",
                        reasoning_content=None,
                        reasoning_details=None,
                        tool_calls=None,
                        annotations=None,
                    ),
                )
            ],
        )
        model._execute_model = MagicMock(return_value=iter([chunk]))
        response = ModelStreamResponse()
        model._stream_chat_completions_generate(stream_response=response, tools=[{}])

        assert response.error is None
        assert response.response_type == "text_generation"
        assert response.commentary == []
        assert response.data == "In stock."
