"""Tests for msgflux.models.providers.openai module."""

import os
from types import SimpleNamespace
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import msgspec
import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.exceptions import AbortRequestedError
from msgflux.generation.reasoning.react import ReAct
from msgflux.runtime import AbortSignal
from msgflux.runtime.context import execution_context
from msgflux.tools import ToolCatalogEntry, ToolCatalogView, ToolRef
from msgflux.tools.definitions import ToolCatalog, ToolSpec
from msgflux.tools.runtime import ToolOutcome
from tests.models._chat_transport import EndpointMockTransport


class TestOpenAIProviderImport:
    """Test OpenAI provider import and initialization."""

    def test_openai_import_available(self):
        """Test that OpenAI provider imports correctly when dependencies are available."""
        try:
            from msgflux.models.providers.openai import (
                OpenAIChatCompletion,
                OpenAIModeration,
                OpenAISpeechToText,
                OpenAITextEmbedder,
                OpenAITextToImage,
                OpenAITextToSpeech,
            )

            # If we get here, imports worked
            assert True
        except ImportError:
            pytest.skip("OpenAI dependencies not available")

    def test_openai_models_registered(self):
        """Test that OpenAI models are registered with @register_model."""

        from msgflux.models.registry import model_registry

        # Check if OpenAI models are registered
        if "chat_completion" in model_registry:
            assert "openai" in model_registry.get("chat_completion", {})

        if "text_to_speech" in model_registry:
            assert "openai" in model_registry.get("text_to_speech", {})


class TestOpenAIChatCompletion:
    """Test suite for OpenAIChatCompletion."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Setup environment variables for tests."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")

    @pytest.fixture
    def mock_openai_client(self):
        """Mock direct OpenAI chat and Responses endpoints."""
        from msgflux.models.providers.openai import OpenAIChatCompletion

        mock_client = MagicMock()
        mock_async_client = MagicMock()
        mock_async_client.return_value.close = AsyncMock()
        transport = EndpointMockTransport(
            mock_client.return_value,
            mock_async_client.return_value,
        )

        with patch.object(OpenAIChatCompletion, "chat_transport", transport):
            yield mock_client, mock_async_client

    def test_openai_defaults_to_direct_chat_transport(self):
        from msgflux.models.chat_transport import HTTPChatTransport
        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(model_id="gpt-5.6-luna")

        assert isinstance(model.chat_transport, HTTPChatTransport)

    def test_chat_completion_initialization(self, mock_openai_client):
        """Test OpenAIChatCompletion initialization."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(model_id="gpt-4")

        assert model.model_id == "gpt-4"
        assert model.provider == "openai"
        assert model.model_type == "chat_completion"
        assert model.api_mode == "responses"
        assert model.reasoning_codec.name == "openai_responses"
        assert not hasattr(model, "_native_client")
        assert not hasattr(model, "_native_aclient")

    def test_openai_is_a_concrete_compatible_provider(self, mock_openai_client):

        from msgflux.models.openai_compatible import OpenAICompatibleChatCompletion
        from msgflux.models.providers.openai import OpenAIChatCompletion

        assert issubclass(OpenAIChatCompletion, OpenAICompatibleChatCompletion)
        assert OpenAIChatCompletion is not OpenAICompatibleChatCompletion
        compatible = OpenAICompatibleChatCompletion(model_id="compatible")
        openai_responses = OpenAIChatCompletion(model_id="gpt-5.6-luna")
        openai_chat = OpenAIChatCompletion(
            model_id="gpt-5.6-luna", api_mode="chat_completions"
        )
        assert compatible.supports_native_compaction() is False
        assert openai_responses.supports_native_compaction() is True
        assert openai_chat.supports_native_compaction() is False

    def test_chat_completion_rejects_unsupported_api_mode(self, mock_openai_client):

        from msgflux.models.providers.openai import OpenAIChatCompletion

        with pytest.raises(ValueError, match="does not support"):
            OpenAIChatCompletion(model_id="gpt-4", api_mode="messages")

    def test_responses_counts_input_tokens_with_provider_endpoint(
        self, mock_openai_client
    ):
        from msgflux.models.providers.openai import OpenAIChatCompletion

        mock_client, _ = mock_openai_client
        mock_client.return_value.responses.input_tokens.count.return_value = (
            SimpleNamespace(input_tokens=321)
        )
        model = OpenAIChatCompletion(
            model_id="gpt-5.6-luna",
            context_length=400_000,
            speed="fast",
        )
        messages = ChatMessages([{"role": "user", "content": "Hello"}])

        estimate = model.count_context_tokens(
            messages,
            system_prompt="Be concise.",
        )

        assert estimate.input_tokens == 321
        assert estimate.source == "provider"
        assert model.context_capacity == 400_000
        kwargs = mock_client.return_value.responses.input_tokens.count.call_args.kwargs
        assert kwargs == {
            "model": "gpt-5.6-luna",
            "input": [{"type": "message", "role": "user", "content": "Hello"}],
            "instructions": "Be concise.",
        }

    def test_responses_native_compaction_preserves_opaque_output(
        self, mock_openai_client
    ):
        from msgflux.models.providers.openai import OpenAIChatCompletion

        mock_client, _ = mock_openai_client
        compacted_output = [
            {"type": "message", "role": "user", "content": "retained"},
            {"type": "compaction", "encrypted_content": "opaque"},
        ]
        mock_client.return_value.responses.compact.return_value = SimpleNamespace(
            output=compacted_output,
            usage={"input_tokens": 100, "output_tokens": 20},
        )
        model = OpenAIChatCompletion(
            model_id="gpt-5.6-luna",
            max_tokens=999,
            reasoning_effort="high",
            speed="fast",
            store=False,
        )

        compacted = model.compact_context(
            ChatMessages([{"role": "user", "content": "Long history"}]),
            system_prompt="Preserve facts.",
        )

        assert compacted.format == "provider"
        assert compacted.items == compacted_output
        assert compacted.provider == "openai"
        assert compacted.api_mode == "responses"
        assert compacted.usage["input_tokens"] == 100
        kwargs = mock_client.return_value.responses.compact.call_args.kwargs
        assert kwargs == {
            "model": "gpt-5.6-luna",
            "input": [{"type": "message", "role": "user", "content": "Long history"}],
            "instructions": "Preserve facts.",
            "service_tier": "fast",
        }
        assert "store" not in kwargs
        assert "reasoning" not in kwargs
        assert "max_output_tokens" not in kwargs
        assert "_native_client" not in model.serialize()["state"]
        model.close()

    @pytest.mark.asyncio
    async def test_responses_async_compaction_uses_async_endpoint(
        self, mock_openai_client
    ):
        from msgflux.models.providers.openai import OpenAIChatCompletion

        _, mock_async_client = mock_openai_client
        mock_async_client.return_value.responses.compact = AsyncMock(
            return_value=SimpleNamespace(
                output=[{"type": "compaction", "encrypted_content": "opaque"}],
                usage=None,
            )
        )
        model = OpenAIChatCompletion(model_id="gpt-5.6-luna")

        compacted = await model.acompact_context(
            ChatMessages([{"role": "user", "content": "History"}])
        )

        assert compacted.items == [
            {"type": "compaction", "encrypted_content": "opaque"}
        ]
        mock_async_client.return_value.responses.compact.assert_awaited_once()
        assert "_native_aclient" not in model.serialize()["state"]
        await model.aclose()

    @pytest.mark.parametrize(
        "effort", ["none", "low", "medium", "high", "xhigh", "max"]
    )
    def test_gpt_5_6_reasoning_efforts_are_forwarded(self, mock_openai_client, effort):

        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(model_id="gpt-5.6-luna", reasoning_effort=effort)

        params = model._adapt_responses_params(
            {**model.sampling_run_params, "model": model.model_id, "input": []}
        )

        assert params["reasoning"] == {"effort": effort, "summary": "auto"}

    def test_responses_reasoning_state_without_text_replays_empty_summary(
        self, mock_openai_client
    ):

        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(model_id="gpt-5.6-luna")
        history = ChatMessages(
            [
                {
                    "type": "reasoning",
                    "role": "assistant",
                    "provider_state": {
                        "provider": "openai",
                        "api_mode": "responses",
                        "codec": "openai_responses",
                        "data": {
                            "id": "rs_1",
                            "type": "reasoning",
                            "encrypted_content": "opaque",
                        },
                    },
                }
            ]
        )

        params = model._build_generation_params(
            history,
            system_prompt=None,
            prefilling=None,
            tool_catalog=None,
        )

        assert params["input"] == [
            {
                "id": "rs_1",
                "type": "reasoning",
                "encrypted_content": "opaque",
                "summary": [],
            }
        ]

    def test_responses_structured_tool_flow_prefers_commentary_phase(
        self, mock_openai_client
    ):

        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(model_id="gpt-5.6-luna")
        commentary = '{"thought":"call tool","actions":null,"final_answer":null}'
        final_answer = '{"thought":"","actions":null,"final_answer":"premature"}'
        output = SimpleNamespace(
            id="resp_phases",
            status="completed",
            incomplete_details=None,
            usage=None,
            output=[
                {
                    "type": "message",
                    "phase": "commentary",
                    "content": [{"type": "output_text", "text": commentary}],
                },
                {
                    "type": "message",
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": final_answer}],
                },
            ],
        )

        response = model._process_responses_model_output(
            output,
            generation_schema=ReAct,
            transport_generation_schema={"decoder_schema": None},
        )

        assert response.data == {
            "thought": "call tool",
            "actions": None,
            "final_answer": None,
        }
        assert [item["phase"] for item in response.history_items] == [
            "commentary",
            "final_answer",
        ]
        replay = ChatMessages(response.history_items).to_responses_input()
        assert [item["phase"] for item in replay] == [
            "commentary",
            "final_answer",
        ]

    def test_responses_mode_converts_frontend_and_preserves_reasoning_state(
        self, mock_openai_client
    ):

        from msgflux.models.providers.openai import OpenAIChatCompletion

        mock_client, _ = mock_openai_client
        reasoning_item = {
            "type": "reasoning",
            "id": "rs_1",
            "encrypted_content": "opaque",
            "summary": [{"type": "summary_text", "text": "Checked inventory."}],
        }
        mock_client.return_value.responses.create.return_value = SimpleNamespace(
            id="resp_1",
            status="completed",
            incomplete_details=None,
            usage={"input_tokens": 8, "output_tokens": 4, "total_tokens": 12},
            output=[
                reasoning_item,
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "In stock."}],
                },
            ],
        )
        model = OpenAIChatCompletion(
            model_id="gpt-5",
            api_mode="responses",
            max_tokens=256,
            reasoning_effort="medium",
            verbosity="low",
        )

        response = model("Is SKU-1842 available?", system_prompt="Be concise.")

        call_kwargs = mock_client.return_value.responses.create.call_args.kwargs
        assert call_kwargs["input"] == [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Is SKU-1842 available?"},
        ]
        assert call_kwargs["max_output_tokens"] == 256
        assert call_kwargs["reasoning"] == {"effort": "medium", "summary": "auto"}
        assert call_kwargs["include"] == ["reasoning.encrypted_content"]
        assert call_kwargs["text"] == {"verbosity": "low"}
        assert "messages" not in call_kwargs
        assert response.consume() == "In stock."
        assert response.reasoning is None
        assert response.reasoning_summary == "Checked inventory."
        assert response.consume_reasoning_summary() == "Checked inventory."
        assert response.metadata.response_id == "resp_1"
        assert response.metadata.model == {
            "provider": "openai",
            "model_id": "gpt-5",
            "api_mode": "responses",
            "reasoning_effort": "medium",
        }
        assert response.history_items == [
            {
                "type": "reasoning",
                "role": "assistant",
                "summary": "Checked inventory.",
                "provider_state": {
                    "provider": "openai",
                    "api_mode": "responses",
                    "codec": "openai_responses",
                    "data": {
                        "type": "reasoning",
                        "id": "rs_1",
                        "encrypted_content": "opaque",
                    },
                },
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "In stock."}],
                "provider_state": {
                    "provider": "openai",
                    "api_mode": "responses",
                    "data": {},
                },
            },
        ]
        assert ChatMessages(response.history_items).to_responses_input() == [
            reasoning_item,
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "In stock."}],
            },
        ]

    def test_non_stream_response_reports_provider_latency(self, mock_openai_client):

        from msgflux.models.providers.openai import OpenAIChatCompletion

        mock_client, _ = mock_openai_client
        mock_client.return_value.responses.create.return_value = SimpleNamespace(
            id="resp_timing",
            status="completed",
            incomplete_details=None,
            usage=None,
            output=[
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Done."}],
                }
            ],
        )
        model = OpenAIChatCompletion(model_id="gpt-5.6-luna")

        response = model("Finish")

        assert response.metadata.timing.source == "provider"
        assert response.metadata.timing.latency_ms >= 0
        assert "ttft_ms" not in response.metadata.timing

    def test_cache_hit_reports_lookup_latency_without_mutating_cached_metadata(
        self, mock_openai_client
    ):

        from msgflux.models.providers.openai import OpenAIChatCompletion

        mock_client, _ = mock_openai_client
        mock_client.return_value.responses.create.return_value = SimpleNamespace(
            id="resp_cached",
            status="completed",
            incomplete_details=None,
            usage=None,
            output=[
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Cached."}],
                }
            ],
        )
        model = OpenAIChatCompletion(
            model_id="gpt-5.6-luna",
            enable_cache=True,
        )

        provider_response = model("Repeat")
        cached_response = model("Repeat")

        assert provider_response is not cached_response
        assert provider_response.metadata.timing.source == "provider"
        assert cached_response.metadata.timing.source == "cache"
        assert cached_response.metadata.timing.latency_ms >= 0
        assert mock_client.return_value.responses.create.call_count == 1
        stored_response = next(iter(model._response_cache._cache.values()))
        assert stored_response.metadata.timing.source == "provider"

    def test_changing_speed_uses_a_distinct_response_cache_key(
        self, mock_openai_client
    ):
        from msgflux.models.providers.openai import OpenAIChatCompletion

        mock_client, _ = mock_openai_client
        mock_client.return_value.responses.create.return_value = SimpleNamespace(
            id="resp_cached",
            status="completed",
            incomplete_details=None,
            service_tier="priority",
            usage=None,
            output=[
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Cached."}],
                }
            ],
        )
        model = OpenAIChatCompletion(
            model_id="gpt-6-astra",
            enable_cache=True,
        )

        model("Repeat")
        model.set_speed("fast")
        model("Repeat")
        model("Repeat")

        assert mock_client.return_value.responses.create.call_count == 2

    def test_responses_stream_reports_ttft_for_first_text_delta(
        self, mock_openai_client
    ):

        from msgflux.models.providers.openai import OpenAIChatCompletion
        from msgflux.models.response import ModelStreamResponse
        from msgflux.models.timing import ModelRequestTimer

        ticks = iter([1_000_000, 4_000_000, 9_000_000])
        timer = ModelRequestTimer(clock_ns=lambda: next(ticks))
        mock_client, _ = mock_openai_client
        mock_client.return_value.responses.create.return_value = iter(
            [
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {"type": "message", "role": "assistant"},
                },
                {
                    "type": "response.output_text.delta",
                    "output_index": 0,
                    "delta": "Done.",
                },
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_stream",
                        "status": "completed",
                        "service_tier": "priority",
                    },
                },
            ]
        )
        model = OpenAIChatCompletion(model_id="gpt-5.6-luna", speed="fast")
        stream_response = ModelStreamResponse()

        model._stream_responses_generate(
            input=[{"role": "user", "content": "Finish"}],
            model="gpt-5.6-luna",
            stream=True,
            stream_response=stream_response,
            _request_timer=timer,
        )

        assert stream_response.metadata.timing == {
            "source": "provider",
            "latency_ms": 8.0,
            "ttft_ms": 3.0,
        }
        assert stream_response.metadata.model.requested_speed == "fast"
        assert stream_response.metadata.model.effective_speed == "priority"

    def test_responses_stream_omits_ttft_for_empty_protocol_events(
        self, mock_openai_client
    ):

        from msgflux.models.providers.openai import OpenAIChatCompletion
        from msgflux.models.response import ModelStreamResponse
        from msgflux.models.timing import ModelRequestTimer

        ticks = iter([1_000_000, 5_000_000])
        timer = ModelRequestTimer(clock_ns=lambda: next(ticks))
        mock_client, _ = mock_openai_client
        mock_client.return_value.responses.create.return_value = iter(
            [
                {"type": "response.created", "response": {"id": "resp_empty"}},
                {
                    "type": "response.completed",
                    "response": {"id": "resp_empty", "status": "completed"},
                },
            ]
        )
        model = OpenAIChatCompletion(model_id="gpt-5.6-luna")
        stream_response = ModelStreamResponse()

        model._stream_responses_generate(
            input=[{"role": "user", "content": "Finish"}],
            model="gpt-5.6-luna",
            stream=True,
            stream_response=stream_response,
            _request_timer=timer,
        )

        assert stream_response.metadata.timing == {
            "source": "provider",
            "latency_ms": 4.0,
        }

    @pytest.mark.parametrize("parameter,value", [("stop", ["END"]), ("audio", {})])
    def test_responses_mode_rejects_parameters_without_equivalent(
        self, mock_openai_client, parameter, value
    ):

        from msgflux.models.providers.openai import OpenAIChatCompletion

        with pytest.raises(ValueError, match=parameter):
            OpenAIChatCompletion(
                model_id="gpt-5",
                api_mode="responses",
                **{parameter: value},
            )

    @pytest.mark.parametrize("store", [False, True])
    def test_responses_mode_forwards_explicit_store_preference(
        self, mock_openai_client, store
    ):

        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(
            model_id="gpt-5",
            api_mode="responses",
            store=store,
        )

        params = model._adapt_responses_params(
            {**model.sampling_run_params, "model": model.model_id, "input": []}
        )

        assert params["store"] is store

    def test_responses_mode_omits_store_when_not_configured(self, mock_openai_client):

        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(model_id="gpt-5", api_mode="responses")

        params = model._adapt_responses_params(
            {**model.sampling_run_params, "model": model.model_id, "input": []}
        )

        assert "store" not in params

    def test_store_rejects_non_boolean_value(self, mock_openai_client):

        from msgflux.models.providers.openai import OpenAIChatCompletion

        with pytest.raises(TypeError, match="store"):
            OpenAIChatCompletion(model_id="gpt-5", store="false")

    def test_responses_mode_converts_tools_and_structured_output(
        self, mock_openai_client
    ):

        from msgflux.models.providers.openai import OpenAIChatCompletion

        class Availability(msgspec.Struct):
            available: bool

        mock_client, _ = mock_openai_client
        mock_client.return_value.responses.create.return_value = SimpleNamespace(
            id="resp_2",
            status="completed",
            incomplete_details=None,
            usage=None,
            output=[
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": '{"available":true}'}],
                }
            ],
        )
        model = OpenAIChatCompletion(model_id="gpt-5", api_mode="responses")
        tools = ToolCatalogView(
            library_id="warehouse_tools",
            thread_id="thread_1",
            entries=(
                ToolCatalogEntry(
                    ref=ToolRef(
                        library_id="warehouse_tools",
                        tool_id="lookup_inventory",
                    ),
                    description="Look up a SKU.",
                    input_schema={"type": "object", "properties": {}},
                    strict=True,
                ),
            ),
            choice="lookup_inventory",
        )

        response = model(
            "Check SKU-1842",
            generation_schema=Availability,
            tool_catalog=tools,
        )

        call_kwargs = mock_client.return_value.responses.create.call_args.kwargs
        assert call_kwargs["tools"] == [
            {
                "type": "function",
                "name": "lookup_inventory",
                "description": "Look up a SKU.",
                "parameters": {"type": "object", "properties": {}},
                "strict": True,
            }
        ]
        assert call_kwargs["tool_choice"] == {
            "type": "function",
            "name": "lookup_inventory",
        }
        assert call_kwargs["text"]["format"]["type"] == "json_schema"
        assert "json_schema" not in call_kwargs["text"]["format"]
        assert response.consume() == {"available": True}

    def test_responses_mode_compiles_deferred_tools_for_hosted_search(
        self, mock_openai_client
    ):

        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(model_id="gpt-5.6", api_mode="responses")
        catalog = ToolCatalogView(
            library_id="warehouse_tools",
            thread_id="thread_1",
            entries=(
                ToolCatalogEntry(
                    ref=ToolRef(
                        library_id="warehouse_tools",
                        tool_id="lookup_inventory",
                    ),
                    description="Look up a SKU.",
                    input_schema={"type": "object", "properties": {}},
                    deferred=True,
                ),
            ),
        )

        params = model._build_generation_params(
            messages="Check SKU-1842",
            system_prompt=None,
            prefilling=None,
            tool_catalog=catalog,
        )

        assert params["tools"] == [
            {"type": "tool_search"},
            {
                "type": "function",
                "name": "lookup_inventory",
                "description": "Look up a SKU.",
                "parameters": {"type": "object", "properties": {}},
                "strict": False,
                "defer_loading": True,
            },
        ]

    @pytest.mark.parametrize(
        ("model_id", "api_mode"),
        [
            ("gpt-5.6", "chat_completions"),
            ("gpt-4.1-mini", "responses"),
            ("custom-gateway-model", "responses"),
        ],
    )
    def test_deferred_tools_use_portable_search_without_native_model_support(
        self,
        mock_openai_client,
        model_id,
        api_mode,
    ):

        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(model_id=model_id, api_mode=api_mode)
        catalog = ToolCatalog(
            tools=[
                ToolSpec(
                    name="lookup_inventory",
                    description="Look up a SKU.",
                    parameters={"type": "object", "properties": {}},
                    defer_loading=True,
                )
            ],
            catalog_id="warehouse_tools",
            search_tool=ToolSpec(
                name="tool_search",
                description="Search tools.",
                parameters={"type": "object", "properties": {}},
            ),
        )

        assert model.supports_native_tool_search() is False
        if api_mode == "responses":
            params = model._build_generation_params(
                messages="Check SKU-1842",
                system_prompt=None,
                prefilling=None,
                tool_catalog=catalog,
            )
            assert [tool["name"] for tool in params["tools"]] == ["tool_search"]

    def test_responses_mode_preserves_hosted_tool_search_items(
        self, mock_openai_client
    ):

        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(model_id="gpt-5.4", api_mode="responses")
        search_call = {
            "type": "tool_search_call",
            "id": "ts_1",
            "status": "completed",
            "arguments": {"query": "inventory"},
        }
        search_output = {
            "type": "tool_search_output",
            "id": "tso_1",
            "tool_search_call_id": "ts_1",
            "tools": [{"type": "function", "name": "lookup_inventory"}],
        }
        function_call = {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "lookup_inventory",
            "arguments": '{"sku":"1842"}',
            "status": "completed",
        }

        response = model._process_responses_model_output(
            SimpleNamespace(
                id="resp_search",
                status="completed",
                incomplete_details=None,
                usage=None,
                output=[search_call, search_output, function_call],
            )
        )

        assert [
            item["provider_state"]["data"] for item in response.history_items[:2]
        ] == [search_call, search_output]
        assert ChatMessages(response.history_items).to_responses_input(
            provider="openai", api_mode="responses"
        )[:2] == [
            search_call,
            search_output,
        ]
        assert response.consume().get_calls()[0][1] == "lookup_inventory"
        intent = response.get_tool_intents()[0]
        assert response.render_tool_outcomes(
            [ToolOutcome.completed(intent, "available")]
        ) == [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "available",
            }
        ]

    @pytest.mark.asyncio
    async def test_responses_stream_accumulates_summary_text_and_tool_call(
        self, mock_openai_client
    ):

        from msgflux.models.providers.openai import OpenAIChatCompletion
        from msgflux.models.response import ModelStreamResponse

        mock_client, _ = mock_openai_client
        reasoning_item = {
            "type": "reasoning",
            "id": "rs_1",
            "encrypted_content": "opaque",
            "summary": [{"type": "summary_text", "text": "Need inventory."}],
        }
        mock_client.return_value.responses.create.return_value = iter(
            [
                {
                    "type": "response.reasoning_summary_text.delta",
                    "delta": "Need inventory.",
                },
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": reasoning_item,
                },
                {
                    "type": "response.output_item.added",
                    "output_index": 1,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "status": "in_progress",
                        "call_id": "call_1",
                        "name": "lookup_inventory",
                        "arguments": "",
                    },
                },
                {
                    "type": "response.function_call_arguments.delta",
                    "output_index": 1,
                    "delta": '{"sku":"1842"}',
                },
                {
                    "type": "response.output_item.done",
                    "output_index": 1,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "status": "completed",
                        "call_id": "call_1",
                        "name": "lookup_inventory",
                        "arguments": '{"sku":"1842"}',
                    },
                },
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_3",
                        "status": "completed",
                        "usage": {"total_tokens": 20},
                    },
                },
            ]
        )
        model = OpenAIChatCompletion(model_id="gpt-5", api_mode="responses")
        stream_response = ModelStreamResponse()

        model._stream_responses_generate(
            input=[{"role": "user", "content": "Check SKU-1842"}],
            model="gpt-5",
            stream=True,
            stream_response=stream_response,
        )

        assert stream_response.response_type == "tool_call"
        assert stream_response.reasoning is None
        assert stream_response.reasoning_summary == "Need inventory."
        assert stream_response.data.get_calls() == [
            ("call_1", "lookup_inventory", {"sku": "1842"})
        ]
        assert stream_response.data.api_mode == "responses"
        assert stream_response.metadata.response_id == "resp_3"
        assert stream_response.chat_accumulator.snapshot()[0] == {
            "type": "reasoning",
            "role": "assistant",
            "summary": "Need inventory.",
            "provider_state": {
                "provider": "openai",
                "api_mode": "responses",
                "codec": "openai_responses",
                "data": {
                    "type": "reasoning",
                    "id": "rs_1",
                    "encrypted_content": "opaque",
                },
            },
        }
        function_call = stream_response.chat_accumulator.snapshot()[1]
        assert function_call["provider_state"] == {
            "provider": "openai",
            "api_mode": "responses",
            "data": {
                "type": "function_call",
                "id": "fc_1",
                "status": "completed",
                "call_id": "call_1",
                "name": "lookup_inventory",
                "arguments": '{"sku":"1842"}',
            },
        }
        assert ChatMessages([function_call]).to_responses_input(
            provider="openai",
            api_mode="responses",
            reasoning_codec=model.reasoning_codec,
        ) == [
            {
                "type": "function_call",
                "id": "fc_1",
                "status": "completed",
                "call_id": "call_1",
                "name": "lookup_inventory",
                "arguments": '{"sku":"1842"}',
            }
        ]
        events = [event async for event in stream_response.consume_events()]
        assert [(event.type, event.data) for event in events] == [
            ("reasoning_summary.delta", "Need inventory."),
        ]

    def test_responses_stream_preserves_message_phase_and_native_identity(
        self, mock_openai_client
    ):

        from msgflux.models.providers.openai import OpenAIChatCompletion
        from msgflux.models.response import ModelStreamResponse

        mock_client, _ = mock_openai_client
        final_message = {
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "phase": "final_answer",
            "content": [{"type": "output_text", "text": "Done."}],
        }
        mock_client.return_value.responses.create.return_value = iter(
            [
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        **final_message,
                        "status": "in_progress",
                        "content": [],
                    },
                },
                {
                    "type": "response.output_text.delta",
                    "output_index": 0,
                    "delta": "Done.",
                },
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": final_message,
                },
                {
                    "type": "response.completed",
                    "response": {"id": "resp_1", "status": "completed"},
                },
            ]
        )
        model = OpenAIChatCompletion(model_id="gpt-5.6-luna")
        stream_response = ModelStreamResponse()

        model._stream_responses_generate(
            input=[{"role": "user", "content": "Finish"}],
            model="gpt-5.6-luna",
            stream=True,
            stream_response=stream_response,
        )

        assert stream_response.data == "Done."
        items = stream_response.chat_accumulator.snapshot()
        assert len(items) == 1
        assert items[0]["phase"] == "final_answer"
        assert items[0]["provider_state"]["data"] == {
            "id": "msg_1",
            "status": "completed",
        }
        assert ChatMessages(items).to_responses_input() == [final_message]

    def test_chat_completion_with_parameters(self, mock_openai_client):
        """Test OpenAIChatCompletion with custom parameters."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(
            model_id="gpt-4",
            max_tokens=1000,
            temperature=0.7,
            top_p=0.9,
        )

        assert model.sampling_run_params["max_tokens"] == 1000
        assert model.sampling_run_params["temperature"] == 0.7
        assert model.sampling_run_params["top_p"] == 0.9

    def test_chat_completion_with_extra_body(self, mock_openai_client):
        """Test provider-specific OpenAI-compatible request extensions."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        extra_body = {"enable_citations": True, "enable_entities": True}
        model = OpenAIChatCompletion(model_id="gpt-4", extra_body=extra_body)

        assert model.sampling_run_params["extra_body"] == extra_body
        assert model.sampling_run_params["extra_body"] is not extra_body

    def test_chat_completion_with_extra_body_kwargs(self, mock_openai_client):
        """Test provider-specific fields passed directly as kwargs."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(
            model_id="gpt-4",
            enable_entities=True,
            enable_citations=True,
        )

        assert model.sampling_run_params["extra_body"] == {
            "enable_entities": True,
            "enable_citations": True,
        }

    def test_chat_completion_aborts_from_execution_context(self, mock_openai_client):
        """AbortSignal is ambient runtime control, not request payload."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        mock_client, _mock_async_client = mock_openai_client
        model = OpenAIChatCompletion(model_id="gpt-4")
        abort_signal = AbortSignal()
        abort_signal.abort("user pressed esc")

        with execution_context(abort_signal=abort_signal):
            with pytest.raises(AbortRequestedError, match="user pressed esc"):
                model("hello")

        mock_client.return_value.chat.completions.create.assert_not_called()

    def test_chat_completion_merges_extra_body_and_kwargs(self, mock_openai_client):
        """Test init merges extra_body dict with direct provider kwargs."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(
            model_id="gpt-4",
            extra_body={"enable_citations": True},
            enable_entities=True,
        )

        assert model.sampling_run_params["extra_body"] == {
            "enable_citations": True,
            "enable_entities": True,
        }

    def test_chat_completion_rejects_duplicated_extra_body_keys(
        self, mock_openai_client
    ):
        """Test duplicated keys between extra_body and kwargs raise error."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        with pytest.raises(
            ValueError,
            match="Duplicate provider extra-body keys",
        ):
            OpenAIChatCompletion(
                model_id="gpt-4",
                extra_body={"enable_citations": True},
                enable_citations=False,
            )

    def test_chat_completion_forwards_extra_body(self, mock_openai_client):
        """Test extra_body is forwarded to the OpenAI-compatible client."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        mock_client, _ = mock_openai_client
        mock_client.return_value.chat.completions.create.return_value = SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content="done",
                        tool_calls=None,
                        audio=None,
                        annotations=None,
                    ),
                )
            ],
        )
        model = OpenAIChatCompletion(
            model_id="gpt-4",
            api_mode="chat_completions",
            extra_body={"enable_citations": True, "enable_entities": True},
        )
        model("Hello")

        call_kwargs = mock_client.return_value.chat.completions.create.call_args.kwargs
        assert call_kwargs["extra_body"] == {
            "enable_citations": True,
            "enable_entities": True,
        }

    def test_tool_call_reasoning_is_kept_in_history_when_not_returned(
        self, mock_openai_client
    ):

        from msgflux.models.providers.openai import OpenAIChatCompletion

        mock_client, _ = mock_openai_client
        mock_client.return_value.chat.completions.create.return_value = SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content="done",
                        reasoning_content="private reasoning",
                        tool_calls=None,
                        audio=None,
                        annotations=None,
                    ),
                )
            ],
        )
        model = OpenAIChatCompletion(
            model_id="gpt-4",
            api_mode="chat_completions",
            return_reasoning=False,
            reasoning_in_tool_call=True,
        )

        response = model("Hello")

        assert response.reasoning is None
        assert response.history_items == [
            {
                "type": "reasoning",
                "role": "assistant",
                "text": "private reasoning",
            }
        ]

    def test_chat_completion_forwards_extra_body_kwargs(self, mock_openai_client):
        """Test direct provider kwargs are forwarded through extra_body."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        mock_client, _ = mock_openai_client
        mock_client.return_value.chat.completions.create.return_value = SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content="done",
                        tool_calls=None,
                        audio=None,
                        annotations=None,
                    ),
                )
            ],
        )
        model = OpenAIChatCompletion(
            model_id="gpt-4",
            api_mode="chat_completions",
            enable_entities=True,
            enable_citations=True,
        )
        model("Hello")

        call_kwargs = mock_client.return_value.chat.completions.create.call_args.kwargs
        assert call_kwargs["extra_body"] == {
            "enable_entities": True,
            "enable_citations": True,
        }

    def test_chat_completion_call_merges_extra_body_kwargs(self, mock_openai_client):
        """Test runtime provider kwargs merge with init extra_body."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        mock_client, _ = mock_openai_client
        mock_client.return_value.chat.completions.create.return_value = SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content="done",
                        tool_calls=None,
                        audio=None,
                        annotations=None,
                    ),
                )
            ],
        )
        model = OpenAIChatCompletion(
            model_id="gpt-4",
            api_mode="chat_completions",
            extra_body={"enable_citations": True, "country": "BR"},
        )
        model(
            "Hello",
            extra_body={"country": "US"},
            enable_entities=True,
        )

        call_kwargs = mock_client.return_value.chat.completions.create.call_args.kwargs
        assert call_kwargs["extra_body"] == {
            "enable_citations": True,
            "country": "US",
            "enable_entities": True,
        }

    def test_chat_completion_call_rejects_duplicated_extra_body_keys(
        self, mock_openai_client
    ):
        """Test runtime extra_body and direct provider kwargs cannot duplicate keys."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(model_id="gpt-4")

        with pytest.raises(
            ValueError,
            match="Duplicate provider extra-body keys",
        ):
            model(
                "Hello",
                extra_body={"enable_citations": True},
                enable_citations=False,
            )

    def test_chat_completion_missing_api_key(self, monkeypatch):
        """Test that missing API key raises ValueError."""

        # Remove API key
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        from msgflux.models.providers.openai import OpenAIChatCompletion

        with pytest.raises(ValueError, match="OpenAI key is not available"):
            OpenAIChatCompletion(model_id="gpt-4")

    def test_chat_completion_with_reasoning_params(self, mock_openai_client):
        """Test OpenAIChatCompletion with reasoning parameters."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(
            model_id="o1-preview",
            reasoning_effort="high",
            enable_thinking=True,
            return_reasoning=True,
        )

        assert model.sampling_run_params.get("reasoning_effort") == "high"
        assert model.enable_thinking is True
        assert model.return_reasoning is True

    def test_set_reasoning_effort_updates_future_requests_and_metadata(
        self, mock_openai_client
    ):
        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(
            model_id="gpt-6-astra",
            reasoning_effort="medium",
        )

        result = model.set_reasoning_effort(" max ")
        params = model._adapt_responses_params(
            {**model.sampling_run_params, "model": model.model_id, "input": []}
        )

        assert result is model
        assert params["reasoning"]["effort"] == "max"
        assert model._build_response_metadata(None).model.reasoning_effort == "max"

        model.set_reasoning_effort(None)

        assert "reasoning_effort" not in model.sampling_run_params
        assert "reasoning_effort" not in model._build_response_metadata(None).model

    @pytest.mark.parametrize(
        ("model_id", "speed"),
        [("gpt-6-astra", "fast"), ("gpt-5.6-sol", "ultrafast")],
    )
    def test_speed_extension_maps_to_openai_service_tier(
        self, mock_openai_client, model_id, speed
    ):
        from msgflux.models.providers.openai import OpenAIChatCompletion

        mock_client, _ = mock_openai_client
        model = OpenAIChatCompletion(model_id=model_id, speed=speed)

        model._execute_model(model=model.model_id, input=[])

        request = mock_client.return_value.responses.create.call_args.kwargs
        assert request["service_tier"] == speed
        assert model.chat_settings == {"speed": speed}

    def test_speed_extension_supports_chat_completions(self, mock_openai_client):
        from msgflux.models.providers.openai import OpenAIChatCompletion

        mock_client, _ = mock_openai_client
        model = OpenAIChatCompletion(
            model_id="gpt-5.6-sol",
            api_mode="chat_completions",
            speed="fast",
        )

        model._execute_model(model=model.model_id, messages=[])

        request = mock_client.return_value.chat.completions.create.call_args.kwargs
        assert request["service_tier"] == "fast"

    def test_init_appends_custom_extension_after_provider_extensions(
        self, mock_openai_client
    ):
        from msgflux.models import ChatModelExtension
        from msgflux.models.providers.openai import OpenAIChatCompletion

        class CustomExtension(ChatModelExtension):
            name = "custom"

        custom = CustomExtension()
        model = OpenAIChatCompletion(
            model_id="gpt-6-astra",
            chat_extensions=[custom],
        )

        assert [extension.name for extension in model.chat_extensions] == [
            "speed",
            "custom",
        ]
        assert model.get_chat_extension("custom") is custom

    def test_openai_warns_and_ignores_nitro_speed(self, mock_openai_client):
        from msgflux.models.providers.openai import OpenAIChatCompletion

        with pytest.warns(UserWarning, match="does not support.*nitro"):
            model = OpenAIChatCompletion(model_id="gpt-6-astra", speed="nitro")

        assert model.chat_settings == {}

    def test_openai_warns_for_unsupported_ultrafast_model(self, mock_openai_client):
        from msgflux.models.providers.openai import OpenAIChatCompletion

        with pytest.warns(UserWarning, match="does not support.*ultrafast"):
            model = OpenAIChatCompletion(
                model_id="gpt-6-astra",
                speed="ultrafast",
            )

        assert model.chat_settings == {}

    def test_response_metadata_records_requested_and_effective_speed(
        self, mock_openai_client
    ):
        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(model_id="gpt-6-astra", speed="fast")

        metadata = model._build_response_metadata(
            SimpleNamespace(service_tier="priority", usage=None)
        )

        assert metadata.model.requested_speed == "fast"
        assert metadata.model.effective_speed == "priority"

    def test_requested_speed_is_serialized_with_model_state(self, mock_openai_client):
        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(model_id="gpt-6-astra", speed="fast")

        state = model.serialize()["state"]

        assert state["chat_settings"] == {"speed": "fast"}
        msgspec.json.encode(state)

    @pytest.mark.parametrize("reasoning_effort", ["", "   ", 1, True])
    def test_set_reasoning_effort_rejects_invalid_values(
        self, mock_openai_client, reasoning_effort
    ):
        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(model_id="gpt-6-astra")

        with pytest.raises(TypeError, match="non-empty string or None"):
            model.set_reasoning_effort(reasoning_effort)

    def test_chat_completion_with_prompt_cache_retention(self, mock_openai_client):
        """Test OpenAIChatCompletion with OpenAI-only prompt cache retention."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(
            model_id="gpt-4",
            prompt_cache_retention="24h",
        )

        assert model.sampling_run_params["prompt_cache_retention"] == "24h"

    def test_chat_completion_with_logprobs_params(self, mock_openai_client):
        """Test OpenAIChatCompletion with logprobs parameters."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(
            model_id="gpt-4",
            logprobs=True,
            top_logprobs=2,
        )

        assert model.sampling_run_params["logprobs"] is True
        assert model.sampling_run_params["top_logprobs"] == 2

    def test_chat_completion_call_forwards_logprobs_params(self, mock_openai_client):
        """Test runtime logprobs parameters are forwarded on sync calls."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        mock_client, _ = mock_openai_client
        mock_client.return_value.chat.completions.create.return_value = SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content="done",
                        tool_calls=None,
                        audio=None,
                        annotations=None,
                    ),
                )
            ],
        )
        model = OpenAIChatCompletion(model_id="gpt-4", api_mode="chat_completions")
        model("Hello", logprobs=True, top_logprobs=2)

        call_kwargs = mock_client.return_value.chat.completions.create.call_args.kwargs
        assert call_kwargs["logprobs"] is True
        assert call_kwargs["top_logprobs"] == 2

    @pytest.mark.asyncio
    async def test_acall_forwards_logprobs_params(self, mock_openai_client):
        """Test runtime logprobs parameters are forwarded on async calls."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        _, mock_async_client = mock_openai_client
        mock_async_client.return_value.chat.completions.create = AsyncMock(
            return_value=SimpleNamespace(
                usage=None,
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(
                            content="done",
                            tool_calls=None,
                            audio=None,
                            annotations=None,
                        ),
                    )
                ],
            )
        )
        model = OpenAIChatCompletion(model_id="gpt-4", api_mode="chat_completions")
        await model.acall("Hello", logprobs=True, top_logprobs=2)

        call_kwargs = (
            mock_async_client.return_value.chat.completions.create.await_args.kwargs
        )
        assert call_kwargs["logprobs"] is True
        assert call_kwargs["top_logprobs"] == 2

    @pytest.mark.asyncio
    async def test_acall_forwards_extra_body_kwargs(self, mock_openai_client):
        """Test runtime provider kwargs are forwarded on async calls."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        _, mock_async_client = mock_openai_client
        mock_async_client.return_value.chat.completions.create = AsyncMock(
            return_value=SimpleNamespace(
                usage=None,
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(
                            content="done",
                            tool_calls=None,
                            audio=None,
                            annotations=None,
                        ),
                    )
                ],
            )
        )
        model = OpenAIChatCompletion(
            model_id="gpt-4",
            api_mode="chat_completions",
            extra_body={"enable_citations": True},
        )
        await model.acall("Hello", enable_entities=True)

        call_kwargs = (
            mock_async_client.return_value.chat.completions.create.await_args.kwargs
        )
        assert call_kwargs["extra_body"] == {
            "enable_citations": True,
            "enable_entities": True,
        }

    def test_chat_completion_adapt_params(self, mock_openai_client):
        """Test parameter adaptation for OpenAI."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(model_id="gpt-4", max_tokens=100)

        # Test adaptation
        params = {"max_tokens": 100, "messages": []}
        adapted = model._adapt_params(params)

        assert "max_completion_tokens" in adapted
        assert "max_tokens" not in adapted

    def test_chat_completion_rejects_top_logprobs_without_logprobs(
        self, mock_openai_client
    ):
        """Test top_logprobs requires logprobs=True at call time."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(model_id="gpt-4")

        with pytest.raises(ValueError, match="`top_logprobs` requires"):
            model("Hello", top_logprobs=2)

    @pytest.mark.asyncio
    async def test_acall_stream_strips_tool_catalog_before_transport(
        self, mock_openai_client
    ):
        """Streaming async calls should not pass tool_catalog to the transport."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        _, mock_async_client = mock_openai_client

        async def empty_stream():
            if False:
                yield None

        create = AsyncMock(return_value=empty_stream())
        mock_async_client.return_value.chat.completions.create = create

        model = OpenAIChatCompletion(model_id="gpt-4", api_mode="chat_completions")

        await model.acall(
            messages=[{"role": "user", "content": "Check order 123"}],
            stream=True,
            tool_catalog=ToolCatalog.from_function_schemas(
                schemas=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_order_status",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "order_id": {"type": "string"},
                                },
                                "required": ["order_id"],
                            },
                        },
                    }
                ],
                choice="auto",
            ),
        )

        create.assert_awaited_once()
        call_kwargs = create.await_args.kwargs
        assert "tool_catalog" not in call_kwargs
        assert call_kwargs["stream"] is True
        assert call_kwargs["tools"][0]["function"]["name"] == "get_order_status"
        assert call_kwargs["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_acall_stream_surfaces_backend_error_on_consume(
        self, mock_openai_client
    ):
        """Streaming async calls should expose provider failures to consumers."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        _, mock_async_client = mock_openai_client
        mock_async_client.return_value.chat.completions.create = AsyncMock(
            side_effect=TypeError(
                "AsyncCompletions.create() got an unexpected keyword argument "
                "'tool_catalog'"
            )
        )

        model = OpenAIChatCompletion(model_id="gpt-4", api_mode="chat_completions")
        stream_response = await model.acall(
            messages=[{"role": "user", "content": "Check order 123"}],
            stream=True,
            tool_catalog=ToolCatalog.from_function_schemas(
                schemas=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_order_status",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "order_id": {"type": "string"},
                                },
                                "required": ["order_id"],
                            },
                        },
                    }
                ],
                choice="auto",
            ),
        )

        with pytest.raises(
            TypeError,
            match="unexpected keyword argument 'tool_catalog'",
        ):
            async for _ in stream_response.consume():
                pass

    @pytest.mark.asyncio
    async def test_acall_stream_accumulates_response_data(self, mock_openai_client):
        """Streaming async text responses should leave the full payload in response.data."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        _, mock_async_client = mock_openai_client

        async def text_stream():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content="Hello",
                            tool_calls=None,
                            annotations=None,
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=" world",
                            tool_calls=None,
                            annotations=None,
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            )
            yield SimpleNamespace(
                # OpenRouter includes usage on a final chunk that can still
                # contain a choice, so usage cannot live in an ``elif`` branch.
                service_tier="priority",
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            tool_calls=None,
                            annotations=None,
                        ),
                        finish_reason=None,
                    )
                ],
                usage=SimpleNamespace(
                    to_dict=lambda: {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5,
                        "prompt_tokens_details": {"cached_tokens": 2},
                    }
                ),
            )

        mock_async_client.return_value.chat.completions.create = AsyncMock(
            return_value=text_stream()
        )

        model = OpenAIChatCompletion(
            model_id="gpt-4",
            api_mode="chat_completions",
            speed="fast",
        )
        response = await model.acall(
            messages=[{"role": "user", "content": "Say hello"}],
            stream=True,
        )

        chunks = []
        async for chunk in response.consume():
            chunks.append(chunk)

        assert chunks == ["Hello", " world"]
        assert response.metadata.model.requested_speed == "fast"
        assert response.metadata.model.effective_speed == "priority"
        assert response.response_type == "text_generation"
        assert response.data == "Hello world"
        assert response.metadata.usage["total_tokens"] == 5
        assert response.metadata.usage["input_tokens"] == 3
        assert response.metadata.usage["output_tokens"] == 2
        assert response.metadata.usage["input_tokens_details"]["cached_tokens"] == 2
        assert response.metadata.usage["raw"]["prompt_tokens"] == 3
        assert response.metadata.model == {
            "provider": "openai",
            "model_id": "gpt-4",
            "api_mode": "chat_completions",
            "requested_speed": "fast",
            "effective_speed": "priority",
        }

    def test_prepare_generate_kwargs_lowers_dict_schema(self, mock_openai_client):
        """Test OpenAI transport schema lowering for dict-based structured outputs."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        class DictOutput(msgspec.Struct):
            entities: List[Dict[str, str]]

        model = OpenAIChatCompletion(model_id="gpt-4")
        kwargs = {"generation_schema": DictOutput}

        generation_schema, transport_generation_schema = model._prepare_generate_kwargs(
            kwargs
        )

        assert generation_schema is DictOutput
        assert transport_generation_schema is not DictOutput
        assert (
            kwargs["response_format"]["json_schema"]["schema"]["properties"][
                "entities"
            ]["items"]["properties"]["entries"]["type"]
            == "array"
        )

    def test_build_generation_params_uses_tool_catalog(self, mock_openai_client):
        """Test native tool calling is derived from ToolCatalog."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(model_id="gpt-4", api_mode="chat_completions")
        params = model._build_generation_params(
            messages=[{"role": "user", "content": "What's the weather?"}],
            system_prompt=None,
            prefilling=None,
            tool_catalog=ToolCatalogView(
                library_id="weather_tools",
                thread_id="thread_1",
                entries=(
                    ToolCatalogEntry(
                        ref=ToolRef(
                            library_id="weather_tools",
                            tool_id="get_weather",
                        ),
                        description=None,
                        input_schema={
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    ),
                ),
                choice="get_weather",
            ),
        )

        assert params["tools"][0]["function"]["name"] == "get_weather"
        assert params["tool_choice"] == {
            "type": "function",
            "function": {"name": "get_weather"},
        }
        assert params["parallel_tool_calls"] is model.parallel_tool_calls

    def test_build_generation_params_does_not_mutate_messages(self, mock_openai_client):
        """Provider-side system prompt injection must not mutate caller history."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(model_id="gpt-4", api_mode="chat_completions")
        history = [{"role": "user", "content": "Hello"}]

        params = model._build_generation_params(
            messages=history,
            system_prompt="You are helpful.",
            prefilling=None,
            tool_catalog=None,
        )

        assert history == [{"role": "user", "content": "Hello"}]
        assert params["messages"][0]["role"] == "system"
        assert params["messages"][1] == history[0]

    def test_call_prefilling_does_not_mutate_messages(self, mock_openai_client):
        """Provider-side prefilling must not mutate caller history."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        mock_client, _ = mock_openai_client
        mock_client.return_value.chat.completions.create.return_value = SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content="done",
                        tool_calls=None,
                        audio=None,
                        annotations=None,
                    ),
                )
            ],
        )

        model = OpenAIChatCompletion(model_id="gpt-4", api_mode="chat_completions")
        history = [{"role": "user", "content": "Hello"}]

        response = model(messages=history, prefilling="Start here")

        assert response.data == "done"
        assert history == [{"role": "user", "content": "Hello"}]

    @pytest.mark.asyncio
    async def test_acall_prefilling_does_not_mutate_messages(self, mock_openai_client):
        """Async provider-side prefilling must not mutate caller history."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        _, mock_async_client = mock_openai_client
        mock_async_client.return_value.chat.completions.create = AsyncMock(
            return_value=SimpleNamespace(
                usage=None,
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(
                            content="done",
                            tool_calls=None,
                            audio=None,
                            annotations=None,
                        ),
                    )
                ],
            )
        )

        model = OpenAIChatCompletion(model_id="gpt-4", api_mode="chat_completions")
        history = [{"role": "user", "content": "Hello"}]

        response = await model.acall(messages=history, prefilling="Start here")

        assert response.data == "done"
        assert history == [{"role": "user", "content": "Hello"}]

    def test_process_completion_model_output_restores_dict_shape(
        self, mock_openai_client
    ):
        """Test transport-schema decoding is restored to the logical dict shape."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        class DictOutput(msgspec.Struct):
            entities: List[Dict[str, str]]

        model = OpenAIChatCompletion(model_id="gpt-4")
        transport_generation_schema = model._prepare_generate_kwargs(
            {"generation_schema": DictOutput}
        )[1]

        model_output = SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content='{"entities":[{"entries":[{"key":"name","value":"Apple"},{"key":"type","value":"Organization"}]}]}',
                        tool_calls=None,
                        audio=None,
                        annotations=None,
                    ),
                )
            ],
        )

        response = model._process_completion_model_output(
            model_output,
            generation_schema=DictOutput,
            transport_generation_schema=transport_generation_schema,
        )

        assert response.response_type == "structured"
        assert response.data == {
            "entities": [{"name": "Apple", "type": "Organization"}]
        }

    def test_process_completion_model_output_includes_logprobs_metadata(
        self, mock_openai_client
    ):
        """Test logprobs are surfaced in response metadata."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(model_id="gpt-4")

        model_output = SimpleNamespace(
            usage=SimpleNamespace(
                to_dict=lambda: {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                }
            ),
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    logprobs=SimpleNamespace(
                        model_dump=lambda: {
                            "content": [
                                {
                                    "token": "Hello",
                                    "logprob": -0.1,
                                    "bytes": [72, 101, 108, 108, 111],
                                    "top_logprobs": [
                                        {"token": "Hello", "logprob": -0.1},
                                        {"token": "Hi", "logprob": -1.0},
                                    ],
                                }
                            ]
                        }
                    ),
                    message=SimpleNamespace(
                        content="Hello",
                        tool_calls=None,
                        audio=None,
                        annotations=None,
                    ),
                )
            ],
        )

        response = model._process_completion_model_output(model_output)

        assert response.metadata.usage["total_tokens"] == 5
        assert response.metadata.logprobs["content"][0]["token"] == "Hello"
        assert (
            response.metadata.logprobs["content"][0]["top_logprobs"][0]["token"]
            == "Hello"
        )
        assert response.metadata.finish_reason == "stop"

    def test_prefilling_is_not_compatible_with_generation_schema(
        self, mock_openai_client
    ):
        """Test prefilling is rejected with structured outputs."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        class DictOutput(msgspec.Struct):
            entities: List[Dict[str, str]]

        model = OpenAIChatCompletion(model_id="gpt-4")

        with pytest.raises(
            ValueError,
            match="`prefilling` is not compatible with `generation_schema`",
        ):
            model(
                messages=[{"role": "user", "content": "test"}],
                prefilling="{",
                generation_schema=DictOutput,
            )

    @pytest.mark.asyncio
    async def test_async_prefilling_is_not_compatible_with_generation_schema(
        self, mock_openai_client
    ):
        """Test async prefilling is rejected with structured outputs."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        class DictOutput(msgspec.Struct):
            entities: List[Dict[str, str]]

        model = OpenAIChatCompletion(model_id="gpt-4")

        with pytest.raises(
            ValueError,
            match="`prefilling` is not compatible with `generation_schema`",
        ):
            await model.acall(
                messages=[{"role": "user", "content": "test"}],
                prefilling="{",
                generation_schema=DictOutput,
            )

    def test_prepare_generate_kwargs_builds_dynamic_react_transport_schema(
        self, mock_openai_client
    ):
        """Test ToolFlowControl schemas can expose a dynamic transport schema."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(model_id="gpt-4")
        kwargs = {
            "generation_schema": ReAct,
            "tool_catalog": ToolCatalog.from_function_schemas(
                schemas=[
                    {
                        "type": "function",
                        "function": {
                            "name": "store_fields",
                            "description": "Store values",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "fields": {
                                        "type": "object",
                                        "properties": {
                                            "entries": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "key": {"type": "string"},
                                                        "value": {"type": "string"},
                                                    },
                                                    "required": ["key", "value"],
                                                    "additionalProperties": False,
                                                },
                                            }
                                        },
                                        "required": ["entries"],
                                        "additionalProperties": False,
                                    }
                                },
                                "required": ["fields"],
                                "additionalProperties": False,
                            },
                            "strict": True,
                        },
                    }
                ],
                annotations={"store_fields": {"fields": dict[str, str]}},
            ),
        }

        generation_schema, transport_generation_schema = model._prepare_generate_kwargs(
            kwargs
        )

        assert generation_schema is ReAct
        assert transport_generation_schema["decoder_schema"] is None
        action_schema = kwargs["response_format"]["json_schema"]["schema"][
            "properties"
        ]["actions"]["anyOf"][0]["items"]
        assert action_schema["properties"]["name"]["enum"] == ["store_fields"]
        assert "fields" in action_schema["properties"]
        assert "arguments" not in action_schema["properties"]

    def test_process_completion_model_output_normalizes_react_transport_shape(
        self, mock_openai_client
    ):
        """Test ToolFlowControl transport payloads are normalized to Action(arguments=...)."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(model_id="gpt-4")
        transport_generation_schema = model._prepare_generate_kwargs(
            {
                "generation_schema": ReAct,
                "tool_catalog": ToolCatalog.from_function_schemas(
                    schemas=[
                        {
                            "type": "function",
                            "function": {
                                "name": "store_fields",
                                "description": "Store values",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "fields": {
                                            "type": "object",
                                            "properties": {
                                                "entries": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "key": {"type": "string"},
                                                            "value": {"type": "string"},
                                                        },
                                                        "required": ["key", "value"],
                                                        "additionalProperties": False,
                                                    },
                                                }
                                            },
                                            "required": ["entries"],
                                            "additionalProperties": False,
                                        }
                                    },
                                    "required": ["fields"],
                                    "additionalProperties": False,
                                },
                                "strict": True,
                            },
                        }
                    ],
                    annotations={"store_fields": {"fields": dict[str, str]}},
                ),
            }
        )[1]

        model_output = SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content='{"thought":"Store the fields","actions":[{"name":"store_fields","fields":{"entries":[{"key":"city","value":"Austin"}]}}],"final_answer":null}',
                        tool_calls=None,
                        audio=None,
                        annotations=None,
                    ),
                )
            ],
        )

        response = model._process_completion_model_output(
            model_output,
            generation_schema=ReAct,
            transport_generation_schema=transport_generation_schema,
        )

        assert response.response_type == "structured"
        assert response.data == {
            "thought": "Store the fields",
            "actions": [
                {
                    "name": "store_fields",
                    "arguments": {"fields": {"city": "Austin"}},
                }
            ],
            "final_answer": None,
        }

    def test_prepare_generate_kwargs_uses_typed_final_answer_for_react_subclass(
        self, mock_openai_client
    ):
        """ToolFlowControl transport schema should follow the subclass final_answer type."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        class Outputs(msgspec.Struct):
            candidates: List[str]
            resolved: bool

        Output = type(
            "Output",
            (ReAct,),
            {
                "__annotations__": {
                    **ReAct.__annotations__,
                    "final_answer": Optional[Outputs],
                }
            },
        )

        model = OpenAIChatCompletion(model_id="gpt-4")
        kwargs = {
            "generation_schema": Output,
            "tool_catalog": ToolCatalog(tools=[]),
        }

        model._prepare_generate_kwargs(kwargs)

        final_answer_schema = kwargs["response_format"]["json_schema"]["schema"][
            "properties"
        ]["final_answer"]["anyOf"][0]
        assert final_answer_schema["type"] == "object"
        assert final_answer_schema["properties"]["candidates"]["type"] == "array"
        assert (
            final_answer_schema["properties"]["candidates"]["items"]["type"] == "string"
        )
        assert final_answer_schema["properties"]["resolved"]["type"] == "boolean"
        assert final_answer_schema["additionalProperties"] is False

    def test_process_completion_model_output_decodes_react_signature_final_answer(
        self, mock_openai_client
    ):
        """Decoded ReAct payload should respect the fused final_answer struct type."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        class Outputs(msgspec.Struct):
            candidates: List[str]
            resolved: bool

        Output = type(
            "Output",
            (ReAct,),
            {
                "__annotations__": {
                    **ReAct.__annotations__,
                    "final_answer": Optional[Outputs],
                }
            },
        )

        model = OpenAIChatCompletion(model_id="gpt-4")
        transport_generation_schema = model._prepare_generate_kwargs(
            {
                "generation_schema": Output,
                "tool_catalog": ToolCatalog(tools=[]),
            }
        )[1]

        model_output = SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content='{"thought":"I have enough information","actions":null,"final_answer":{"candidates":["Alice Johnson"],"resolved":true}}',
                        tool_calls=None,
                        audio=None,
                        annotations=None,
                    ),
                )
            ],
        )

        response = model._process_completion_model_output(
            model_output,
            generation_schema=Output,
            transport_generation_schema=transport_generation_schema,
        )

        assert response.response_type == "structured"
        assert response.data == {
            "thought": "I have enough information",
            "actions": None,
            "final_answer": {
                "candidates": ["Alice Johnson"],
                "resolved": True,
            },
        }


class TestOpenAITextToSpeech:
    """Test suite for OpenAITextToSpeech."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Setup environment variables for tests."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")

    def test_text_to_speech_initialization(self):
        """Test OpenAITextToSpeech initialization."""
        from msgflux.models.providers.openai import OpenAITextToSpeech

        model = OpenAITextToSpeech(model_id="tts-1")

        assert model.model_id == "tts-1"
        assert model.provider == "openai"
        assert model.model_type == "text_to_speech"
        assert model.stream_chunk_size == 1024

    def test_text_to_speech_with_voice_and_speed(self):
        """Test OpenAITextToSpeech with voice and speed parameters."""
        from msgflux.models.providers.openai import OpenAITextToSpeech

        model = OpenAITextToSpeech(
            model_id="tts-1",
            voice="nova",
            speed=1.5,
        )

        assert model.sampling_run_params["voice"] == "nova"
        assert model.sampling_run_params["speed"] == 1.5

    def test_text_to_speech_with_stream_chunk_size(self):
        """Test OpenAITextToSpeech with custom stream chunk size."""
        from msgflux.models.providers.openai import OpenAITextToSpeech

        model = OpenAITextToSpeech(
            model_id="tts-1",
            stream_chunk_size=512,
        )

        assert model.stream_chunk_size == 512

    @pytest.mark.parametrize("stream_chunk_size", [0, -1, 1.5, "1024"])
    def test_text_to_speech_rejects_invalid_stream_chunk_size(self, stream_chunk_size):
        """TTS stream chunk size must be a positive integer."""
        from msgflux.models.providers.openai import OpenAITextToSpeech

        with pytest.raises(ValueError, match="positive integer"):
            OpenAITextToSpeech(
                model_id="tts-1",
                stream_chunk_size=stream_chunk_size,
            )

    def test_text_to_speech_streams_transport_chunks(self):
        """TTS streaming should forward transport bytes to the response."""
        from msgflux.models.providers.openai import OpenAITextToSpeech
        from msgflux.models.response import ModelStreamResponse

        def fake_execute_model(**kwargs):
            yield b"audio"

        model = OpenAITextToSpeech(model_id="tts-1", stream_chunk_size=512)
        model._execute_model = fake_execute_model
        stream_response = ModelStreamResponse(mode="sync")

        model._stream_generate(
            input="hello",
            response_format="pcm",
            stream_response=stream_response,
        )

        assert stream_response.data == b"audio"

    @pytest.mark.asyncio
    async def test_text_to_speech_acall_stream_uses_async_detached(self):
        """TTS async streaming should use F.adetached, not the global Executor."""
        from msgflux.models.providers.openai import OpenAITextToSpeech

        model = OpenAITextToSpeech(model_id="tts-1")

        with (
            patch(
                "msgflux.models.providers.openai.F.adetached",
                new_callable=AsyncMock,
            ) as adetached,
            patch(
                "msgflux.models.providers.openai.F.await_for_event",
                new_callable=AsyncMock,
            ) as await_for_event,
            patch(
                "msgflux.nn.functional.Executor.get_instance",
                side_effect=AssertionError("Executor should not be used"),
            ),
        ):
            stream_response = await model.acall("hello", stream=True)

        assert stream_response.mode == "async"
        adetached.assert_awaited_once()
        await_for_event.assert_awaited_once_with(stream_response.first_chunk_event)


class TestOpenAITextToImage:
    """Test suite for OpenAITextToImage."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Setup environment variables for tests."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")

    def test_text_to_image_initialization(self):
        """Test OpenAITextToImage initialization."""

        from msgflux.models.providers.openai import OpenAITextToImage

        model = OpenAITextToImage(model_id="dall-e-3")

        assert model.model_id == "dall-e-3"
        assert model.provider == "openai"
        assert model.model_type == "text_to_image"

    def test_text_to_image_with_moderation(self):
        """Test OpenAITextToImage with moderation parameter."""

        from msgflux.models.providers.openai import OpenAITextToImage

        model = OpenAITextToImage(
            model_id="dall-e-3",
            moderation="low",
        )

        assert model.sampling_run_params.get("moderation") == "low"


class TestOpenAISpeechToText:
    """Test suite for OpenAISpeechToText."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Setup environment variables for tests."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")

    def test_speech_to_text_initialization(self):
        """Test OpenAISpeechToText initialization."""
        from msgflux.models.providers.openai import OpenAISpeechToText

        model = OpenAISpeechToText(model_id="whisper-1")

        assert model.model_id == "whisper-1"
        assert model.provider == "openai"
        assert model.model_type == "speech_to_text"

    def test_speech_to_text_with_temperature(self):
        """Test OpenAISpeechToText with temperature parameter."""
        from msgflux.models.providers.openai import OpenAISpeechToText

        model = OpenAISpeechToText(
            model_id="whisper-1",
            temperature=0.5,
        )

        assert model.sampling_run_params["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_speech_to_text_acall_stream_uses_async_detached(self):
        """STT async streaming should use F.adetached, not the global Executor."""
        from msgflux.models.providers.openai import OpenAISpeechToText

        model = OpenAISpeechToText(model_id="gpt-4o-transcribe")

        with (
            patch(
                "msgflux.models.providers.openai.aprepare_multipart_file",
                new_callable=AsyncMock,
                return_value=("audio.wav", b"audio", "audio/x-wav"),
            ),
            patch(
                "msgflux.models.providers.openai.F.adetached",
                new_callable=AsyncMock,
            ) as adetached,
            patch(
                "msgflux.models.providers.openai.F.await_for_event",
                new_callable=AsyncMock,
            ) as await_for_event,
            patch(
                "msgflux.nn.functional.Executor.get_instance",
                side_effect=AssertionError("Executor should not be used"),
            ),
        ):
            stream_response = await model.acall("audio.wav", stream=True)

        assert stream_response.mode == "async"
        adetached.assert_awaited_once()
        await_for_event.assert_awaited_once_with(stream_response.first_chunk_event)


class TestOpenAITextEmbedder:
    """Test suite for OpenAITextEmbedder."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Setup environment variables for tests."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")

    def test_text_embedder_initialization(self):
        """Test OpenAITextEmbedder initialization."""

        from msgflux.models.providers.openai import OpenAITextEmbedder

        model = OpenAITextEmbedder(model_id="text-embedding-3-large")

        assert model.model_id == "text-embedding-3-large"
        assert model.provider == "openai"
        assert model.model_type == "text_embedder"

    def test_text_embedder_with_dimensions(self):
        """Test OpenAITextEmbedder with dimensions parameter."""

        from msgflux.models.providers.openai import OpenAITextEmbedder

        model = OpenAITextEmbedder(
            model_id="text-embedding-3-large",
            dimensions=1536,
        )

        assert model.sampling_run_params["dimensions"] == 1536


class TestOpenAIModeration:
    """Test suite for OpenAIModeration."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Setup environment variables for tests."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")

    def test_moderation_initialization(self):
        """Test OpenAIModeration initialization."""

        from msgflux.models.providers.openai import OpenAIModeration

        model = OpenAIModeration(model_id="omni-moderation-latest")

        assert model.model_id == "omni-moderation-latest"
        assert model.provider == "openai"
        assert model.model_type == "moderation"


class TestOpenAIBaseURL:
    """Test suite for custom base_url parameter."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Setup environment variables for tests."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")

    def test_chat_completion_custom_base_url(self):
        """Test OpenAIChatCompletion with custom base_url."""

        from msgflux.models.providers.openai import OpenAIChatCompletion

        custom_url = "https://custom-api.example.com"
        model = OpenAIChatCompletion(
            model_id="gpt-4",
            base_url=custom_url,
        )

        assert model.sampling_params["base_url"] == custom_url

    def test_text_embedder_custom_base_url(self):
        """Test OpenAITextEmbedder with custom base_url."""

        from msgflux.models.providers.openai import OpenAITextEmbedder

        custom_url = "https://custom-api.example.com"
        model = OpenAITextEmbedder(
            model_id="text-embedding-ada-002",
            base_url=custom_url,
        )

        assert model.sampling_params["base_url"] == custom_url
