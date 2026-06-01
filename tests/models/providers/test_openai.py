"""Tests for msgflux.models.providers.openai module."""

import os
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import msgspec
import pytest

from msgflux.generation.reasoning.react import ReAct
from msgflux.tools.definitions import ToolDefinitions


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
        pytest.importorskip("openai", reason="openai not installed")

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
        """Mock OpenAI client."""
        with (
            patch("msgflux.models.providers.openai.OpenAI") as mock_client,
            patch("msgflux.models.providers.openai.AsyncOpenAI") as mock_async_client,
        ):
            yield mock_client, mock_async_client

    def test_chat_completion_initialization(self, mock_openai_client):
        """Test OpenAIChatCompletion initialization."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(model_id="gpt-4")

        assert model.model_id == "gpt-4"
        assert model.provider == "openai"
        assert model.model_type == "chat_completion"

    def test_chat_completion_with_parameters(self, mock_openai_client):
        """Test OpenAIChatCompletion with custom parameters."""
        pytest.importorskip("openai")

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
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAIChatCompletion

        extra_body = {"enable_citations": True, "enable_entities": True}
        model = OpenAIChatCompletion(model_id="gpt-4", extra_body=extra_body)

        assert model.sampling_run_params["extra_body"] == extra_body
        assert model.sampling_run_params["extra_body"] is not extra_body

    def test_chat_completion_with_extra_body_kwargs(self, mock_openai_client):
        """Test provider-specific fields passed directly as kwargs."""
        pytest.importorskip("openai")

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

    def test_chat_completion_merges_extra_body_and_kwargs(self, mock_openai_client):
        """Test init merges extra_body dict with direct provider kwargs."""
        pytest.importorskip("openai")

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
        pytest.importorskip("openai")

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
        pytest.importorskip("openai")

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
            extra_body={"enable_citations": True, "enable_entities": True},
        )
        model("Hello")

        call_kwargs = mock_client.return_value.chat.completions.create.call_args.kwargs
        assert call_kwargs["extra_body"] == {
            "enable_citations": True,
            "enable_entities": True,
        }

    def test_chat_completion_forwards_extra_body_kwargs(self, mock_openai_client):
        """Test direct provider kwargs are forwarded through extra_body."""
        pytest.importorskip("openai")

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
        pytest.importorskip("openai")

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
        pytest.importorskip("openai")

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
        pytest.importorskip("openai")

        # Remove API key
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        from msgflux.models.providers.openai import OpenAIChatCompletion

        with pytest.raises(ValueError, match="OpenAI key is not available"):
            OpenAIChatCompletion(model_id="gpt-4")

    def test_chat_completion_with_reasoning_params(self, mock_openai_client):
        """Test OpenAIChatCompletion with reasoning parameters."""
        pytest.importorskip("openai")

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

    def test_chat_completion_with_prompt_cache_retention(self, mock_openai_client):
        """Test OpenAIChatCompletion with OpenAI-only prompt cache retention."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(
            model_id="gpt-4",
            prompt_cache_retention="24h",
        )

        assert model.sampling_run_params["prompt_cache_retention"] == "24h"

    def test_chat_completion_with_logprobs_params(self, mock_openai_client):
        """Test OpenAIChatCompletion with logprobs parameters."""
        pytest.importorskip("openai")

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
        pytest.importorskip("openai")

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
        model = OpenAIChatCompletion(model_id="gpt-4")
        model("Hello", logprobs=True, top_logprobs=2)

        call_kwargs = mock_client.return_value.chat.completions.create.call_args.kwargs
        assert call_kwargs["logprobs"] is True
        assert call_kwargs["top_logprobs"] == 2

    @pytest.mark.asyncio
    async def test_acall_forwards_logprobs_params(self, mock_openai_client):
        """Test runtime logprobs parameters are forwarded on async calls."""
        pytest.importorskip("openai")

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
        model = OpenAIChatCompletion(model_id="gpt-4")
        await model.acall("Hello", logprobs=True, top_logprobs=2)

        call_kwargs = (
            mock_async_client.return_value.chat.completions.create.await_args.kwargs
        )
        assert call_kwargs["logprobs"] is True
        assert call_kwargs["top_logprobs"] == 2

    @pytest.mark.asyncio
    async def test_acall_forwards_extra_body_kwargs(self, mock_openai_client):
        """Test runtime provider kwargs are forwarded on async calls."""
        pytest.importorskip("openai")

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
        pytest.importorskip("openai")

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
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(model_id="gpt-4")

        with pytest.raises(ValueError, match="`top_logprobs` requires"):
            model("Hello", top_logprobs=2)

    @pytest.mark.asyncio
    async def test_acall_stream_strips_tool_definitions_before_async_client(
        self, mock_openai_client
    ):
        """Streaming async calls should not pass tool_definitions to the OpenAI SDK."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAIChatCompletion

        _, mock_async_client = mock_openai_client

        async def empty_stream():
            if False:
                yield None

        create = AsyncMock(return_value=empty_stream())
        mock_async_client.return_value.chat.completions.create = create

        model = OpenAIChatCompletion(model_id="gpt-4")

        await model.acall(
            messages=[{"role": "user", "content": "Check order 123"}],
            stream=True,
            tool_definitions=ToolDefinitions(
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
        assert "tool_definitions" not in call_kwargs
        assert call_kwargs["stream"] is True
        assert call_kwargs["tools"][0]["function"]["name"] == "get_order_status"
        assert call_kwargs["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_acall_stream_surfaces_backend_error_on_consume(
        self, mock_openai_client
    ):
        """Streaming async calls should expose provider failures to consumers."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAIChatCompletion

        _, mock_async_client = mock_openai_client
        mock_async_client.return_value.chat.completions.create = AsyncMock(
            side_effect=TypeError(
                "AsyncCompletions.create() got an unexpected keyword argument "
                "'tool_definitions'"
            )
        )

        model = OpenAIChatCompletion(model_id="gpt-4")
        stream_response = await model.acall(
            messages=[{"role": "user", "content": "Check order 123"}],
            stream=True,
            tool_definitions=ToolDefinitions(
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
            match="unexpected keyword argument 'tool_definitions'",
        ):
            async for _ in stream_response.consume():
                pass

    @pytest.mark.asyncio
    async def test_acall_stream_accumulates_response_data(self, mock_openai_client):
        """Streaming async text responses should leave the full payload in response.data."""
        pytest.importorskip("openai")

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
                choices=[],
                usage=SimpleNamespace(
                    to_dict=lambda: {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5,
                    }
                ),
            )

        mock_async_client.return_value.chat.completions.create = AsyncMock(
            return_value=text_stream()
        )

        model = OpenAIChatCompletion(model_id="gpt-4")
        response = await model.acall(
            messages=[{"role": "user", "content": "Say hello"}],
            stream=True,
        )

        chunks = []
        async for chunk in response.consume():
            chunks.append(chunk)

        assert chunks == ["Hello", " world"]
        assert response.response_type == "text_generation"
        assert response.data == "Hello world"
        assert response.metadata.usage["total_tokens"] == 5

    def test_prepare_generate_kwargs_lowers_dict_schema(self, mock_openai_client):
        """Test OpenAI transport schema lowering for dict-based structured outputs."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAIChatCompletion

        class DictOutput(msgspec.Struct):
            entities: List[Dict[str, str]]

        model = OpenAIChatCompletion(model_id="gpt-4")
        kwargs = {"typed_parser": None, "generation_schema": DictOutput}

        (
            typed_parser,
            generation_schema,
            transport_generation_schema,
        ) = model._prepare_generate_kwargs(kwargs)

        assert typed_parser is None
        assert generation_schema is DictOutput
        assert transport_generation_schema is not DictOutput
        assert (
            kwargs["response_format"]["json_schema"]["schema"]["properties"][
                "entities"
            ]["items"]["properties"]["entries"]["type"]
            == "array"
        )

    def test_build_generation_params_uses_tool_definitions(self, mock_openai_client):
        """Test native tool calling is derived from ToolDefinitions."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(model_id="gpt-4")
        params = model._build_generation_params(
            messages=[{"role": "user", "content": "What's the weather?"}],
            system_prompt=None,
            prefilling=None,
            tool_definitions=ToolDefinitions(
                schemas=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"location": {"type": "string"}},
                                "required": ["location"],
                            },
                        },
                    }
                ],
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
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(model_id="gpt-4")
        history = [{"role": "user", "content": "Hello"}]

        params = model._build_generation_params(
            messages=history,
            system_prompt="You are helpful.",
            prefilling=None,
            tool_definitions=None,
        )

        assert history == [{"role": "user", "content": "Hello"}]
        assert params["messages"][0]["role"] == "system"
        assert params["messages"][1] == history[0]

    def test_call_prefilling_does_not_mutate_messages(self, mock_openai_client):
        """Provider-side prefilling must not mutate caller history."""
        pytest.importorskip("openai")

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

        model = OpenAIChatCompletion(model_id="gpt-4")
        history = [{"role": "user", "content": "Hello"}]

        response = model(messages=history, prefilling="Start here")

        assert response.data == "done"
        assert history == [{"role": "user", "content": "Hello"}]

    @pytest.mark.asyncio
    async def test_acall_prefilling_does_not_mutate_messages(self, mock_openai_client):
        """Async provider-side prefilling must not mutate caller history."""
        pytest.importorskip("openai")

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

        model = OpenAIChatCompletion(model_id="gpt-4")
        history = [{"role": "user", "content": "Hello"}]

        response = await model.acall(messages=history, prefilling="Start here")

        assert response.data == "done"
        assert history == [{"role": "user", "content": "Hello"}]

    def test_process_completion_model_output_restores_dict_shape(
        self, mock_openai_client
    ):
        """Test transport-schema decoding is restored to the logical dict shape."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAIChatCompletion

        class DictOutput(msgspec.Struct):
            entities: List[Dict[str, str]]

        model = OpenAIChatCompletion(model_id="gpt-4")
        transport_generation_schema = model._prepare_generate_kwargs(
            {"typed_parser": None, "generation_schema": DictOutput}
        )[2]

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
        pytest.importorskip("openai")

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
        pytest.importorskip("openai")

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
        pytest.importorskip("openai")

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
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(model_id="gpt-4")
        kwargs = {
            "typed_parser": None,
            "generation_schema": ReAct,
            "tool_definitions": ToolDefinitions(
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

        (
            typed_parser,
            generation_schema,
            transport_generation_schema,
        ) = model._prepare_generate_kwargs(kwargs)

        assert typed_parser is None
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
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(model_id="gpt-4")
        transport_generation_schema = model._prepare_generate_kwargs(
            {
                "typed_parser": None,
                "generation_schema": ReAct,
                "tool_definitions": ToolDefinitions(
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
        )[2]

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
        pytest.importorskip("openai")

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
            "typed_parser": None,
            "generation_schema": Output,
            "tool_definitions": ToolDefinitions(schemas=[]),
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
        pytest.importorskip("openai")

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
                "typed_parser": None,
                "generation_schema": Output,
                "tool_definitions": ToolDefinitions(schemas=[]),
            }
        )[2]

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

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client."""
        with (
            patch("msgflux.models.providers.openai.OpenAI") as mock_client,
            patch("msgflux.models.providers.openai.AsyncOpenAI") as mock_async_client,
        ):
            yield mock_client, mock_async_client

    def test_text_to_speech_initialization(self, mock_openai_client):
        """Test OpenAITextToSpeech initialization."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAITextToSpeech

        model = OpenAITextToSpeech(model_id="tts-1")

        assert model.model_id == "tts-1"
        assert model.provider == "openai"
        assert model.model_type == "text_to_speech"
        assert model.stream_chunk_size == 1024

    def test_text_to_speech_with_voice_and_speed(self, mock_openai_client):
        """Test OpenAITextToSpeech with voice and speed parameters."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAITextToSpeech

        model = OpenAITextToSpeech(
            model_id="tts-1",
            voice="nova",
            speed=1.5,
        )

        assert model.sampling_run_params["voice"] == "nova"
        assert model.sampling_run_params["speed"] == 1.5

    def test_text_to_speech_with_stream_chunk_size(self, mock_openai_client):
        """Test OpenAITextToSpeech with custom stream chunk size."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAITextToSpeech

        model = OpenAITextToSpeech(
            model_id="tts-1",
            stream_chunk_size=512,
        )

        assert model.stream_chunk_size == 512

    @pytest.mark.parametrize("stream_chunk_size", [0, -1, 1.5, "1024"])
    def test_text_to_speech_rejects_invalid_stream_chunk_size(
        self, mock_openai_client, stream_chunk_size
    ):
        """TTS stream chunk size must be a positive integer."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAITextToSpeech

        with pytest.raises(ValueError, match="positive integer"):
            OpenAITextToSpeech(
                model_id="tts-1",
                stream_chunk_size=stream_chunk_size,
            )

    def test_text_to_speech_stream_uses_configured_chunk_size(self, mock_openai_client):
        """TTS streaming should pass the configured chunk size to iter_bytes."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAITextToSpeech
        from msgflux.models.response import ModelStreamResponse

        class FakeModelOutput:
            def __init__(self):
                self.chunk_size = None

            def iter_bytes(self, *, chunk_size):
                self.chunk_size = chunk_size
                yield b"audio"

        fake_output = FakeModelOutput()

        @contextmanager
        def fake_execute_model(**kwargs):
            yield fake_output

        model = OpenAITextToSpeech(
            model_id="tts-1",
            stream_chunk_size=512,
        )
        model._execute_model = fake_execute_model
        stream_response = ModelStreamResponse()

        model._stream_generate(
            input="hello",
            response_format="pcm",
            stream_response=stream_response,
        )

        assert fake_output.chunk_size == 512
        assert stream_response.data == b"audio"

    @pytest.mark.asyncio
    async def test_text_to_speech_acall_stream_uses_asyncio_spawn(
        self, mock_openai_client
    ):
        """TTS async streaming should stay on F.aspawn, not the global Executor."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAITextToSpeech

        model = OpenAITextToSpeech(model_id="tts-1")

        with (
            patch(
                "msgflux.models.providers.openai.F.aspawn",
                new_callable=AsyncMock,
            ) as aspawn,
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
        aspawn.assert_awaited_once()
        await_for_event.assert_awaited_once_with(stream_response.first_chunk_event)


class TestOpenAITextToImage:
    """Test suite for OpenAITextToImage."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Setup environment variables for tests."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client."""
        with (
            patch("msgflux.models.providers.openai.OpenAI") as mock_client,
            patch("msgflux.models.providers.openai.AsyncOpenAI") as mock_async_client,
        ):
            yield mock_client, mock_async_client

    def test_text_to_image_initialization(self, mock_openai_client):
        """Test OpenAITextToImage initialization."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAITextToImage

        model = OpenAITextToImage(model_id="dall-e-3")

        assert model.model_id == "dall-e-3"
        assert model.provider == "openai"
        assert model.model_type == "text_to_image"

    def test_text_to_image_with_moderation(self, mock_openai_client):
        """Test OpenAITextToImage with moderation parameter."""
        pytest.importorskip("openai")

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

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client."""
        with (
            patch("msgflux.models.providers.openai.OpenAI") as mock_client,
            patch("msgflux.models.providers.openai.AsyncOpenAI") as mock_async_client,
        ):
            yield mock_client, mock_async_client

    def test_speech_to_text_initialization(self, mock_openai_client):
        """Test OpenAISpeechToText initialization."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAISpeechToText

        model = OpenAISpeechToText(model_id="whisper-1")

        assert model.model_id == "whisper-1"
        assert model.provider == "openai"
        assert model.model_type == "speech_to_text"

    def test_speech_to_text_with_temperature(self, mock_openai_client):
        """Test OpenAISpeechToText with temperature parameter."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAISpeechToText

        model = OpenAISpeechToText(
            model_id="whisper-1",
            temperature=0.5,
        )

        assert model.sampling_run_params["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_speech_to_text_acall_stream_uses_asyncio_spawn(
        self, mock_openai_client
    ):
        """STT async streaming should stay on F.aspawn, not the global Executor."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAISpeechToText

        model = OpenAISpeechToText(model_id="gpt-4o-transcribe")

        with (
            patch(
                "msgflux.models.providers.openai.encode_data_to_bytes",
                return_value=b"audio",
            ),
            patch(
                "msgflux.models.providers.openai.F.aspawn",
                new_callable=AsyncMock,
            ) as aspawn,
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
        aspawn.assert_awaited_once()
        await_for_event.assert_awaited_once_with(stream_response.first_chunk_event)


class TestOpenAITextEmbedder:
    """Test suite for OpenAITextEmbedder."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Setup environment variables for tests."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client."""
        with (
            patch("msgflux.models.providers.openai.OpenAI") as mock_client,
            patch("msgflux.models.providers.openai.AsyncOpenAI") as mock_async_client,
        ):
            yield mock_client, mock_async_client

    def test_text_embedder_initialization(self, mock_openai_client):
        """Test OpenAITextEmbedder initialization."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAITextEmbedder

        model = OpenAITextEmbedder(model_id="text-embedding-3-large")

        assert model.model_id == "text-embedding-3-large"
        assert model.provider == "openai"
        assert model.model_type == "text_embedder"

    def test_text_embedder_with_dimensions(self, mock_openai_client):
        """Test OpenAITextEmbedder with dimensions parameter."""
        pytest.importorskip("openai")

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

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client."""
        with (
            patch("msgflux.models.providers.openai.OpenAI") as mock_client,
            patch("msgflux.models.providers.openai.AsyncOpenAI") as mock_async_client,
        ):
            yield mock_client, mock_async_client

    def test_moderation_initialization(self, mock_openai_client):
        """Test OpenAIModeration initialization."""
        pytest.importorskip("openai")

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

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client."""
        with (
            patch("msgflux.models.providers.openai.OpenAI") as mock_client,
            patch("msgflux.models.providers.openai.AsyncOpenAI") as mock_async_client,
        ):
            yield mock_client, mock_async_client

    def test_chat_completion_custom_base_url(self, mock_openai_client):
        """Test OpenAIChatCompletion with custom base_url."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAIChatCompletion

        custom_url = "https://custom-api.example.com"
        model = OpenAIChatCompletion(
            model_id="gpt-4",
            base_url=custom_url,
        )

        assert model.sampling_params["base_url"] == custom_url

    def test_text_embedder_custom_base_url(self, mock_openai_client):
        """Test OpenAITextEmbedder with custom base_url."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAITextEmbedder

        custom_url = "https://custom-api.example.com"
        model = OpenAITextEmbedder(
            model_id="text-embedding-ada-002",
            base_url=custom_url,
        )

        assert model.sampling_params["base_url"] == custom_url
