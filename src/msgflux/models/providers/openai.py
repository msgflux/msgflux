import base64
import tempfile
from contextlib import asynccontextmanager, contextmanager
from copy import deepcopy
from functools import partial
from os import getenv
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

import msgspec

try:
    import httpx
    import openai
    from openai import AsyncOpenAI, OpenAI
    from opentelemetry.instrumentation.openai import OpenAIInstrumentor

    if not getattr(openai, "_otel_instrumented", False):
        OpenAIInstrumentor().instrument()
        openai._otel_instrumented = True
except ImportError:
    httpx = None
    openai = None
    OpenAI = None
    AsyncOpenAI = None

import msgflux.nn.functional as F
from msgflux.core.dotdict import dotdict
from msgflux.dsl.typed_parsers import typed_parser_registry
from msgflux.exceptions import TypedParserNotFoundError
from msgflux.generation.control_flow import ToolFlowControl
from msgflux.models.base import BaseModel
from msgflux.models.cache import ResponseCache, generate_cache_key
from msgflux.models.profiles import get_model_profile
from msgflux.models.registry import register_model
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.models.types import (
    ChatCompletionModel,
    ImageTextToImageModel,
    ModerationModel,
    SpeechToTextModel,
    TextEmbedderModel,
    TextToImageModel,
    TextToSpeechModel,
)
from msgflux.tools.definitions import ToolDefinitions
from msgflux.utils.chat import ChatBlock, response_format_from_msgspec_struct
from msgflux.utils.console import cprint
from msgflux.utils.encode import encode_data_to_bytes
from msgflux.utils.msgspec import (
    lower_msgspec_struct_for_openai,
    restore_openai_structured_output,
    struct_to_dict,
)
from msgflux.utils.tenacity import apply_retry, default_model_retry
from msgflux.utils.validation import is_subclass_of


class _BaseOpenAI(BaseModel):
    provider: str = "openai"

    def _initialize(self):
        """Initialize the OpenAI client with empty API key."""
        if openai is None or OpenAI is None:
            raise ImportError(
                "`openai` client is not available. "
                "Install with `pip install msgflux[openai]`."
            )
        self.current_key_index = 0
        max_retries = getenv("OPENAI_MAX_RETRIES", openai.DEFAULT_MAX_RETRIES)
        timeout = getenv("OPENAI_TIMEOUT", None)
        self.client = OpenAI(
            **self.sampling_params,
            api_key=self._get_api_key(),
            timeout=timeout,
            max_retries=max_retries,
            http_client=httpx.Client(
                limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100)
            ),
        )
        self.aclient = AsyncOpenAI(
            **self.sampling_params,
            api_key=self._get_api_key(),
            timeout=timeout,
            max_retries=max_retries,
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100)
            ),
        )
        # Initialize response cache
        cache_size = getattr(self, "cache_size", 128)
        enable_cache = getattr(self, "enable_cache", None)
        self._response_cache = (
            ResponseCache(maxsize=cache_size) if enable_cache else None
        )

        # Apply retry
        retry_config = getattr(self, "retry", None)
        self.__call__ = apply_retry(
            self.__call__, retry_config, default=default_model_retry
        )
        self.acall = apply_retry(self.acall, retry_config, default=default_model_retry)

    def _get_base_url(self):
        return None

    def _get_api_key(self):
        """Load API keys from environment variable."""
        key = getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "The OpenAI key is not available. Please set `OPENAI_API_KEY`"
            )
        return key

    @property
    def profile(self):
        """Get model profile from registry.

        Returns:
            ModelProfile if found, None otherwise
        """
        return get_model_profile(self.model_id, provider_id=self.provider)


@register_model
class OpenAIChatCompletion(_BaseOpenAI, ChatCompletionModel):
    """OpenAI Chat Completion."""

    @staticmethod
    def _merge_extra_body(
        base_extra_body: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        extra_body_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        merged_extra_body = dict(base_extra_body or {})
        if extra_body is not None:
            merged_extra_body.update(extra_body)
        if extra_body_kwargs:
            duplicated_extra_body_keys = sorted(
                set(extra_body or {}).intersection(extra_body_kwargs)
            )
            if duplicated_extra_body_keys:
                duplicated = ", ".join(duplicated_extra_body_keys)
                raise ValueError(
                    "Duplicate provider extra-body keys passed in both "
                    "`extra_body` and direct kwargs: "
                    f"{duplicated}"
                )
            merged_extra_body.update(extra_body_kwargs)
        if not merged_extra_body and extra_body is None and not extra_body_kwargs:
            return None
        return merged_extra_body

    def __init__(  # noqa: C901
        self,
        model_id: str,
        *,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        prompt_cache_retention: Optional[Literal["in_memory", "24h"]] = None,
        enable_thinking: Optional[bool] = None,
        return_reasoning: Optional[bool] = True,
        reasoning_in_tool_call: Optional[bool] = True,
        validate_typed_parser_output: Optional[bool] = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        parallel_tool_calls: Optional[bool] = True,
        modalities: Optional[List[str]] = None,
        audio: Optional[Dict[str, str]] = None,
        verbosity: Optional[str] = None,
        web_search_options: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        verbose: Optional[bool] = False,
        base_url: Optional[str] = None,
        context_length: Optional[int] = None,
        reasoning_max_tokens: Optional[int] = None,
        enable_cache: Optional[bool] = False,
        cache_size: Optional[int] = 128,
        retry: Optional[Any] = None,
        warmup_max_tokens: Optional[int] = None,
        **extra_body_kwargs: Any,
    ):
        """Args:
        model_id:
            Model ID in provider.
        max_tokens:
            An upper bound for the number of tokens that can be
            generated for a completion, including visible output
            tokens and reasoning tokens.
        reasoning_effort:
            Constrains effort on reasoning for reasoning models.
            Currently supported values are low, medium, and high.
            Reducing reasoning effort can result in faster responses
            and fewer tokens used on reasoning in a response.
            Can be: "minimal", "low", "medium" or "high".
        prompt_cache_retention:
            OpenAI-only prompt cache retention policy.
            Allowed values are "in_memory" and "24h".
        enable_thinking:
            If True, enable the model reasoning.
        return_reasoning:
            If the model returns the `reasoning` field it will be added
            along with the response.
        reasoning_in_tool_call:
            If True, maintains the reasoning for using the tool call.
        validate_typed_parser_output:
            If True, use the generation_schema to validate typed parser output.
        temperature:
            What sampling temperature to use, between 0 and 2.
            Higher values like 0.8 will make the output more random,
            while lower values like 0.2 will make it more focused and
            deterministic.
        stop:
            Up to 4 sequences where the API will stop generating further
            tokens. The returned text will not contain the stop sequence.
        top_p:
            An alternative to sampling with temperature, called nucleus
            sampling, where the model considers the results of the tokens
            with top_p probability mass. So 0.1 means only the tokens
            comprising the top 10% probability mass are considered.
        logprobs:
            Token log probability output. When enabled, the response
            metadata includes the token-level logprob payload.
        top_logprobs:
            Number of alternative tokens to return per generated token.
            Use with `logprobs=True`.
        parallel_tool_calls:
            If True, enable parallel tool calls.
        modalities:
            Types of output you would like the model to generate.
            Can be: ["text"], ["audio"] or ["text", "audio"].
        audio:
            Audio configurations. Define voice and output format.
        verbosity:
            Constrains the verbosity of the model's response. Lower
            values will result in more concise responses, while higher
            values will result in more verbose responses. Currently
            supported values are low, medium, and high.
        web_search_options:
            This tool searches the web for relevant results to use in a response.
            OpenAI and OpenRouter only.
        extra_body:
            Provider-specific request body extensions forwarded to
            OpenAI-compatible clients.
        extra_body_kwargs:
            Additional provider-specific request body extensions passed
            directly as keyword arguments. These are merged into
            ``extra_body``.
        verbose:
            If True, Prints the model output to the console before it is transformed
            into typed structured output.
        base_url:
            URL to model provider.
        context_length:
            The maximum context length supported by the model.
        reasoning_max_tokens:
            OpenRouter-only maximum number of tokens for reasoning/thinking.
            This maps to ``extra_body={"reasoning": {"max_tokens": ...}}``
            and cannot be combined with ``reasoning_effort``.
        enable_cache:
            If True, enable response caching to avoid redundant API calls.
        cache_size:
            Maximum number of cached responses (default: 128).
        warmup_max_tokens:
            Maximum generated tokens used by prompt warmup requests. Defaults
            to 1 for OpenAI-compatible chat completions.
        """
        super().__init__()
        self.model_id = model_id
        self.context_length = context_length
        self.reasoning_max_tokens = reasoning_max_tokens
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.sampling_params = {"base_url": base_url or self._get_base_url()}
        sampling_run_params = {"max_tokens": max_tokens}
        if temperature:
            sampling_run_params["temperature"] = temperature
        if top_p:
            sampling_run_params["top_p"] = top_p
        if stop:
            sampling_run_params["stop"] = stop
        if self.provider == "openai" and logprobs is not None:
            sampling_run_params["logprobs"] = logprobs
        if self.provider == "openai" and top_logprobs is not None:
            sampling_run_params["top_logprobs"] = top_logprobs
        if verbosity:
            sampling_run_params["verbosity"] = verbosity
        if modalities:
            sampling_run_params["modalities"] = modalities
        if web_search_options:
            sampling_run_params["web_search_options"] = web_search_options
        merged_extra_body = self._merge_extra_body(
            extra_body=extra_body,
            extra_body_kwargs=extra_body_kwargs,
        )
        if merged_extra_body is not None:
            sampling_run_params["extra_body"] = merged_extra_body
        if audio:
            sampling_run_params["audio"] = audio
        if reasoning_effort:
            sampling_run_params["reasoning_effort"] = reasoning_effort
        if self.provider == "openrouter" and reasoning_max_tokens is not None:
            if reasoning_effort is not None:
                raise ValueError(
                    "`reasoning_max_tokens` cannot be used together with "
                    "`reasoning_effort` for OpenRouter."
                )
            sampling_run_params["reasoning_max_tokens"] = reasoning_max_tokens
        if self.provider == "openai" and prompt_cache_retention is not None:
            sampling_run_params["prompt_cache_retention"] = prompt_cache_retention
        self.sampling_run_params = sampling_run_params
        self.enable_thinking = enable_thinking
        self.parallel_tool_calls = parallel_tool_calls
        self.reasoning_in_tool_call = reasoning_in_tool_call
        self.validate_typed_parser_output = validate_typed_parser_output
        self.return_reasoning = return_reasoning
        self.verbose = verbose
        self.retry = retry
        self.warmup_max_tokens = warmup_max_tokens or 1
        self._initialize()
        self._get_api_key()

    def _adapt_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params.pop("provider_tools", None)
        if self.provider == "openai":
            max_tokens = params.pop("max_tokens", None)
            if max_tokens is not None:
                params["max_completion_tokens"] = max_tokens
        return params

    def _build_usage_metadata(self, model_output) -> dotdict:
        metadata = dotdict()
        usage = self._serialize_openai_value(getattr(model_output, "usage", None))
        if usage is not None:
            metadata.usage = usage
        return metadata

    @staticmethod
    def _serialize_openai_value(value):
        if value is None:
            return None
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if isinstance(value, Mapping):
            return dict(value)
        return value

    def _set_stop_metadata(
        self,
        metadata: dotdict,
        *,
        finish_reason: Optional[str] = None,
        stop_reason: Optional[str] = None,
    ) -> None:
        if finish_reason is None and stop_reason is not None:
            finish_reason = stop_reason
        if stop_reason is None and finish_reason is not None:
            stop_reason = finish_reason
        if finish_reason is not None:
            metadata.finish_reason = finish_reason
        if stop_reason is not None:
            metadata.stop_reason = stop_reason

    @staticmethod
    def _extract_reasoning(message) -> Optional[str]:
        return (
            getattr(message, "reasoning_content", None)
            or getattr(message, "reasoning", None)
            or getattr(message, "thinking", None)
        )

    @staticmethod
    def _extract_finish_reason(choice) -> Optional[str]:
        return getattr(choice, "finish_reason", None)

    @classmethod
    def _extract_annotations(cls, message) -> Optional[list]:
        annotations = getattr(message, "annotations", None)
        if annotations:
            return [cls._serialize_openai_value(item) for item in annotations]
        return None

    @staticmethod
    def _extract_logprobs(choice):
        return OpenAIChatCompletion._serialize_openai_value(
            getattr(choice, "logprobs", None)
        )

    def _build_completion_metadata(self, model_output, choice, message=None) -> dotdict:
        metadata = self._build_usage_metadata(model_output)
        self._set_stop_metadata(
            metadata, finish_reason=self._extract_finish_reason(choice)
        )
        if message is None:
            message = getattr(choice, "message", None)
        annotations = self._extract_annotations(message)
        if annotations:
            metadata.annotations = annotations
        logprobs = self._extract_logprobs(choice)
        if logprobs is not None:
            metadata.logprobs = logprobs
        return metadata

    @staticmethod
    def _merge_logprobs_metadata(metadata: dotdict, logprobs) -> None:
        if logprobs is None:
            return
        existing = metadata.get("logprobs")
        if existing is None:
            metadata.logprobs = logprobs
            return
        if isinstance(existing, Mapping) and isinstance(logprobs, Mapping):
            existing_content = existing.get("content")
            new_content = logprobs.get("content")
            if isinstance(existing_content, list) and isinstance(new_content, list):
                existing_content.extend(new_content)
                return
        metadata.logprobs = logprobs

    @staticmethod
    def _process_stream_tool_calls(delta, stream_response, aggregator):
        tool_call = delta.tool_calls[0]
        if stream_response.response_type is None:
            stream_response.set_response_type("tool_call")
        aggregator.process(
            tool_call.index,
            tool_call.id,
            tool_call.function.name,
            tool_call.function.arguments,
        )

    @staticmethod
    def _stream_add_chunk(stream_response, chunk, response_type):
        if stream_response.response_type is None:
            stream_response.set_response_type(response_type)
        stream_response.add(chunk)

    @staticmethod
    def _stream_add_reasoning_chunk(stream_response, chunk):
        stream_response.add_reasoning(chunk)

    def _execute_model(self, **kwargs):
        prefilling = kwargs.get("prefilling")
        params = {**self.sampling_run_params, **kwargs}
        params.pop("prefilling", None)
        if prefilling:
            params["messages"] = [
                *params["messages"],
                {"role": "assistant", "content": prefilling},
            ]
        adapted_params = self._adapt_params(params)
        model_output = self.client.chat.completions.create(**adapted_params)

        return model_output

    async def _aexecute_model(self, **kwargs):
        prefilling = kwargs.get("prefilling")
        params = {**self.sampling_run_params, **kwargs}
        params.pop("prefilling", None)
        if prefilling:
            params["messages"] = [
                *params["messages"],
                {"role": "assistant", "content": prefilling},
            ]
        adapted_params = self._adapt_params(params)
        model_output = await self.aclient.chat.completions.create(**adapted_params)

        return model_output

    def _build_warmup_params(
        self,
        *,
        system_prompt: Optional[str],
        tool_definitions: Optional[ToolDefinitions],
    ) -> Dict[str, Any]:
        generation_params = self._build_generation_params(
            [],
            system_prompt,
            None,
            tool_definitions,
        )
        generation_params["max_tokens"] = self.warmup_max_tokens
        generation_params.pop("prefilling", None)
        # Warmup intentionally bypasses typed parsers, checkpointers, chat history
        # and response caching. The request only contains the stable system prompt
        # plus tool schemas so provider-side prompt caches can prefill that prefix.
        return self._adapt_params({**self.sampling_run_params, **generation_params})

    def warmup_system_prompt(
        self,
        *,
        system_prompt: Optional[str],
        tool_definitions: Optional[ToolDefinitions] = None,
    ):
        params = self._build_warmup_params(
            system_prompt=system_prompt,
            tool_definitions=tool_definitions,
        )
        return self.client.chat.completions.create(**params)

    async def awarmup_system_prompt(
        self,
        *,
        system_prompt: Optional[str],
        tool_definitions: Optional[ToolDefinitions] = None,
    ):
        params = self._build_warmup_params(
            system_prompt=system_prompt,
            tool_definitions=tool_definitions,
        )
        return await self.aclient.chat.completions.create(**params)

    def _process_completion_model_output(  # noqa: C901
        self,
        model_output,
        typed_parser=None,
        generation_schema=None,
        transport_generation_schema=None,
    ):
        """Build a ModelResponse from the raw OpenAI completion output.

        `generation_schema` is the canonical msgflux schema exposed to callers.
        `transport_generation_schema` is an OpenAI-specific wire schema used when
        the canonical schema must be lowered to satisfy Structured Outputs
        constraints, for example lowering ``Dict[K, V]`` to an ``entries`` list.
        """
        response = ModelResponse()
        choice = model_output.choices[0]
        metadata = self._build_completion_metadata(model_output, choice)

        reasoning = self._extract_reasoning(choice.message)

        reasoning_tool_call = reasoning if self.reasoning_in_tool_call else None

        reasoning_content = None
        if self.return_reasoning is True and reasoning is not None:
            reasoning_content = reasoning

        annotations = self._extract_annotations(choice.message)
        if annotations:
            metadata.annotations = annotations

        if choice.message.tool_calls:
            aggregator = ToolCallAggregator(reasoning_tool_call)
            response.set_response_type("tool_call")
            for call_index, tool_call in enumerate(choice.message.tool_calls):
                tool_id = tool_call.id
                name = tool_call.function.name
                arguments = tool_call.function.arguments
                aggregator.process(call_index, tool_id, name, arguments)
            response_content = aggregator
        elif choice.message.content:
            if (typed_parser or generation_schema) and self.verbose:
                repr_str = f"[{self.model_id}][raw_response] {choice.message.content}"
                cprint(repr_str, lc="r", ls="b")
            if typed_parser is not None:
                response.set_response_type("structured")
                parser = typed_parser_registry[typed_parser]
                response_content = dotdict(parser.decode(choice.message.content))
                if generation_schema and self.validate_typed_parser_output:
                    decoder = self._get_decoder(generation_schema)
                    decoder.decode(self._encoder.encode(response_content))
            elif generation_schema is not None:
                response.set_response_type("structured")
                # The raw payload follows the OpenAI transport schema, which may be
                # a lowered or dynamically generated version of the logical msgflux
                # generation schema.
                transport_info = transport_generation_schema or {}
                decoder_schema = transport_info.get("decoder_schema", generation_schema)
                normalize = transport_info.get("normalize")

                if decoder_schema is None:
                    response_content = msgspec.json.decode(choice.message.content)
                else:
                    decoder = self._get_decoder(decoder_schema)
                    struct = decoder.decode(choice.message.content)
                    response_content = struct_to_dict(struct)

                if normalize is not None:
                    response_content = normalize(response_content)

                decoder = self._get_decoder(generation_schema)
                struct = decoder.decode(self._encoder.encode(response_content))
                response_content = dotdict(struct_to_dict(struct))
            else:
                response.set_response_type("text_generation")
                response_content = choice.message.content
        elif choice.message.audio:
            response_content = dotdict(
                {
                    "id": choice.message.audio.id,
                    "audio": base64.b64decode(choice.message.audio.data),
                }
            )
            if choice.message.audio.transcript:
                response.set_response_type("audio_text_generation")
                response_content.text = choice.message.audio.transcript
            else:
                response.set_response_type("audio_generation")
        else:
            response.set_response_type("text_generation")
            response_content = ""

        response.reasoning = reasoning_content
        response.add(response_content)
        response.set_metadata(metadata)
        return response

    def _process_model_output(
        self,
        model_output,
        typed_parser=None,
        generation_schema=None,
        transport_generation_schema=None,
    ):
        return self._process_completion_model_output(
            model_output,
            typed_parser,
            generation_schema,
            transport_generation_schema,
        )

    def _check_cache(self, **kwargs):
        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(**kwargs)
            hit, cached_response = self._response_cache.get(cache_key)
            if hit:
                return cached_response
        return None

    def _store_cache(self, response, **kwargs):
        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(**kwargs)
            self._response_cache.set(cache_key, response)

    def _prepare_generate_kwargs(self, kwargs):
        """Prepare generation kwargs and derive the OpenAI transport schema.

        `generation_schema` remains the canonical schema for msgflux.
        `transport_generation_schema` is the schema sent to OpenAI in
        `response_format`; it may be the same type or a lowered variant that only
        exists to satisfy OpenAI Structured Outputs restrictions.
        """
        typed_parser = kwargs.pop("typed_parser", None)
        generation_schema = kwargs.pop("generation_schema", None)
        tool_definitions = kwargs.pop("tool_definitions", None)
        transport_generation_schema = None

        if generation_schema is not None and typed_parser is None:
            if issubclass(generation_schema, ToolFlowControl):
                response_format = generation_schema.build_provider_response_format(
                    tool_definitions
                )
                if response_format is not None:
                    transport_generation_schema = {
                        "decoder_schema": None,
                        "normalize": lambda payload: (
                            generation_schema.normalize_provider_response(
                                payload,
                                tool_definitions=tool_definitions,
                            )
                        ),
                    }
                    kwargs["response_format"] = response_format

            if transport_generation_schema is None:
                # Lower only for the OpenAI transport layer; the logical schema
                # stays unchanged so decoded outputs can be restored to the
                # original shape.
                decoder_schema = lower_msgspec_struct_for_openai(generation_schema)
                normalize = None
                if decoder_schema is not generation_schema:
                    normalize = partial(
                        restore_openai_structured_output,
                        logical_type=generation_schema,
                    )
                transport_generation_schema = {
                    "decoder_schema": decoder_schema,
                    "normalize": normalize,
                }
                kwargs["response_format"] = response_format_from_msgspec_struct(
                    decoder_schema
                )

        return typed_parser, generation_schema, transport_generation_schema

    def _prepare_stream_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Strip internal generation-only args before raw streaming requests."""
        kwargs.pop("typed_parser", None)
        kwargs.pop("generation_schema", None)
        kwargs.pop("tool_definitions", None)
        return kwargs

    def _generate(self, **kwargs: Mapping[str, Any]) -> ModelResponse:
        cached = self._check_cache(**kwargs)
        if cached:
            return cached

        (
            typed_parser,
            generation_schema,
            transport_generation_schema,
        ) = self._prepare_generate_kwargs(kwargs)

        model_output = self._execute_model(**kwargs)
        response = self._process_model_output(
            model_output,
            typed_parser,
            generation_schema,
            transport_generation_schema,
        )

        self._store_cache(
            response,
            **kwargs,
            typed_parser=typed_parser,
            generation_schema=generation_schema,
        )
        return response

    async def _agenerate(self, **kwargs: Mapping[str, Any]) -> ModelResponse:
        cached = self._check_cache(**kwargs)
        if cached:
            return cached

        (
            typed_parser,
            generation_schema,
            transport_generation_schema,
        ) = self._prepare_generate_kwargs(kwargs)

        model_output = await self._aexecute_model(**kwargs)
        response = self._process_model_output(
            model_output,
            typed_parser,
            generation_schema,
            transport_generation_schema,
        )

        self._store_cache(
            response,
            **kwargs,
            typed_parser=typed_parser,
            generation_schema=generation_schema,
        )
        return response

    def _stream_generate(  # noqa: C901
        self, **kwargs: Mapping[str, Any]
    ) -> ModelStreamResponse:
        stream_response = kwargs.pop("stream_response")
        metadata = dotdict()
        reasoning_tool_call = ""
        reasoning_accumulated = ""

        try:
            aggregator = ToolCallAggregator()
            model_output = self._execute_model(**kwargs)
            finish_reason = None

            for chunk in model_output:
                if chunk.choices:
                    choice = chunk.choices[0]
                    delta = choice.delta
                    fr = self._extract_finish_reason(choice)
                    if fr is not None:
                        finish_reason = fr

                    chunk_metadata = self._build_completion_metadata(
                        chunk,
                        choice,
                        delta,
                    )
                    annotations = getattr(chunk_metadata, "annotations", None)
                    if annotations:
                        metadata.annotations = annotations
                    self._merge_logprobs_metadata(
                        metadata,
                        getattr(chunk_metadata, "logprobs", None),
                    )

                    reasoning_chunk = self._extract_reasoning(delta)

                    if reasoning_chunk:
                        if self.reasoning_in_tool_call:
                            reasoning_tool_call += reasoning_chunk
                        if self.return_reasoning:
                            reasoning_accumulated += reasoning_chunk
                            self._stream_add_reasoning_chunk(
                                stream_response,
                                reasoning_chunk,
                            )
                        continue

                    if getattr(delta, "content", None):
                        self._stream_add_chunk(
                            stream_response,
                            delta.content,
                            "text_generation",
                        )
                        continue

                    if getattr(delta, "tool_calls", None):
                        self._process_stream_tool_calls(
                            delta,
                            stream_response,
                            aggregator,
                        )
                        continue

                elif chunk.usage:
                    usage = self._serialize_openai_value(chunk.usage)
                    if usage is not None:
                        metadata.update(usage)
                        metadata.usage = usage

            if aggregator.tool_calls:
                if reasoning_tool_call:
                    aggregator.reasoning = reasoning_tool_call
                stream_response.data = aggregator
                stream_response.first_chunk_event.set()
            stream_response.reasoning = reasoning_accumulated or None
            self._set_stop_metadata(metadata, finish_reason=finish_reason)
        except Exception as e:
            stream_response.set_error(e)
        finally:
            if not stream_response.first_chunk_event.is_set():
                stream_response.first_chunk_event.set()
            if not stream_response._response_type_event.is_set():
                stream_response._response_type_event.set()
            stream_response.set_metadata(metadata)
            stream_response.add_reasoning(None)
            stream_response.add(None)

    async def _astream_generate(  # noqa: C901
        self, **kwargs: Mapping[str, Any]
    ) -> ModelStreamResponse:
        stream_response = kwargs.pop("stream_response")
        metadata = dotdict()
        reasoning_tool_call = ""
        reasoning_accumulated = ""

        try:
            aggregator = ToolCallAggregator()
            model_output = await self._aexecute_model(**kwargs)
            finish_reason = None

            async for chunk in model_output:
                if chunk.choices:
                    choice = chunk.choices[0]
                    delta = choice.delta
                    fr = self._extract_finish_reason(choice)
                    if fr is not None:
                        finish_reason = fr

                    chunk_metadata = self._build_completion_metadata(
                        chunk,
                        choice,
                        delta,
                    )
                    annotations = getattr(chunk_metadata, "annotations", None)
                    if annotations:
                        metadata.annotations = annotations
                    self._merge_logprobs_metadata(
                        metadata,
                        getattr(chunk_metadata, "logprobs", None),
                    )

                    reasoning_chunk = self._extract_reasoning(delta)

                    if reasoning_chunk:
                        if self.reasoning_in_tool_call:
                            reasoning_tool_call += reasoning_chunk
                        if self.return_reasoning:
                            reasoning_accumulated += reasoning_chunk
                            self._stream_add_reasoning_chunk(
                                stream_response,
                                reasoning_chunk,
                            )
                        continue

                    if getattr(delta, "content", None):
                        self._stream_add_chunk(
                            stream_response,
                            delta.content,
                            "text_generation",
                        )
                        continue

                    if getattr(delta, "tool_calls", None):
                        self._process_stream_tool_calls(
                            delta,
                            stream_response,
                            aggregator,
                        )
                        continue

                elif chunk.usage:
                    usage = self._serialize_openai_value(chunk.usage)
                    if usage is not None:
                        metadata.update(usage)
                        metadata.usage = usage

            if aggregator.tool_calls:
                if reasoning_tool_call:
                    aggregator.reasoning = reasoning_tool_call
                stream_response.data = aggregator
                stream_response.first_chunk_event.set()
            stream_response.reasoning = reasoning_accumulated or None
            self._set_stop_metadata(metadata, finish_reason=finish_reason)
        except Exception as e:
            stream_response.set_error(e)
        finally:
            if not stream_response.first_chunk_event.is_set():
                stream_response.first_chunk_event.set()
            if not stream_response._response_type_event.is_set():
                stream_response._response_type_event.set()
            stream_response.set_metadata(metadata)
            stream_response.add_reasoning(None)
            stream_response.add(None)

    def _build_generation_params(
        self,
        messages: Union[str, List[Dict[str, Any]]],
        system_prompt: Optional[str],
        prefilling: Optional[str],
        tool_definitions: Optional[ToolDefinitions],
        *,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        extra_body_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if isinstance(messages, str):
            messages = [ChatBlock.user(messages)]
        else:
            messages = deepcopy(messages)
        if isinstance(system_prompt, str):
            messages.insert(0, ChatBlock.system(system_prompt))

        tool_choice = tool_definitions.choice if tool_definitions else None
        if isinstance(tool_choice, str):
            if tool_choice not in ["auto", "required", "none"]:
                tool_choice = {
                    "type": "function",
                    "function": {"name": tool_choice},
                }

        generation_params = {
            "messages": messages,
            "prefilling": prefilling,
            "model": self.model_id,
        }

        if logprobs is not None:
            generation_params["logprobs"] = logprobs
        if top_logprobs is not None:
            generation_params["top_logprobs"] = top_logprobs
        merged_extra_body = self._merge_extra_body(
            self.sampling_run_params.get("extra_body"),
            extra_body,
            extra_body_kwargs,
        )
        if merged_extra_body is not None:
            generation_params["extra_body"] = merged_extra_body

        if tool_definitions and tool_definitions.schemas:
            generation_params["tools"] = tool_definitions.schemas
            generation_params["tool_choice"] = tool_choice
            generation_params["parallel_tool_calls"] = self.parallel_tool_calls

        return generation_params

    @staticmethod
    def _validate_chat_completion_options(
        *,
        prefilling: Optional[str],
        logprobs: Optional[bool],
        top_logprobs: Optional[int],
        generation_schema: Optional[msgspec.Struct],
        typed_parser: Optional[str],
        stream: Optional[bool],
    ) -> None:
        if prefilling is not None and generation_schema is not None:
            raise ValueError(
                "`prefilling` is not compatible with `generation_schema` in "
                "OpenAI chat completions."
            )
        if top_logprobs is not None and logprobs is not True:
            raise ValueError("`top_logprobs` requires `logprobs=True`")
        if stream is True and typed_parser is not None:
            raise ValueError("`typed_parser` is not `stream=True` compatible")

    def __call__(
        self,
        messages: Union[str, List[Dict[str, Any]]],
        *,
        system_prompt: Optional[str] = None,
        prefilling: Optional[str] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        stream: Optional[bool] = False,
        generation_schema: Optional[msgspec.Struct] = None,
        tool_definitions: Optional[ToolDefinitions] = None,
        typed_parser: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        **extra_body_kwargs: Any,
    ) -> Union[ModelResponse, ModelStreamResponse]:
        """Args:
            messages:
                Conversation history. Can be simple string or list of messages.
            system_prompt:
                A set of instructions that defines the overarching behavior
                and role of the model across all interactions.
            prefilling:
                Forces an initial message from the model. From that message
                it will continue its response from there.
            logprobs:
                Token log probability output for this request.
            top_logprobs:
                Number of alternative tokens to return per generated token
                for this request.
            stream:
                Whether generation should be in streaming mode.
            generation_schema:
                Schema that defines how the output should be structured.
            tool_definitions:
                Optional container with tool schemas, annotations, and
                tool-choice metadata. This is the single tool-calling entrypoint
                for the provider.
            typed_parser:
                Converts the model raw output into a typed-dict. Supported parser:
                `typed_xml`.
            extra_body:
                Provider-specific request body extensions for this request.
            extra_body_kwargs:
                Additional provider-specific request body extensions for this
                request, merged into ``extra_body``.

        Raises:
            ValueError:
                Raised if `generation_schema` and `stream=True`.
            ValueError:
                Raised if `typed_xml=True` and `stream=True`.
        """
        self._validate_chat_completion_options(
            prefilling=prefilling,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            generation_schema=generation_schema,
            typed_parser=typed_parser,
            stream=stream,
        )
        is_flow_control = is_subclass_of(generation_schema, ToolFlowControl)
        generation_params = self._build_generation_params(
            messages,
            system_prompt,
            prefilling,
            None if is_flow_control else tool_definitions,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            extra_body=extra_body,
            extra_body_kwargs=extra_body_kwargs,
        )
        if tool_definitions is not None:
            generation_params["tool_definitions"] = tool_definitions

        if stream is True:
            self._prepare_stream_kwargs(generation_params)
            stream_response = ModelStreamResponse(mode="sync")
            F.spawn(
                self._stream_generate,
                **generation_params,
                stream=stream,
                stream_response=stream_response,
                stream_options={"include_usage": True},
            )
            F.wait_for_event(stream_response.first_chunk_event)
            return stream_response
        else:
            if typed_parser and typed_parser not in typed_parser_registry:
                available = ", ".join(typed_parser_registry.keys())
                raise TypedParserNotFoundError(
                    f"Typed parser `{typed_parser}` not found. "
                    f"Available parsers: {available}"
                )
            response = self._generate(
                **generation_params,
                typed_parser=typed_parser,
                generation_schema=generation_schema,
            )
            return response

    async def acall(
        self,
        messages: Union[str, List[Dict[str, Any]]],
        *,
        system_prompt: Optional[str] = None,
        prefilling: Optional[str] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        stream: Optional[bool] = False,
        generation_schema: Optional[msgspec.Struct] = None,
        tool_definitions: Optional[ToolDefinitions] = None,
        typed_parser: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        **extra_body_kwargs: Any,
    ) -> Union[ModelResponse, ModelStreamResponse]:
        """Async version of __call__. Args:
            messages:
                Conversation history. Can be simple string or list of messages.
            system_prompt:
                A set of instructions that defines the overarching behavior
                and role of the model across all interactions.
            prefilling:
                Forces an initial message from the model. From that message
                it will continue its response from there.
            logprobs:
                Token log probability output for this request.
            top_logprobs:
                Number of alternative tokens to return per generated token
                for this request.
            stream:
                Whether generation should be in streaming mode.
            generation_schema:
                Schema that defines how the output should be structured.
            tool_definitions:
                Optional container with tool schemas, annotations, and
                tool-choice metadata. This is the single tool-calling entrypoint
                for the provider.
            typed_parser:
                Converts the model raw output into a typed-dict. Supported parser:
                `typed_xml`.
            extra_body:
                Provider-specific request body extensions for this request.
            extra_body_kwargs:
                Additional provider-specific request body extensions for this
                request, merged into ``extra_body``.

        Raises:
            ValueError:
                Raised if `generation_schema` and `stream=True`.
            ValueError:
                Raised if `typed_xml=True` and `stream=True`.
        """
        self._validate_chat_completion_options(
            prefilling=prefilling,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            generation_schema=generation_schema,
            typed_parser=typed_parser,
            stream=stream,
        )
        is_flow_control = is_subclass_of(generation_schema, ToolFlowControl)
        generation_params = self._build_generation_params(
            messages,
            system_prompt,
            prefilling,
            None if is_flow_control else tool_definitions,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            extra_body=extra_body,
            extra_body_kwargs=extra_body_kwargs,
        )
        if tool_definitions is not None:
            generation_params["tool_definitions"] = tool_definitions

        if stream is True:
            self._prepare_stream_kwargs(generation_params)
            stream_response = ModelStreamResponse(mode="async")
            await F.aspawn(
                self._astream_generate,
                **generation_params,
                stream=stream,
                stream_response=stream_response,
                stream_options={"include_usage": True},
            )
            await F.await_for_event(stream_response.first_chunk_event)
            return stream_response
        else:
            if typed_parser and typed_parser not in typed_parser_registry:
                available = ", ".join(typed_parser_registry.keys())
                raise TypedParserNotFoundError(
                    f"Typed parser `{typed_parser}` not found. "
                    f"Available parsers: {available}"
                )
            response = await self._agenerate(
                **generation_params,
                typed_parser=typed_parser,
                generation_schema=generation_schema,
            )
            return response


@register_model
class OpenAITextToSpeech(_BaseOpenAI, TextToSpeechModel):
    """OpenAI Text to Speech."""

    def __init__(
        self,
        model_id: str,
        voice: Optional[str] = "alloy",
        speed: Optional[float] = 1.0,
        stream_chunk_size: int = 1024,
        base_url: Optional[str] = None,
        retry: Optional[Any] = None,
    ):
        """Args:
        model_id:
            Model ID in provider.
        voice:
            The voice to use when generating the audio.
        speed:
            the speed of the generated audio. Select a value
            from 0.25 to 4.0. 1.0 is the default.
        stream_chunk_size:
            Number of bytes yielded per streaming audio chunk.
        base_url:
            URL to model provider.
        retry:
            Retry config. A tenacity decorator, False to disable, or None for default.
        """
        super().__init__()
        if not isinstance(stream_chunk_size, int) or stream_chunk_size <= 0:
            raise ValueError("`stream_chunk_size` must be a positive integer")
        self.model_id = model_id
        self.stream_chunk_size = stream_chunk_size
        self.sampling_params = {"base_url": base_url or self._get_base_url()}
        self.sampling_run_params = {
            "voice": voice,
            "speed": speed,
        }
        self.retry = retry
        self._initialize()
        self._get_api_key()

    @contextmanager
    def _execute_model(self, **kwargs):
        with self.client.audio.speech.with_streaming_response.create(
            model=self.model_id, **kwargs, **self.sampling_run_params
        ) as model_output:
            yield model_output

    @asynccontextmanager
    async def _aexecute_model(self, **kwargs):
        async with self.aclient.audio.speech.with_streaming_response.create(
            model=self.model_id, **kwargs, **self.sampling_run_params
        ) as model_output:
            yield model_output

    def _generate(self, **kwargs):
        response = ModelResponse()

        with self._execute_model(**kwargs) as model_output:
            with tempfile.NamedTemporaryFile(
                suffix=f".{kwargs.get('response_format')}", delete=False
            ) as temp_file:
                temp_file_path = temp_file.name
                model_output.stream_to_file(temp_file_path)

            response.set_response_type("audio_generation")
            response.add(temp_file_path)

        return response

    async def _agenerate(self, **kwargs):
        response = ModelResponse()

        async with self._aexecute_model(**kwargs) as model_output:
            with tempfile.NamedTemporaryFile(
                suffix=f".{kwargs.get('response_format')}", delete=False
            ) as temp_file:
                temp_file_path = temp_file.name
                await model_output.stream_to_file(temp_file_path)

            response.set_response_type("audio_generation")
            response.add(temp_file_path)

        return response

    def _stream_generate(self, **kwargs):
        stream_response = kwargs.pop("stream_response")
        stream_response.set_response_type("audio_generation")

        with self._execute_model(**kwargs) as model_output:
            for chunk in model_output.iter_bytes(chunk_size=self.stream_chunk_size):
                stream_response.add(chunk)
                if not stream_response.first_chunk_event.is_set():
                    stream_response.first_chunk_event.set()

        stream_response.add(None)

    async def _astream_generate(self, **kwargs):
        stream_response = kwargs.pop("stream_response")
        stream_response.set_response_type("audio_generation")

        async with self._aexecute_model(**kwargs) as model_output:
            async for chunk in model_output.iter_bytes(
                chunk_size=self.stream_chunk_size
            ):
                stream_response.add(chunk)
                if not stream_response.first_chunk_event.is_set():
                    stream_response.first_chunk_event.set()

        stream_response.add(None)

    def __call__(
        self,
        data: str,
        *,
        stream: Optional[bool] = False,
        prompt: Optional[str] = None,
        response_format: Optional[
            Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]
        ] = "opus",
    ) -> Union[ModelResponse, ModelStreamResponse]:
        """Args:
        data:
            The text to generate audio for.
        stream:
            Whether generation should be in streaming mode.
        prompt:
            Control the voice of your generated audio with additional instructions.
        response_format:
            The format to audio in.
        """
        params = dotdict({"input": data, "response_format": response_format})
        if prompt:
            params.instructions = prompt
        if stream:
            stream_response = ModelStreamResponse(mode="sync")
            params.stream_response = stream_response
            F.spawn(self._stream_generate, **params)
            F.wait_for_event(stream_response.first_chunk_event)
            return stream_response
        else:
            response = self._generate(**params)
            return response

    async def acall(
        self,
        data: str,
        *,
        stream: Optional[bool] = False,
        prompt: Optional[str] = None,
        response_format: Optional[
            Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]
        ] = "opus",
    ) -> Union[ModelResponse, ModelStreamResponse]:
        """Async version of __call__. Args:
        data:
            The text to generate audio for.
        stream:
            Whether generation should be in streaming mode.
        prompt:
            Control the voice of your generated audio with additional instructions.
        response_format:
            The format to audio in.
        """
        params = dotdict({"input": data, "response_format": response_format})
        if prompt:
            params.instructions = prompt
        if stream:
            stream_response = ModelStreamResponse(mode="async")
            params.stream_response = stream_response
            await F.aspawn(self._astream_generate, **params)
            await F.await_for_event(stream_response.first_chunk_event)
            return stream_response
        else:
            response = await self._agenerate(**params)
            return response


@register_model
class OpenAITextToImage(_BaseOpenAI, TextToImageModel):
    """OpenAI Image Generation."""

    def __init__(
        self,
        *,
        model_id: str,
        moderation: Optional[Literal["auto", "low"]] = None,
        base_url: Optional[str] = None,
        retry: Optional[Any] = None,
    ):
        """Args:
        model_id:
            Model ID in provider.
        moderation:
            Control the content-moderation level for images generated.
        base_url:
            URL to model provider.
        retry:
            Retry config. A tenacity decorator, False to disable, or None for default.
        """
        super().__init__()
        self.model_id = model_id
        self.sampling_params = {"base_url": base_url or self._get_base_url()}
        sampling_run_params = {}
        if moderation:
            sampling_run_params["moderation"] = moderation
        self.sampling_run_params = sampling_run_params
        self.retry = retry
        self._initialize()
        self._get_api_key()

    def _execute_model(self, **kwargs):
        model_output = self.client.images.generate(**kwargs, **self.sampling_run_params)
        return model_output

    async def _aexecute_model(self, **kwargs):
        model_output = await self.aclient.images.generate(
            **kwargs, **self.sampling_run_params
        )
        return model_output

    def _get_metadata(self, model_output):
        metadata = dotdict(
            usage=(
                model_output.usage.to_dict() if model_output.usage is not None else {}
            ),
            details={
                "size": getattr(model_output, "size", None),
                "quality": getattr(model_output, "quality", None),
                "output_format": getattr(model_output, "output_format", None),
                "background": getattr(model_output, "background", None),
            },
        )
        return metadata

    def _generate(self, **kwargs):
        response = ModelResponse()
        response.set_response_type("image_generation")

        model_output = self._execute_model(**kwargs)

        metadata = self._get_metadata(model_output)

        images = []
        for item in model_output.data:
            if item.url:
                images.append(item.url)
            if item.b64_json:
                images.append(item.b64_json)

        if len(images) == 1:
            images = images[0]

        response.add(images)
        response.set_metadata(metadata)

        return response

    async def _agenerate(self, **kwargs):
        response = ModelResponse()
        response.set_response_type("image_generation")

        model_output = await self._aexecute_model(**kwargs)

        metadata = self._get_metadata(model_output)

        images = []
        for item in model_output.data:
            if item.url:
                images.append(item.url)
            if item.b64_json:
                images.append(item.b64_json)

        if len(images) == 1:
            images = images[0]

        response.add(images)
        response.set_metadata(metadata)

        return response

    def __call__(
        self,
        prompt: str,
        *,
        response_format: Optional[Literal["url", "base64"]] = None,
        n: Optional[int] = 1,
        size: Optional[str] = None,
        quality: Optional[str] = None,
        background: Optional[Literal["transparent", "opaque", "auto"]] = None,
    ) -> ModelResponse:
        """Args:
        prompt:
            A text description of the desired image(s).
        response_format:
            Format in which images are returned.
        n:
            The number of images to generate.
        size:
            The size of the generated images.
        quality:
            The quality of the image that will be generated.
        background:
            Allows to set transparency for the background of the generated image(s).
        """
        generation_params = dotdict(prompt=prompt, n=n, model=self.model_id)

        if size is not None:
            generation_params.size = size
        if quality is not None:
            generation_params.quality = quality
        if background is not None:
            generation_params.background = background
        if response_format is not None:
            if response_format == "base64":
                response_format = "b64_json"
            generation_params.response_format = response_format

        response = self._generate(**generation_params)
        return response

    async def acall(
        self,
        prompt: str,
        *,
        response_format: Optional[Literal["url", "base64"]] = None,
        n: Optional[int] = 1,
        size: Optional[str] = None,
        quality: Optional[str] = None,
        background: Optional[Literal["transparent", "opaque", "auto"]] = None,
    ) -> ModelResponse:
        """Async version of __call__. Args:
        prompt:
            A text description of the desired image(s).
        response_format:
            Format in which images are returned.
        n:
            The number of images to generate.
        size:
            The size of the generated images.
        quality:
            The quality of the image that will be generated.
        background:
            Allows to set transparency for the background of the generated image(s).
        """
        generation_params = dotdict(prompt=prompt, n=n, model=self.model_id)

        if size is not None:
            generation_params.size = size
        if quality is not None:
            generation_params.quality = quality
        if background is not None:
            generation_params.background = background
        if response_format is not None:
            if response_format == "base64":
                response_format = "b64_json"
            generation_params.response_format = response_format

        response = await self._agenerate(**generation_params)
        return response


@register_model
class OpenAIImageTextToImage(ImageTextToImageModel, OpenAITextToImage):
    """OpenAI Image Edit."""

    def _execute_model(self, **kwargs):
        model_output = self.client.images.edit(**kwargs, **self.sampling_run_params)
        return model_output

    async def _aexecute_model(self, **kwargs):
        model_output = await self.aclient.images.edit(
            **kwargs, **self.sampling_run_params
        )
        return model_output

    def _prepare_inputs(self, image, mask):
        inputs = {}
        if isinstance(image, (str, bytes)):
            image = [image]
        inputs["image"] = [encode_data_to_bytes(item) for item in image]
        if mask:
            inputs["mask"] = encode_data_to_bytes(mask)
        return inputs

    def __call__(
        self,
        prompt: str,
        image: Union[str, List[str]],
        *,
        mask: Optional[str] = None,
        response_format: Optional[Literal["url", "base64"]] = None,
        n: Optional[int] = 1,
    ) -> ModelResponse:
        """Args:
        prompt:
            A text description of the desired image(s).
        image:
            The image(s) to edit. Can be a path, an url or base64 string.
        mask:
            An additional image whose fully transparent areas
            (e.g. where alpha is zero) indicate where image
            should be edited. If there are multiple images provided,
            the mask will be applied on the first image.
        response_format:
            Format in which images are returned.
        n:
            The number of images to generate.
        """
        generation_params = dotdict(prompt=prompt, n=n, model=self.model_id)

        if response_format is not None:
            if response_format == "base64":
                response_format = "b64_json"
            generation_params.response_format = response_format

        inputs = self._prepare_inputs(image, mask)
        response = self._generate(**generation_params, **inputs)
        return response

    async def acall(
        self,
        prompt: str,
        image: Union[str, List[str]],
        *,
        mask: Optional[str] = None,
        response_format: Optional[Literal["url", "base64"]] = None,
        n: Optional[int] = 1,
    ) -> ModelResponse:
        """Async version of __call__. Args:
        prompt:
            A text description of the desired image(s).
        image:
            The image(s) to edit. Can be a path, an url or base64 string.
        mask:
            An additional image whose fully transparent areas
            (e.g. where alpha is zero) indicate where image
            should be edited. If there are multiple images provided,
            the mask will be applied on the first image.
        response_format:
            Format in which images are returned.
        n:
            The number of images to generate.
        """
        generation_params = dotdict(prompt=prompt, n=n, model=self.model_id)

        if response_format is not None:
            if response_format == "base64":
                response_format = "b64_json"
            generation_params.response_format = response_format

        inputs = self._prepare_inputs(image, mask)
        response = await self._agenerate(**generation_params, **inputs)
        return response


@register_model
class OpenAISpeechToText(_BaseOpenAI, SpeechToTextModel):
    """OpenAI Speech to Text."""

    def __init__(
        self,
        *,
        model_id: str,
        temperature: Optional[float] = 0.0,
        base_url: Optional[str] = None,
        retry: Optional[Any] = None,
    ):
        """Args:
        model_id:
            Model ID in provider.
        temperature:
            The sampling temperature, between 0 and 1.
        base_url:
            URL to model provider.
        retry:
            Retry config. A tenacity decorator, False to disable, or None for default.
        """
        super().__init__()
        self.model_id = model_id
        self.sampling_params = {"base_url": base_url or self._get_base_url()}
        self.sampling_run_params = {"temperature": temperature}
        self.retry = retry
        self._initialize()
        self._get_api_key()

    def _execute_model(self, **kwargs):
        model_output = self.client.audio.transcriptions.create(
            **kwargs, **self.sampling_run_params
        )
        return model_output

    async def _aexecute_model(self, **kwargs):
        model_output = await self.aclient.audio.transcriptions.create(
            **kwargs, **self.sampling_run_params
        )
        return model_output

    def _generate(self, **kwargs):
        response = ModelResponse()

        model_output = self._execute_model(**kwargs)

        response.set_response_type("transcript")

        transcript = {}

        if isinstance(model_output, str):
            transcript["text"] = model_output
        else:
            if model_output.text:
                transcript["text"] = model_output.text
            words = getattr(model_output, "words", None)
            if words:
                transcript["words"] = [
                    {"word": w.word, "start": w.start, "end": w.end} for w in words
                ]
            segments = getattr(model_output, "segments", None)
            if segments:
                transcript["segments"] = [
                    {"id": seg.id, "start": seg.start, "end": seg.end, "text": seg.text}
                    for seg in segments
                ]

        response.add(transcript)

        return response

    async def _agenerate(self, **kwargs):
        response = ModelResponse()

        model_output = await self._aexecute_model(**kwargs)

        response.set_response_type("transcript")

        transcript = {}

        if isinstance(model_output, str):
            transcript["text"] = model_output
        else:
            if model_output.text:
                transcript["text"] = model_output.text
            words = getattr(model_output, "words", None)
            if words:
                transcript["words"] = [
                    {"word": w.word, "start": w.start, "end": w.end} for w in words
                ]
            segments = getattr(model_output, "segments", None)
            if segments:
                transcript["segments"] = [
                    {"id": seg.id, "start": seg.start, "end": seg.end, "text": seg.text}
                    for seg in segments
                ]
                transcript["segments"] = segments

        response.add(transcript)

        return response

    def _stream_generate(self, **kwargs):
        stream_response = kwargs.pop("stream_response")
        stream_response.set_response_type("transcript")

        model_output = self._execute_model(**kwargs)

        for event in model_output:
            if event.type == "transcript.text.delta":
                chunk = event.delta
                if chunk:
                    stream_response.add(chunk)
                    if not stream_response.first_chunk_event.is_set():
                        stream_response.first_chunk_event.set()
            elif event.type == "transcript.text.done":
                stream_response.add(None)

        return stream_response

    async def _astream_generate(self, **kwargs):
        stream_response = kwargs.pop("stream_response")
        stream_response.set_response_type("transcript")

        model_output = await self._aexecute_model(**kwargs)

        async for event in model_output:
            if event.type == "transcript.text.delta":
                chunk = event.delta
                if chunk:
                    stream_response.add(chunk)
                    if not stream_response.first_chunk_event.is_set():
                        stream_response.first_chunk_event.set()
            elif event.type == "transcript.text.done":
                stream_response.add(None)

        return stream_response

    def __call__(
        self,
        data: str,
        *,
        stream: Optional[bool] = False,
        response_format: Optional[
            Literal["json", "text", "srt", "verbose_json", "vtt"]
        ] = "text",
        timestamp_granularities: Optional[List[str]] = None,
        prompt: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Union[ModelResponse, ModelStreamResponse]:
        """Args:
        data:
            Url, path, base64 to audio.
        stream:
            Whether generation should be in streaming mode.
        response_format:
            The format of the output, in one of these options:
            json, text, srt, verbose_json, or vtt.
        timestamp_granularities:
            The timestamp granularities to populate for this
            transcription. `response_format` must be set `verbose_json`
            to use timestamp granularities. Either or both of these
            options are supported: word, or segment. Note: There is no
            additional latency for segment timestamps, but generating
            word timestamps incurs additional latency.
        prompt:
            An optional text to guide the model's style or continue a
            previous audio segment. The prompt should match the audio language.
        language:
            The language of the input audio. Supplying the input language in
            ISO-639-1 (e.g. en) format will improve accuracy and latency.
        """
        file = encode_data_to_bytes(data)
        params = {
            "file": file,
            "language": language,
            "response_format": response_format,
            "timestamp_granularities": timestamp_granularities,
            "prompt": prompt,
            "model": self.model_id,
        }
        if stream:
            stream_response = ModelStreamResponse(mode="sync")
            params["stream_response"] = stream_response
            params["stream"] = stream
            F.spawn(self._stream_generate, **params)
            F.wait_for_event(stream_response.first_chunk_event)
            return stream_response
        else:
            response = self._generate(**params)
            return response

    async def acall(
        self,
        data: str,
        *,
        stream: Optional[bool] = False,
        response_format: Optional[
            Literal["json", "text", "srt", "verbose_json", "vtt"]
        ] = "text",
        timestamp_granularities: Optional[List[str]] = None,
        prompt: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Union[ModelResponse, ModelStreamResponse]:
        """Async version of __call__. Args:
        data:
            Url, path, base64 to audio.
        stream:
            Whether generation should be in streaming mode.
        response_format:
            The format of the output, in one of these options:
            json, text, srt, verbose_json, or vtt.
        timestamp_granularities:
            The timestamp granularities to populate for this
            transcription. `response_format` must be set `verbose_json`
            to use timestamp granularities. Either or both of these
            options are supported: word, or segment. Note: There is no
            additional latency for segment timestamps, but generating
            word timestamps incurs additional latency.
        prompt:
            An optional text to guide the model's style or continue a
            previous audio segment. The prompt should match the audio language.
        language:
            The language of the input audio. Supplying the input language in
            ISO-639-1 (e.g. en) format will improve accuracy and latency.
        """
        file = encode_data_to_bytes(data)
        params = {
            "file": file,
            "language": language,
            "response_format": response_format,
            "timestamp_granularities": timestamp_granularities,
            "prompt": prompt,
            "model": self.model_id,
        }
        if stream:
            stream_response = ModelStreamResponse(mode="async")
            params["stream_response"] = stream_response
            params["stream"] = stream
            await F.aspawn(self._astream_generate, **params)
            await F.await_for_event(stream_response.first_chunk_event)
            return stream_response
        else:
            response = await self._agenerate(**params)
            return response


@register_model
class OpenAITextEmbedder(_BaseOpenAI, TextEmbedderModel):
    """OpenAI Text Embedder."""

    batch_support: bool = True

    def __init__(
        self,
        *,
        model_id: str,
        dimensions: Optional[int] = None,
        base_url: Optional[str] = None,
        enable_cache: Optional[bool] = False,
        cache_size: Optional[int] = 128,
        retry: Optional[Any] = None,
    ):
        """Args:
        model_id:
            Model ID in provider.
        dimensions:
            The number of dimensions the resulting output embeddings should have.
        base_url:
            URL to model provider.
        enable_cache:
            If True, enables response caching to avoid redundant API calls.
        cache_size:
            Maximum number of responses to cache (default: 128).
        retry:
            Retry config. A tenacity decorator, False to disable, or None for default.
        """
        super().__init__()
        self.model_id = model_id
        self.sampling_params = {"base_url": base_url or self._get_base_url()}
        self.sampling_run_params = {"dimensions": dimensions}
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.retry = retry
        self._initialize()
        self._get_api_key()

    def _execute_model(self, **kwargs):
        model_output = self.client.embeddings.create(
            **kwargs,
            **self.sampling_run_params,
        )
        return model_output

    async def _aexecute_model(self, **kwargs):
        model_output = await self.aclient.embeddings.create(
            **kwargs,
            **self.sampling_run_params,
        )
        return model_output

    def _generate(self, **kwargs):
        # Check cache if enabled
        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(**kwargs)
            hit, cached_response = self._response_cache.get(cache_key)
            if hit:
                return cached_response

        response = ModelResponse()
        response.set_response_type("text_embedding")
        model_output = self._execute_model(**kwargs)
        embeddings = [item.embedding for item in model_output.data]
        metadata = dotdict({"usage": model_output.usage.to_dict()})
        response.add(embeddings)
        response.set_metadata(metadata)

        # Store in cache if enabled
        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(**kwargs)
            self._response_cache.set(cache_key, response)

        return response

    async def _agenerate(self, **kwargs):
        # Check cache if enabled
        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(**kwargs)
            hit, cached_response = self._response_cache.get(cache_key)
            if hit:
                return cached_response

        response = ModelResponse()
        response.set_response_type("text_embedding")
        model_output = await self._aexecute_model(**kwargs)
        embeddings = [item.embedding for item in model_output.data]
        metadata = dotdict({"usage": model_output.usage.to_dict()})
        response.add(embeddings)
        response.set_metadata(metadata)

        # Store in cache if enabled
        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(**kwargs)
            self._response_cache.set(cache_key, response)

        return response

    def __call__(
        self,
        data: Union[str, List[str]],
    ):
        """Args:
        data:
            Input text to embed.
        """
        response = self._generate(input=data, model=self.model_id)
        return response

    async def acall(
        self,
        data: Union[str, List[str]],
    ):
        """Async version of __call__. Args:
        data:
            Input text to embed.
        """
        response = await self._agenerate(input=data, model=self.model_id)
        return response


@register_model
class OpenAIModeration(_BaseOpenAI, ModerationModel):
    """OpenAI Moderation."""

    def __init__(
        self,
        *,
        model_id: str,
        base_url: Optional[str] = None,
        enable_cache: Optional[bool] = False,
        cache_size: Optional[int] = 128,
        retry: Optional[Any] = None,
    ):
        """Args:
        model_id:
            Model ID in provider.
        base_url:
            URL to model provider.
        enable_cache:
            If True, enables response caching to avoid redundant API calls.
        cache_size:
            Maximum number of responses to cache (default: 128).
        retry:
            Retry config. A tenacity decorator, False to disable, or None for default.
        """
        super().__init__()
        self.model_id = model_id
        self.sampling_params = {"base_url": base_url or self._get_base_url()}
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.retry = retry
        self._initialize()
        self._get_api_key()

    def _execute_model(self, **kwargs):
        model_output = self.client.moderations.create(**kwargs)
        return model_output

    async def _aexecute_model(self, **kwargs):
        model_output = await self.aclient.moderations.create(**kwargs)
        return model_output

    def _generate(self, **kwargs):
        # Check cache if enabled
        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(**kwargs)
            hit, cached_response = self._response_cache.get(cache_key)
            if hit:
                return cached_response

        response = ModelResponse()
        response.set_response_type("moderation")
        model_output = self._execute_model(**kwargs)
        moderation = dotdict({"results": model_output.results[0].model_dump()})
        moderation.safe = not moderation.results.flagged
        response.add(moderation)

        # Store in cache if enabled
        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(**kwargs)
            self._response_cache.set(cache_key, response)

        return response

    async def _agenerate(self, **kwargs):
        # Check cache if enabled
        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(**kwargs)
            hit, cached_response = self._response_cache.get(cache_key)
            if hit:
                return cached_response

        response = ModelResponse()
        response.set_response_type("moderation")
        model_output = await self._aexecute_model(**kwargs)
        moderation = dotdict({"results": model_output.results[0].model_dump()})
        moderation.safe = not moderation.results.flagged
        response.add(moderation)

        # Store in cache if enabled
        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(**kwargs)
            self._response_cache.set(cache_key, response)

        return response

    def __call__(
        self,
        data: Union[str, List[Dict[str, Any]]],
    ) -> ModelResponse:
        """Args:
        data:
            Input (or inputs) to classify. Can be a single string,
            an array of strings, or an array of multi-modal input
            objects similar to other models.
        """
        response = self._generate(input=data, model=self.model_id)
        return response

    async def acall(
        self,
        data: Union[str, List[Dict[str, Any]]],
    ) -> ModelResponse:
        """Async version of __call__. Args:
        data:
            Input (or inputs) to classify. Can be a single string,
            an array of strings, or an array of multi-modal input
            objects similar to other models.
        """
        response = await self._agenerate(input=data, model=self.model_id)
        return response
