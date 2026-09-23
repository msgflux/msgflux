"""Google-free native Anthropic Messages API provider (no OpenAI wire)."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Dict, Iterator, Mapping, Optional

import httpx2

from msgflux.core.dotdict import dotdict
from msgflux.exceptions import ModelProviderHTTPError
from msgflux.models.cache import ResponseCache
from msgflux.models.chat_api import ChatAPIAdapter
from msgflux.models.chat_capabilities import (
    ChatAPIModeCapabilities,
    ChatProviderCapabilities,
)
from msgflux.models.openai_compatible import OpenAICompatibleChatCompletion
from msgflux.models.provider_env import ProviderEnvBase
from msgflux.models.reasoning import AnthropicThinkingCodec
from msgflux.models.registry import register_model
from msgflux.utils.tenacity import apply_retry, default_model_retry

ANTHROPIC_API_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096

_EFFORT_TO_ANTHROPIC = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}

_STOP_REASON_TO_FINISH = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
    "model_context_window_exceeded": "length",
    "pause_turn": "tool_calls",
}


class _BaseAnthropic(ProviderEnvBase):
    """Configurations for Anthropic Claude models."""

    provider: str = "anthropic"
    display_name: str = "Anthropic"
    api_key_env: str = "ANTHROPIC_API_KEY"
    base_url_env: str = "ANTHROPIC_BASE_URL"
    base_url: str = "https://api.anthropic.com"


class AnthropicMessagesAPI(ChatAPIAdapter):
    """Anthropic native Messages wire protocol."""

    name = "anthropic_messages"
    endpoint = "/v1/messages"
    canonical_history = True

    def prepare_request(self, owner, params):
        raise NotImplementedError(
            "Anthropic models send native requests directly; "
            "use the owner wire converter instead."
        )

    def build_generation_params(self, owner, *args, **kwargs):
        return owner._build_anthropic_generation_params(*args, **kwargs)

    def process_output(self, owner, *args, **kwargs):
        return owner._process_completion_model_output(*args, **kwargs)

    def decode_response(self, payload):
        return payload

    def decode_stream_event(self, payload):
        return payload


def _response_value(payload: Any, field: str, default: Any = None) -> Any:
    if isinstance(payload, Mapping):
        return payload.get(field, default)
    return getattr(payload, field, default)


def _set_field(payload: Any, field: str, value: Any) -> None:
    if isinstance(payload, Mapping):
        payload[field] = value
    else:
        setattr(payload, field, value)


def _parse_data_url(url: str) -> tuple[str, str] | None:
    if not url.startswith("data:"):
        return None
    header, _, encoded = url.partition(",")
    if not encoded:
        return None
    media_type = header[5:].split(";", 1)[0] or "image/jpeg"
    return media_type, encoded


@register_model
class AnthropicChatCompletion(_BaseAnthropic, OpenAICompatibleChatCompletion):
    """Anthropic Claude chat completion through the Messages API.

    `POST /v1/messages` on `https://api.anthropic.com` with
    `x-api-key` and `anthropic-version: 2023-06-01` headers. This is a
    native (non OpenAI-compatible) protocol: chat history, tools, and
    reasoning are translated at the model boundary following the same
    adapter pattern as the native Ollama transport.

    Reasoning maps msgFlux `reasoning_effort` to adaptive thinking with
    `output_config.effort` (`low`/`medium`/`high`/`xhigh`/`max`,
    `minimal` approximates `low`); `"none"` disables thinking on models
    that allow it. `reasoning_max_tokens` instead requests manual
    extended thinking (`budget_tokens`) and cannot be combined with
    `reasoning_effort`. Thinking summaries are always requested
    (`display: "summarized"`). Thinking and `redacted_thinking` blocks
    are stored verbatim in history and replayed unmodified, as the API
    requires (modified blocks are rejected); `max_tokens` defaults to
    4096 when unset since the API requires it and thinking tokens count
    toward it. `prompt_cache=True` sends top-level automatic
    `cache_control`; usage is aggregated so `input_tokens` includes
    cache reads/writes (unlike the native split) and
    `cache_hit_percentage` stays meaningful.

    Scope notes: client function tools only (server-side tools such as
    web search/code execution, structured outputs via
    `output_config.format`, and temperature/top_p deprecations on newer
    models are caller responsibility and documented follow-ups).
    `generation_schema` and non-empty `extra_body` raise instead of
    failing remotely with an unknown-field error.
    """

    capabilities = ChatProviderCapabilities(
        default_api_mode="anthropic_messages",
        api_modes=(
            ChatAPIModeCapabilities(
                name="anthropic_messages",
                adapter=AnthropicMessagesAPI(),
                request_reasoning_effort=True,
            ),
        ),
        default_reasoning_codec=AnthropicThinkingCodec(),
    )

    def __init__(
        self,
        model_id: str,
        *,
        prompt_cache: bool = False,
        **kwargs: Any,
    ):
        """Args:
        model_id:
            Model ID in provider.
        prompt_cache:
            Send top-level automatic `cache_control` so repeated prefixes
            are cached server-side (5-minute TTL).
        kwargs:
            Remaining `OpenAICompatibleChatCompletion` options.
        """
        super().__init__(model_id, **kwargs)
        if not isinstance(prompt_cache, bool):
            raise TypeError("`prompt_cache` must be a boolean")
        self.prompt_cache = prompt_cache
        if (
            self.reasoning_max_tokens is not None
            and self.sampling_run_params.get("reasoning_effort") is not None
        ):
            raise ValueError(
                "`reasoning_max_tokens` cannot be used together with "
                "`reasoning_effort` for Anthropic."
            )

    def _initialize(self):
        self.current_key_index = 0
        self.client = httpx2.Client(timeout=None)
        self.aclient = httpx2.AsyncClient(timeout=None)
        self._response_cache = (
            ResponseCache(maxsize=self.cache_size) if self.enable_cache else None
        )
        self.__call__ = apply_retry(
            self.__call__, self.retry, default=default_model_retry
        )
        self.acall = apply_retry(self.acall, self.retry, default=default_model_retry)

    def _native_headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self._get_api_key(),
            "anthropic-version": ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        }

    def _native_url(self) -> str:
        return f"{self.sampling_params['base_url'].rstrip('/')}/v1/messages"

    # -- generation params -------------------------------------------------

    def _build_anthropic_generation_params(
        self,
        messages,
        system_prompt,
        prefilling,
        tool_catalog,
        *,
        logprobs=None,
        top_logprobs=None,
        extra_body=None,
        extra_body_kwargs=None,
    ) -> Dict[str, Any]:
        return self._build_chat_completions_generation_params(
            messages,
            system_prompt,
            prefilling,
            tool_catalog,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            extra_body=extra_body,
            extra_body_kwargs=extra_body_kwargs,
        )

    def _thinking_request_config(self) -> Dict[str, Any]:
        effort = self.sampling_run_params.get("reasoning_effort")
        budget = self.reasoning_max_tokens
        if budget is not None and effort is not None:
            raise ValueError(
                "`reasoning_max_tokens` cannot be used together with "
                "`reasoning_effort` for Anthropic."
            )
        if budget is not None:
            return {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": budget,
                    "display": "summarized",
                }
            }
        if not isinstance(effort, str):
            return {}
        normalized = effort.strip().lower()
        if normalized == "none":
            return {"thinking": {"type": "disabled"}}
        level = _EFFORT_TO_ANTHROPIC.get(normalized, normalized)
        if normalized == "minimal":
            level = "low"
        return {
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": level},
        }

    @staticmethod
    def _convert_tool_choice(tool_choice: Any, *, parallel_tool_calls: bool) -> Any:
        disable_parallel = (
            {} if parallel_tool_calls else {"disable_parallel_tool_use": True}
        )
        if tool_choice is None or tool_choice == "auto":
            choice: Any = {"type": "auto"}
        elif tool_choice in {"required", "any"}:
            choice = {"type": "any"}
        elif tool_choice == "none":
            choice = {"type": "none"}
        elif isinstance(tool_choice, Mapping):
            function = tool_choice.get("function") or {}
            choice = {"type": "tool", "name": function.get("name")}
        else:
            choice = {"type": "tool", "name": tool_choice}
        if isinstance(choice, dict):
            choice.update(disable_parallel)
        return choice

    @classmethod
    def _convert_tool_schema(cls, tool: Mapping[str, Any]) -> Dict[str, Any]:
        function = dict(tool.get("function") or {})
        converted = {"name": function.get("name")}
        if function.get("description"):
            converted["description"] = function["description"]
        converted["input_schema"] = function.get("parameters") or {"type": "object"}
        return converted

    @classmethod
    def _convert_user_content(cls, content: Any) -> Any:
        if content is None or isinstance(content, str):
            return content
        if not isinstance(content, list):
            raise TypeError(
                "Anthropic user content must be a string or a content-part list"
            )
        blocks: list[Dict[str, Any]] = []
        for part in content:
            if not isinstance(part, Mapping):
                raise TypeError("Anthropic content parts must be mappings")
            part_type = part.get("type")
            if part_type in {"text", "input_text"}:
                blocks.append({"type": "text", "text": part.get("text", "")})
            elif part_type == "image_url":
                image = part.get("image_url") or {}
                url = image.get("url", "") if isinstance(image, Mapping) else ""
                parsed = _parse_data_url(url) if isinstance(url, str) else None
                if parsed is not None:
                    media_type, data = parsed
                    blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": data,
                            },
                        }
                    )
                elif isinstance(url, str) and url.startswith(("http://", "https://")):
                    blocks.append(
                        {"type": "image", "source": {"type": "url", "url": url}}
                    )
                else:
                    raise ValueError(
                        "Anthropic image inputs must be data URLs or http(s) URLs"
                    )
            else:
                raise TypeError(
                    f"Unsupported Anthropic content part type: {part_type!r}"
                )
        return blocks

    @classmethod
    def _convert_assistant_message(cls, message: Mapping[str, Any]) -> Dict[str, Any]:
        content: list[Dict[str, Any]] = []
        thinking_blocks = message.get("thinking_blocks")
        if isinstance(thinking_blocks, list):
            content.extend(deepcopy(block) for block in thinking_blocks)
        text = message.get("content")
        if isinstance(text, str) and text:
            content.append({"type": "text", "text": text})
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            raw_arguments = function.get("arguments")
            if isinstance(raw_arguments, Mapping):
                arguments = dict(raw_arguments)
            else:
                try:
                    arguments = json.loads(raw_arguments or "{}")
                except (TypeError, ValueError):
                    arguments = {}
                if not isinstance(arguments, dict):
                    arguments = {}
            content.append(
                {
                    "type": "tool_use",
                    "id": call.get("id"),
                    "name": function.get("name"),
                    "input": arguments,
                }
            )
        return {"role": "assistant", "content": content}

    @classmethod
    def _convert_messages(
        cls, messages: list[Dict[str, Any]], *, prefilling: Optional[str]
    ) -> tuple[Any, list[Dict[str, Any]]]:
        system: Any = None
        converted: list[Dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role == "system":
                text = message.get("content")
                system = text if system is None else system
                continue
            if role == "assistant":
                converted.append(cls._convert_assistant_message(message))
                continue
            if role == "tool":
                converted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message.get("tool_call_id"),
                                "content": message.get("content") or "",
                            }
                        ],
                    }
                )
                continue
            converted.append(
                {
                    "role": "user",
                    "content": cls._convert_user_content(message.get("content")),
                }
            )
        if prefilling:
            converted.append({"role": "assistant", "content": prefilling})
        return system, converted

    def _to_anthropic_body(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if params.get("generation_schema") is not None:
            raise ValueError(
                "Anthropic provider does not support `generation_schema` yet; "
                "structured outputs are a follow-up."
            )
        merged_extra_body = params.get("extra_body")
        if merged_extra_body:
            raise ValueError(
                "Anthropic provider does not support `extra_body` yet; "
                "pass provider options through the documented init params."
            )
        if params.get("response_format") is not None:
            raise ValueError(
                "Anthropic provider does not support `generation_schema` "
                "(`response_format`) yet; structured outputs are a follow-up."
            )
        body: Dict[str, Any] = {
            "model": params.get("model", self.model_id),
            "max_tokens": params.get("max_tokens") or DEFAULT_MAX_TOKENS,
            "stream": bool(params.get("stream", False)),
        }
        system, body["messages"] = self._convert_messages(
            params.get("messages") or [], prefilling=params.get("prefilling")
        )
        if system is not None:
            body["system"] = system
        for source, target in (
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("stop", "stop_sequences"),
        ):
            value = params.get(source)
            if value is not None:
                body[target] = value
        tools = params.get("tools")
        if tools:
            body["tools"] = [self._convert_tool_schema(tool) for tool in tools]
            body["tool_choice"] = self._convert_tool_choice(
                params.get("tool_choice"),
                parallel_tool_calls=params.get("parallel_tool_calls", True),
            )
        body.update(self._thinking_request_config())
        if self.prompt_cache:
            body["cache_control"] = {"type": "ephemeral"}
        return body

    # -- native transport --------------------------------------------------

    def _raise_anthropic_error_for_payload(
        self, *, status_code: int, headers: Any, raw: Any
    ) -> None:
        if isinstance(raw, (bytes, str)):
            try:
                payload = json.loads(raw)
            except ValueError:
                payload = {}
        else:
            payload = raw
        error = payload.get("error") if isinstance(payload, Mapping) else {}
        description = (
            error.get("message") if isinstance(error, Mapping) else str(raw)[:500]
        )
        request_id = None
        if headers is not None:
            getter = getattr(headers, "get", None)
            if callable(getter):
                request_id = getter("request-id")
        raise ModelProviderHTTPError(
            status_code=status_code,
            description=str(description),
            provider=self.provider,
            model_id=self.model_id,
            error_type=error.get("type") if isinstance(error, Mapping) else None,
            request_id=request_id,
            response=raw,
        )

    def _raise_anthropic_error(self, response: Any) -> None:
        try:
            raw = response.json()
        except ValueError:
            raw = response.text
        self._raise_anthropic_error_for_payload(
            status_code=response.status_code,
            headers=getattr(response, "headers", None),
            raw=raw,
        )

    def _execute_model(self, **kwargs):
        self._raise_if_aborted()
        body = self._to_anthropic_body({**self.sampling_run_params, **kwargs})
        if body.get("stream"):
            return self._stream_chunks(body)
        response = self.client.post(
            self._native_url(), headers=self._native_headers(), json=body
        )
        if response.status_code >= 400:
            self._raise_anthropic_error(response)
        self._raise_if_aborted()
        return self._anthropic_to_completion(response.json(), stream=False)

    async def _aexecute_model(self, **kwargs):
        self._raise_if_aborted()
        body = self._to_anthropic_body({**self.sampling_run_params, **kwargs})
        if body.get("stream"):
            return self._astream_chunks(body)

        async def _post():
            return await self.aclient.post(
                self._native_url(), headers=self._native_headers(), json=body
            )

        response = await _post()
        if response.status_code >= 400:
            self._raise_anthropic_error(response)
        self._raise_if_aborted()
        return self._anthropic_to_completion(response.json(), stream=False)

    # -- native -> completion conversion -----------------------------------

    @staticmethod
    def _aggregate_usage(usage: Mapping[str, Any]) -> Dict[str, Any]:
        read = usage.get("cache_read_input_tokens") or 0
        created = usage.get("cache_creation_input_tokens") or 0
        uncached = usage.get("input_tokens") or 0
        aggregated = dict(usage)
        aggregated["input_tokens"] = read + created + uncached
        return aggregated

    @classmethod
    def _split_content_blocks(
        cls, content: Any
    ) -> tuple[str | None, str | None, list[Dict[str, Any]], list[Dict[str, Any]]]:
        thinking_texts: list[str] = []
        thinking_blocks: list[Dict[str, Any]] = []
        tool_calls: list[Dict[str, Any]] = []
        texts: list[str] = []
        for index, block in enumerate(content or []):
            if not isinstance(block, Mapping):
                continue
            block_type = block.get("type")
            if block_type == "thinking":
                if block.get("thinking"):
                    thinking_texts.append(str(block["thinking"]))
                thinking_blocks.append(
                    {
                        "type": "thinking",
                        "thinking": block.get("thinking") or "",
                        "signature": block.get("signature"),
                    }
                )
            elif block_type == "redacted_thinking":
                thinking_blocks.append(
                    {"type": "redacted_thinking", "data": block.get("data")}
                )
            elif block_type == "text":
                if block.get("text"):
                    texts.append(str(block["text"]))
            elif block_type == "tool_use":
                tool_calls.append(
                    {
                        "index": index,
                        "id": block.get("id"),
                        "type": "function",
                        "function": {
                            "name": block.get("name"),
                            "arguments": json.dumps(block.get("input") or {}),
                        },
                    }
                )
        text = "".join(texts) or None
        thinking = "".join(thinking_texts) or None
        return text, thinking, thinking_blocks, tool_calls

    @classmethod
    def _anthropic_to_completion(cls, payload: Mapping[str, Any], *, stream: bool):
        text, thinking, thinking_blocks, tool_calls = cls._split_content_blocks(
            payload.get("content")
        )
        usage = cls._aggregate_usage(payload.get("usage") or {})
        converted = {
            "id": payload.get("id"),
            "model": payload.get("model"),
            "usage": usage,
            "choices": [
                {
                    "finish_reason": _STOP_REASON_TO_FINISH.get(
                        payload.get("stop_reason")
                    ),
                    "logprobs": None,
                    ("delta" if stream else "message"): {
                        "content": text,
                        "tool_calls": tool_calls or None,
                        "audio": None,
                        "annotations": None,
                        **({"thinking": thinking} if thinking is not None else {}),
                        **(
                            {"thinking_blocks": thinking_blocks}
                            if thinking_blocks
                            else {}
                        ),
                    },
                }
            ],
        }
        service_tier = usage.get("service_tier")
        if isinstance(service_tier, str) and service_tier:
            converted["service_tier"] = service_tier
        return dotdict(converted)

    def _process_completion_model_output(
        self, model_output, generation_schema=None, transport_generation_schema=None
    ):
        response = super()._process_completion_model_output(
            model_output, generation_schema, transport_generation_schema
        )
        message = None
        choices = _response_value(model_output, "choices", []) or []
        if choices:
            message = _response_value(choices[0], "message")
        blocks = _response_value(message, "thinking_blocks", []) if message else []
        if not blocks:
            return response
        history_items: list[Dict[str, Any]] = []
        for block in blocks:
            if not isinstance(block, Mapping):
                continue
            item: Dict[str, Any] = {"type": "reasoning", "role": "assistant"}
            if block.get("type") == "thinking" and block.get("thinking"):
                item["text"] = str(block["thinking"])
            item["provider_state"] = {
                **self.reasoning_codec.state_identity(
                    provider=self.provider,
                    api_mode=self.api_mode,
                ),
                "data": deepcopy(dict(block)),
            }
            history_items.append(item)
        if history_items:
            response.history_items = history_items
        return response

    # -- streaming ---------------------------------------------------------

    @staticmethod
    def _iter_sse_events(lines: Iterator[str]) -> Iterator[tuple[str | None, Any]]:
        event_type: str | None = None
        data_lines: list[str] = []
        for raw_line in lines:
            line = raw_line.rstrip("\r\n")
            if not line:
                if data_lines:
                    payload = json.loads("\n".join(data_lines))
                    yield event_type, payload
                    event_type = None
                    data_lines = []
                continue
            if line.startswith(":"):
                continue
            field, _, value = line.partition(":")
            value = value.lstrip(" ")
            if field == "event":
                event_type = value or None
            elif field == "data":
                data_lines.append(value)
        if data_lines:
            yield event_type, json.loads("\n".join(data_lines))

    @staticmethod
    async def _aiter_sse_events(lines) -> Any:
        event_type: str | None = None
        data_lines: list[str] = []
        async for raw_line in lines:
            line = raw_line.rstrip("\r\n")
            if not line:
                if data_lines:
                    payload = json.loads("\n".join(data_lines))
                    yield event_type, payload
                    event_type = None
                    data_lines = []
                continue
            if line.startswith(":"):
                continue
            field, _, value = line.partition(":")
            value = value.lstrip(" ")
            if field == "event":
                event_type = value or None
            elif field == "data":
                data_lines.append(value)
        if data_lines:
            yield event_type, json.loads("\n".join(data_lines))

    @classmethod
    def _blank_delta(cls) -> Dict[str, Any]:
        return {
            "content": None,
            "tool_calls": None,
            "audio": None,
            "annotations": None,
        }

    @classmethod
    def _delta_chunk(
        cls,
        *,
        delta: Dict[str, Any],
        finish_reason: Any = None,
        usage: Any = None,
    ):
        chunk: Dict[str, Any] = {
            "choices": [
                {"finish_reason": finish_reason, "logprobs": None, "delta": delta}
            ]
        }
        if usage is not None:
            chunk["usage"] = usage
        return dotdict(chunk)

    @staticmethod
    def _new_stream_state() -> Dict[str, Any]:
        return {"blocks": {}, "start_usage": {}}

    def _handle_stream_event(  # noqa: C901
        self, event_type: str | None, data: Any, state: Dict[str, Any]
    ) -> list:
        blocks = state["blocks"]
        chunks: list = []
        if not isinstance(data, Mapping):
            return chunks
        if event_type == "error":
            error = data.get("error") or {}
            raise ModelProviderHTTPError(
                status_code=400,
                description=str(error.get("message", data)),
                provider=self.provider,
                model_id=self.model_id,
                error_type=error.get("type"),
            )
        if event_type == "message_start":
            message = data.get("message") or {}
            if isinstance(message.get("usage"), Mapping):
                state["start_usage"] = dict(message["usage"])
            return chunks
        if event_type == "content_block_start":
            index = data.get("index")
            block = data.get("content_block") or {}
            blocks[index] = {"type": block.get("type"), "block": dict(block)}
            if block.get("type") == "redacted_thinking":
                chunks.append(
                    self._delta_chunk(
                        delta={
                            **self._blank_delta(),
                            "thinking_block": {
                                "type": "redacted_thinking",
                                "data": block.get("data"),
                            },
                        }
                    )
                )
            return chunks
        if event_type == "content_block_delta":
            index = data.get("index")
            delta = data.get("delta") or {}
            delta_type = delta.get("type")
            if delta_type == "thinking_delta":
                thinking = delta.get("thinking") or ""
                block_state = blocks.get(index)
                if isinstance(block_state, dict):
                    block_state["thinking"] = (
                        f"{block_state.get('thinking', '')}{thinking}"
                    )
                chunks.append(
                    self._delta_chunk(
                        delta={**self._blank_delta(), "thinking": thinking or None}
                    )
                )
            elif delta_type == "signature_delta":
                block_state = blocks.get(index)
                if isinstance(block_state, dict):
                    block_state["block"] = {
                        **block_state.get("block", {}),
                        "signature": delta.get("signature"),
                    }
            elif delta_type == "text_delta":
                chunks.append(
                    self._delta_chunk(
                        delta={**self._blank_delta(), "content": delta.get("text")}
                    )
                )
            elif delta_type == "input_json_delta":
                block_state = blocks.get(index)
                partial = delta.get("partial_json") or ""
                tool_use = (block_state or {}).get("block") or {}
                if isinstance(block_state, dict):
                    block_state["input"] = f"{block_state.get('input', '')}{partial}"
                chunks.append(
                    self._delta_chunk(
                        delta={
                            **self._blank_delta(),
                            "tool_calls": [
                                {
                                    "index": index,
                                    "id": tool_use.get("id"),
                                    "type": "function",
                                    "function": {
                                        "name": tool_use.get("name"),
                                        "arguments": partial,
                                    },
                                }
                            ],
                        }
                    )
                )
            return chunks
        if event_type == "content_block_stop":
            index = data.get("index")
            block_state = blocks.pop(index, None)
            if not isinstance(block_state, dict):
                return chunks
            if block_state.get("type") == "thinking":
                block = dict(block_state.get("block") or {})
                block["thinking"] = block_state.get("thinking", "")
                chunks.append(
                    self._delta_chunk(
                        delta={**self._blank_delta(), "thinking_block": block}
                    )
                )
            elif block_state.get("type") == "tool_use":
                tool_use = block_state.get("block") or {}
                chunks.append(
                    self._delta_chunk(
                        delta={
                            **self._blank_delta(),
                            "tool_calls": [
                                {
                                    "index": index,
                                    "id": tool_use.get("id"),
                                    "type": "function",
                                    "function": {
                                        "name": tool_use.get("name"),
                                        "arguments": block_state.get("input") or "",
                                    },
                                }
                            ],
                        }
                    )
                )
            return chunks
        if event_type == "message_delta":
            message_usage = data.get("usage") or {}
            merged = (
                {**state["start_usage"], **message_usage}
                if isinstance(message_usage, Mapping)
                else dict(state["start_usage"])
            )
            stop = (data.get("delta") or {}).get("stop_reason")
            chunks.append(
                self._delta_chunk(
                    delta=self._blank_delta(),
                    finish_reason=_STOP_REASON_TO_FINISH.get(stop),
                    usage=self._aggregate_usage(merged),
                )
            )
            return chunks
        return chunks

    def _expand_stream_events(self, events: Iterator[tuple[str | None, Any]]):
        state = self._new_stream_state()
        for event_type, data in events:
            yield from self._handle_stream_event(event_type, data, state)

    async def _expand_astream_events(self, events):
        state = self._new_stream_state()
        async for event_type, data in events:
            for item in self._handle_stream_event(event_type, data, state):
                yield item

    def _stream_chunks(self, body: Dict[str, Any]):
        with self.client.stream(
            "POST", self._native_url(), headers=self._native_headers(), json=body
        ) as response:
            if response.status_code >= 400:
                self._raise_anthropic_error_for_payload(
                    status_code=response.status_code,
                    headers=response.headers,
                    raw=response.read(),
                )
            self._raise_if_aborted()
            yield from self._expand_stream_events(
                self._iter_sse_events(response.iter_lines())
            )

    async def _astream_chunks(self, body: Dict[str, Any]):
        async with self.aclient.stream(
            "POST", self._native_url(), headers=self._native_headers(), json=body
        ) as response:
            if response.status_code >= 400:
                self._raise_anthropic_error_for_payload(
                    status_code=response.status_code,
                    headers=response.headers,
                    raw=await response.aread(),
                )
            self._raise_if_aborted()
            async for item in self._expand_astream_events(
                self._aiter_sse_events(response.aiter_lines())
            ):
                yield item

    def warmup_system_prompt(self, *, system_prompt, tool_catalog=None):
        generation_params = self._build_generation_params(
            [], system_prompt, None, tool_catalog
        )
        generation_params["max_tokens"] = self.warmup_max_tokens
        generation_params.pop("prefilling", None)
        return self._execute_model(**generation_params)

    async def awarmup_system_prompt(self, *, system_prompt, tool_catalog=None):
        generation_params = self._build_generation_params(
            [], system_prompt, None, tool_catalog
        )
        generation_params["max_tokens"] = self.warmup_max_tokens
        generation_params.pop("prefilling", None)
        return await self._aexecute_model(**generation_params)
