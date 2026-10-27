"""Native Google Gemini Interactions chat-provider integration."""

from __future__ import annotations

import json
import re
from contextlib import aclosing, closing
from copy import deepcopy
from typing import Any, Dict, Iterator, Mapping, Optional, Union

import msgspec

from msgflux.chat_messages import ChatMessages
from msgflux.core.dotdict import dotdict
from msgflux.exceptions import ModelProviderHTTPError
from msgflux.models.chat_api import ChatAPIAdapter, PreparedChatRequest
from msgflux.models.chat_capabilities import (
    ChatAPIModeCapabilities,
    ChatProviderCapabilities,
)
from msgflux.models.http_transport import HTTPTransport
from msgflux.models.model_credentials import (
    ModelCredentialResolver,
    ResolvedModelCredentials,
)
from msgflux.models.openai_compatible import OpenAICompatibleChatCompletion
from msgflux.models.provider_env import ProviderEnvBase
from msgflux.models.reasoning import GeminiInteractionsReasoningCodec
from msgflux.models.registry import register_model
from msgflux.models.response import ModelResponse
from msgflux.models.sse import aiter_sse_json, iter_sse_json
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.models.usage import default_usage_codec
from msgflux.tools.catalog import ToolCatalogView
from msgflux.tools.definitions import ToolCatalog
from msgflux.utils.msgspec import struct_to_dict

_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
_DEFAULT_API_VERSION = "v1"

_REASONING_LEVELS = {
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}

_FINISH_REASONS = {
    "completed": "stop",
    "requires_action": "tool_calls",
    "incomplete": "length",
    "budget_exceeded": "length",
    "cancelled": "stop",
}

_SUPPORTED_THINKING_SUMMARIES = {"auto", "none"}

# Native Interactions top-level request fields that are safe to forward from
# ``extra_body`` without changing the canonical history.
_NATIVE_EXTRA_FIELDS = {"labels", "safety_settings", "service_tier"}

# Parameters accepted by the shared OpenAI-compatible base but not expressible
# in the native Interactions model request in this round.
_UNSUPPORTED_CONSTRUCTOR_OPTIONS = (
    "audio",
    "enable_thinking",
    "logprobs",
    "modalities",
    "prompt_cache_retention",
    "reasoning_max_tokens",
    "speed",
    "top_logprobs",
    "verbosity",
    "web_search_options",
)

_UNSUPPORTED_REQUEST_PARAMETERS = (
    "audio",
    "background",
    "enable_thinking",
    "environment",
    "environment_id",
    "logprobs",
    "max_total_tokens",
    "modalities",
    "previous_interaction_id",
    "prompt_cache_retention",
    "reasoning_max_tokens",
    "speed",
    "top_logprobs",
    "verbosity",
    "web_search_options",
)

_DATA_URL_RE = re.compile(r"^data:(?P<mime>[^;,]+)?(?P<base64>;base64)?,")
_TEXT_PART_TYPES = {"input_text", "text"}
_IMAGE_PART_TYPES = {"image", "image_url", "input_image"}


def _value(payload: Any, field: str, default: Any = None) -> Any:
    if isinstance(payload, Mapping):
        return payload.get(field, default)
    return getattr(payload, field, default)


def _parse_data_url(url: str) -> tuple[str | None, str] | None:
    """Return ``(mime_type, base64_data)`` for a base64 data URL."""
    match = _DATA_URL_RE.match(url)
    if match is None or not match.group("base64"):
        return None
    return match.group("mime"), url[match.end() :]


def _plain(value: Any) -> Any:
    """Convert dot-accessible mappings into plain JSON-compatible containers."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return deepcopy(value)


class GeminiCredentialResolver(ModelCredentialResolver):
    """Resolve the Google API key using the native header."""

    def resolve(self, owner: Any) -> ResolvedModelCredentials:
        return ResolvedModelCredentials(
            headers={"x-goog-api-key": owner._get_api_key()}
        )


class _BaseGemini(ProviderEnvBase):
    """Configurations for Google Gemini models."""

    provider: str = "gemini"
    display_name: str = "Gemini"
    api_key_env: str = "GEMINI_API_KEY"
    base_url_env: str = "GEMINI_BASE_URL"
    base_url: str = _DEFAULT_BASE_URL
    credential_resolver: ModelCredentialResolver = GeminiCredentialResolver()


class GeminiInteractionsAPI(ChatAPIAdapter):
    """Native Gemini Interactions wire protocol."""

    name = "interactions"
    endpoint = "/v1/interactions"
    canonical_history = True

    def prepare_request(self, owner, params):
        raise NotImplementedError(
            "Gemini Interactions models send native requests directly; "
            "use the owner request builder instead."
        )

    def build_generation_params(self, owner, *args, **kwargs):
        return owner._build_interactions_generation_params(*args, **kwargs)

    def process_output(self, owner, *args, **kwargs):
        return owner._process_interaction_output(*args, **kwargs)

    def decode_response(self, payload):
        return payload

    def decode_stream_event(self, payload):
        return payload

    def stream(self, owner, **kwargs):
        return owner._stream_chat_completions_generate(**kwargs)

    async def astream(self, owner, **kwargs):
        return await owner._astream_chat_completions_generate(**kwargs)


@register_model
class GeminiChatCompletion(_BaseGemini, OpenAICompatibleChatCompletion):
    """Google Gemini through the native Interactions API.

    ``POST /v1/interactions`` with ``x-goog-api-key``. Requests are stateless
    by default (``store=false``) and send the complete canonical history as
    ordered native steps, so local checkpoints remain authoritative. Thought
    steps, including their opaque signatures, are preserved in provider state
    and replayed verbatim. Server-side state through
    ``previous_interaction_id`` is an explicit follow-up.

    Reasoning effort maps to ``generation_config.thinking_level``. Gemini
    exposes ``minimal``/``low``/``medium``/``high``; ``none`` cannot be
    represented by this API and is rejected instead of silently enabling
    thinking. Thought summaries are requested when reasoning is enabled and
    the caller wants reasoning returned.

    Client function tools only: function declarations are flattened into the
    native ``function`` tool shape and forced tool choice uses
    ``generation_config.tool_choice.allowed_tools``. Managed agents,
    background execution, remote MCP, hosted search and explicit server state
    are intentionally out of scope for this round.
    """

    capabilities = ChatProviderCapabilities(
        default_api_mode="interactions",
        api_modes=(
            ChatAPIModeCapabilities(
                name="interactions",
                adapter=GeminiInteractionsAPI(),
                request_reasoning_effort=True,
            ),
        ),
        default_reasoning_codec=GeminiInteractionsReasoningCodec(),
    )
    usage_codec = default_usage_codec

    def __init__(
        self,
        model_id: str,
        *,
        store: bool | None = None,
        thinking_summaries: str | bool | None = None,
        api_version: str | None = None,
        **kwargs: Any,
    ):
        """Args:
        model_id:
            Model ID in the Gemini API.
        store:
            Whether Google should retain the interaction. Defaults to
            ``False`` so local checkpoints stay authoritative. Storage does
            not enable ``previous_interaction_id`` replay in this provider.
        thinking_summaries:
            ``"auto"``/``True`` to request thought summaries, ``"none"``/
            ``False`` to omit them, or ``None`` to enable summaries when a
            reasoning level is configured and reasoning is returned.
        api_version:
            Interactions API version in the request path. Defaults to the
            stable ``v1`` endpoint.
        kwargs:
            Remaining ``OpenAICompatibleChatCompletion`` options. Options
            that the native model request cannot represent are rejected.
        """
        if store is None:
            store = False
        if not isinstance(store, bool):
            raise TypeError("`store` must be a boolean or None")
        if api_version is None:
            api_version = _DEFAULT_API_VERSION
        if not isinstance(api_version, str) or not api_version.strip():
            raise ValueError("`api_version` must be a non-empty string")
        self.store = store
        self.thinking_summaries = self._coerce_thinking_summaries(
            value=thinking_summaries
        )
        self.api_version = api_version.strip().strip("/")
        self._reject_unsupported_constructor_options(kwargs)
        super().__init__(model_id, store=store, **kwargs)
        if kwargs.get("credential_resolver") is None:
            self._get_api_key()

    @staticmethod
    def _coerce_thinking_summaries(
        *,
        value: str | bool | None,
    ) -> str | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return "auto" if value else "none"
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in _SUPPORTED_THINKING_SUMMARIES:
                return normalized
        raise ValueError(
            "`thinking_summaries` must be 'auto', 'none', a boolean, or None"
        )

    @staticmethod
    def _reject_unsupported_constructor_options(options: Mapping[str, Any]) -> None:
        rejected = []
        for name in _UNSUPPORTED_CONSTRUCTOR_OPTIONS:
            value = options.get(name)
            if value is not None:
                rejected.append(f"`{name}`")
        if options.get("parallel_tool_calls") is False:
            rejected.append("`parallel_tool_calls=False`")
        if options.get("chat_transport") is not None:
            rejected.append("`chat_transport`")
        if options.get("chat_extensions"):
            rejected.append("`chat_extensions`")
        if rejected:
            joined = ", ".join(rejected)
            raise ValueError(
                f"Native Gemini Interactions does not support these options: {joined}."
            )

    def _initialize(self):
        self.http_transport = HTTPTransport(timeout=None)
        if not hasattr(self, "store"):
            self.store = False
        if not hasattr(self, "thinking_summaries"):
            self.thinking_summaries = None
        if not hasattr(self, "api_version"):
            self.api_version = _DEFAULT_API_VERSION
        super()._initialize()

    def _native_endpoint(self) -> str:
        return f"/{self.api_version}/interactions"

    @staticmethod
    def _native_headers() -> Dict[str, str]:
        return {"content-type": "application/json"}

    def _capture_reasoning_metadata(self, request, metadata):
        generation_config = request.json.get("generation_config")
        level = (
            generation_config.get("thinking_level")
            if isinstance(generation_config, Mapping)
            else None
        )
        metadata["request_reasoning_effort"] = level
        metadata["conversation_reasoning_effort"] = None
        if level is not None:
            metadata["reasoning_effort"] = level
        else:
            metadata.pop("reasoning_effort", None)

    # -- request building --------------------------------------------------

    @staticmethod
    def _content_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, Mapping):
            content = [content]
        if not isinstance(content, (list, tuple)):
            raise TypeError("Message content must be a string or a content list")
        chunks: list[str] = []
        for part in content:
            if not isinstance(part, Mapping):
                raise TypeError("Native Gemini content parts must be mappings")
            if part.get("type") not in _TEXT_PART_TYPES:
                raise ValueError(
                    "System instructions and text-only content cannot include "
                    f"native part type `{part.get('type')}`."
                )
            text = part.get("text")
            if not isinstance(text, str):
                raise TypeError("Text content parts require a string `text`")
            chunks.append(text)
        return "".join(chunks)

    @classmethod
    def _native_content(  # noqa: C901
        cls, content: Any
    ) -> list[dict[str, Any]]:
        if content is None:
            return []
        if isinstance(content, str):
            return [{"type": "text", "text": content}] if content else []
        if isinstance(content, Mapping):
            content = [content]
        if not isinstance(content, (list, tuple)):
            raise TypeError("Message content must be a string or a content list")

        parts: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, Mapping):
                raise TypeError("Native Gemini content parts must be mappings")
            part_type = part.get("type")
            if part_type in _TEXT_PART_TYPES:
                text = part.get("text")
                if not isinstance(text, str):
                    raise TypeError("Text content parts require a string `text`")
                parts.append({"type": "text", "text": text})
            elif part_type in _IMAGE_PART_TYPES:
                parts.append(cls._native_image(part))
            elif part_type == "document":
                parts.append(cls._native_document(part))
            elif part_type in {"input_audio", "audio"}:
                parts.append(cls._native_audio(part))
            else:
                raise ValueError(
                    f"Unsupported native Gemini content part type `{part_type}`."
                )
        return parts

    @staticmethod
    def _native_image(part: Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(part.get("data"), str):
            return {
                "type": "image",
                "data": part["data"],
                **(
                    {"mime_type": part["mime_type"]}
                    if isinstance(part.get("mime_type"), str)
                    else {}
                ),
            }
        if isinstance(part.get("uri"), str) and part["uri"]:
            return {
                "type": "image",
                "uri": part["uri"],
                **(
                    {"mime_type": part["mime_type"]}
                    if isinstance(part.get("mime_type"), str)
                    else {}
                ),
            }
        image = part.get("image_url") or part.get("image") or part.get("url")
        if isinstance(image, Mapping):
            image = image.get("url")
        if not isinstance(image, str) or not image:
            raise ValueError("Image content requires a URL or data URL")
        data_url = _parse_data_url(image)
        if data_url is not None:
            mime_type, data = data_url
            return {
                "type": "image",
                "data": data,
                **({"mime_type": mime_type} if mime_type else {}),
            }
        return {"type": "image", "uri": image}

    @staticmethod
    def _native_document(part: Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(part.get("data"), str):
            return {
                "type": "document",
                "data": part["data"],
                **(
                    {"mime_type": part["mime_type"]}
                    if isinstance(part.get("mime_type"), str)
                    else {}
                ),
            }
        uri = part.get("uri") or part.get("url")
        if isinstance(uri, str) and uri:
            return {
                "type": "document",
                "uri": uri,
                **(
                    {"mime_type": part["mime_type"]}
                    if isinstance(part.get("mime_type"), str)
                    else {}
                ),
            }
        raise ValueError("Document content requires `data` or `uri`")

    @staticmethod
    def _native_audio(part: Mapping[str, Any]) -> dict[str, Any]:
        audio = part.get("input_audio") or part.get("audio") or part
        if isinstance(audio, Mapping):
            data = audio.get("data")
            audio_format = audio.get("format") or audio.get("mime_type")
        else:
            data = None
            audio_format = None
        if not isinstance(data, str) or not data:
            raise ValueError("Audio content requires base64 `data`")
        mime_type = None
        if isinstance(audio_format, str):
            mime_type = (
                audio_format if "/" in audio_format else f"audio/{audio_format.lower()}"
            )
        return {
            "type": "audio",
            "data": data,
            **({"mime_type": mime_type} if mime_type else {}),
        }

    @staticmethod
    def _provider_state_step(
        item: Mapping[str, Any],
        expected_type: str,
        *,
        provider: str,
        api_mode: str,
    ) -> dict[str, Any] | None:
        state = item.get("provider_state")
        if (
            not isinstance(state, Mapping)
            or state.get("provider") != provider
            or state.get("api_mode") not in {None, api_mode}
        ):
            return None
        data = state.get("data")
        if isinstance(data, Mapping) and data.get("type") == expected_type:
            return _plain(dict(data))
        return None

    @staticmethod
    def _arguments_object(arguments: Any) -> Dict[str, Any]:
        if arguments is None:
            return {}
        if isinstance(arguments, Mapping):
            return deepcopy(dict(arguments))
        if isinstance(arguments, str):
            if not arguments.strip():
                return {}
            try:
                parsed = json.loads(arguments)
            except ValueError as error:
                raise ValueError(
                    "Gemini function call arguments must be valid JSON"
                ) from error
            if not isinstance(parsed, Mapping):
                raise ValueError("Gemini function call arguments must be a JSON object")
            return dict(parsed)
        raise TypeError(
            "Gemini function call arguments must be a mapping or JSON object string"
        )

    @staticmethod
    def _arguments_json(arguments: Any) -> str:
        if isinstance(arguments, str):
            return arguments or "{}"
        return json.dumps(arguments or {}, separators=(",", ":"))

    def _canonical_items(self, messages: Any) -> list[dict[str, Any]]:
        if isinstance(messages, ChatMessages):
            canonical = messages.materialize_context(
                provider=self.provider,
                api_mode=self.api_mode,
            )
            return canonical.to_items()
        if isinstance(messages, str):
            return [{"type": "message", "role": "user", "content": messages}]
        canonical = ChatMessages(messages).materialize_context(
            provider=self.provider,
            api_mode=self.api_mode,
        )
        return canonical.to_items()

    def _to_native_input_steps(  # noqa: C901
        self,
        items: list[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], str | None]:
        steps: list[dict[str, Any]] = []
        system_chunks: list[str] = []
        call_names: dict[str, str] = {}

        for source in items:
            item = dict(source)
            item_type = item.get("type")
            if item_type in {None, "message"} and item.get("role") in {
                "developer",
                "system",
            }:
                text = self._content_text(item.get("content"))
                if text:
                    system_chunks.append(text)
                continue
            if item_type in {"turn", "model_configuration"}:
                continue
            if item_type == "reasoning":
                text = item.get("text")
                if not isinstance(text, str) or not text:
                    text = None
                step = self._provider_state_step(
                    item,
                    "thought",
                    provider=self.provider,
                    api_mode=self.api_mode,
                )
                if step is None:
                    step = {"type": "thought"}
                if text is not None and (
                    GeminiInteractionsReasoningCodec.summary_text(step) != text
                ):
                    step["summary"] = [{"type": "text", "text": text}]
                elif not isinstance(step.get("summary"), list):
                    step["summary"] = []
                steps.append(step)
                continue
            if item_type == "function_call":
                call_id = item.get("call_id") or item.get("id")
                name = item.get("name")
                if not isinstance(call_id, str) or not call_id:
                    raise ValueError("Gemini function calls require a non-empty id")
                if not isinstance(name, str) or not name:
                    raise ValueError("Gemini function calls require a name")
                arguments = self._arguments_object(item.get("arguments"))
                call_names[call_id] = name
                step = self._provider_state_step(
                    item,
                    "function_call",
                    provider=self.provider,
                    api_mode=self.api_mode,
                )
                if (
                    step is None
                    or step.get("id") != call_id
                    or step.get("name") != name
                    or step.get("arguments") != arguments
                ):
                    step = {
                        "type": "function_call",
                        "id": call_id,
                        "name": name,
                        "arguments": arguments,
                    }
                steps.append(step)
                continue
            if item_type == "function_call_output":
                call_id = item.get("call_id")
                if not isinstance(call_id, str) or not call_id:
                    raise ValueError(
                        "Gemini function results require a non-empty call_id"
                    )
                result = item.get("output")
                step = self._provider_state_step(
                    item,
                    "function_result",
                    provider=self.provider,
                    api_mode=self.api_mode,
                )
                if step is None or step.get("call_id") != call_id:
                    step = {"type": "function_result", "call_id": call_id}
                    if call_id in call_names:
                        step["name"] = call_names[call_id]
                    step["result"] = self._native_result(result)
                steps.append(step)
                continue

            role = item.get("role")
            if item_type in {None, "message"} and role == "user":
                steps.append(
                    {
                        "type": "user_input",
                        "content": self._native_content(item.get("content")),
                    }
                )
                continue
            if item_type in {None, "message"} and role == "assistant":
                content = self._native_content(item.get("content"))
                step = self._provider_state_step(
                    item,
                    "model_output",
                    provider=self.provider,
                    api_mode=self.api_mode,
                )
                if step is None or self._native_content(step.get("content")) != content:
                    step = {"type": "model_output", "content": content}
                steps.append(step)
                continue
            if item_type in {None, "message"} and role == "tool":
                call_id = item.get("tool_call_id")
                if not isinstance(call_id, str) or not call_id:
                    raise ValueError(
                        "Gemini function results require a non-empty call_id"
                    )
                step = {"type": "function_result", "call_id": call_id}
                if call_id in call_names:
                    step["name"] = call_names[call_id]
                step["result"] = self._native_result(item.get("content"))
                steps.append(step)
                continue

            raise ValueError(
                "Native Gemini Interactions cannot replay canonical item type "
                f"`{item_type}` with role `{role}`."
            )

        system_instruction = "\n\n".join(chunk for chunk in system_chunks if chunk)
        return steps, system_instruction or None

    def _native_result(self, result: Any) -> Any:
        if isinstance(result, str):
            return result
        if isinstance(result, Mapping):
            return deepcopy(dict(result))
        if isinstance(result, (list, tuple)):
            converted: list[Any] = []
            for part in result:
                if isinstance(part, Mapping) and part.get("type") in (
                    _TEXT_PART_TYPES | _IMAGE_PART_TYPES
                ):
                    converted.extend(self._native_content([part]))
                else:
                    converted.append(deepcopy(part))
            return converted
        if result is None:
            return ""
        raise TypeError(
            "Gemini function results must be strings, mappings, or content lists"
        )

    @staticmethod
    def _native_tools(
        catalog: Optional[Union[ToolCatalog, ToolCatalogView]],
    ) -> tuple[list[dict[str, Any]], Any]:
        if catalog is None:
            return [], None
        tools: list[dict[str, Any]] = []
        for schema in GeminiChatCompletion._portable_tool_schemas(catalog):
            function = schema.get("function")
            if not isinstance(function, Mapping):
                continue
            name = function.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("Gemini function tools require a name")
            native: dict[str, Any] = {"type": "function", "name": name}
            if isinstance(function.get("description"), str):
                native["description"] = function["description"]
            parameters = function.get("parameters")
            if isinstance(parameters, Mapping):
                native["parameters"] = deepcopy(dict(parameters))
            tools.append(native)
        return tools, GeminiChatCompletion._catalog_choice_value(catalog)

    @staticmethod
    def _native_tool_choice(choice: Any) -> Any:
        if choice is None:
            return None
        if isinstance(choice, Mapping):
            if "allowed_tools" in choice:
                return deepcopy(dict(choice))
            function = choice.get("function")
            if choice.get("type") == "function" and isinstance(function, Mapping):
                choice = function.get("name")
            elif choice.get("type") == "function":
                choice = choice.get("name")
            else:
                raise ValueError(
                    "Gemini tool choice mappings must select a function tool"
                )
        if not isinstance(choice, str) or not choice.strip():
            raise TypeError("Gemini tool choice must be a string or mapping")
        normalized = choice.strip()
        if normalized == "auto":
            return "auto"
        if normalized in {"any", "required"}:
            return "any"
        if normalized in {"none", "validated"}:
            return normalized
        return {"allowed_tools": {"mode": "any", "tools": [normalized]}}

    @staticmethod
    def _thinking_level(reasoning_effort: str) -> str:
        if not isinstance(reasoning_effort, str):
            raise TypeError("`reasoning_effort` must be a string")
        normalized = reasoning_effort.strip().lower()
        if normalized == "none":
            raise ValueError(
                "Gemini Interactions cannot disable thinking with "
                "`reasoning_effort='none'`; use 'minimal' as the lowest level."
            )
        level = _REASONING_LEVELS.get(normalized)
        if level is None:
            supported = ", ".join(sorted(_REASONING_LEVELS))
            raise ValueError(
                f"Unsupported Gemini reasoning effort `{reasoning_effort}`; "
                f"supported values: {supported}."
            )
        return level

    def _thinking_summaries_for(self, reasoning_effort: str | None) -> str | None:
        if self.thinking_summaries is not None:
            return self.thinking_summaries
        if reasoning_effort is not None and (
            self.return_reasoning or self.reasoning_in_tool_call
        ):
            return "auto"
        return None

    def _build_interactions_generation_params(
        self,
        messages: Union[str, list[dict[str, Any]], ChatMessages],
        system_prompt: Optional[str],
        prefilling: Optional[str],
        tool_catalog: Optional[Union[ToolCatalog, ToolCatalogView]],
        *,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        extra_body_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if prefilling:
            raise ValueError(
                "Native Gemini Interactions does not support assistant prefilling."
            )
        if logprobs is not None or top_logprobs is not None:
            raise ValueError(
                "Native Gemini Interactions does not expose token log probabilities."
            )
        if isinstance(messages, list) and any(
            isinstance(item, Mapping) and item.get("type") is not None
            for item in messages
        ):
            messages = ChatMessages(messages)
        items = self._canonical_items(messages)
        steps, system_instruction = self._to_native_input_steps(items)
        if system_prompt:
            system_instruction = (
                f"{system_prompt}\n\n{system_instruction}"
                if system_instruction
                else system_prompt
            )
        if not steps:
            raise ValueError(
                "Native Gemini Interactions requires at least one input step."
            )

        generation_params: Dict[str, Any] = {
            "model": self.model_id,
            "input": steps,
            "parallel_tool_calls": self.parallel_tool_calls,
        }
        if system_instruction:
            generation_params["system_instruction"] = system_instruction
        tools, tool_choice = self._native_tools(tool_catalog)
        if tools:
            generation_params["tools"] = tools
            if tool_choice is not None:
                generation_params["tool_choice"] = tool_choice
        merged_extra_body = self._merge_extra_body(
            self.sampling_run_params.get("extra_body"),
            extra_body,
            extra_body_kwargs,
        )
        if merged_extra_body is not None:
            generation_params["extra_body"] = merged_extra_body
        return generation_params

    def _adapt_interactions_params(  # noqa: C901
        self,
        params: Dict[str, Any],
        *,
        stream: bool,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": params.pop("model", self.model_id),
            "input": params.pop("input"),
            "store": bool(self.store),
        }
        params.pop("store", None)
        params.pop("stream", None)
        system_instruction = params.pop("system_instruction", None)
        if system_instruction:
            body["system_instruction"] = system_instruction
        tools = params.pop("tools", None)
        if tools:
            body["tools"] = tools

        parallel_tool_calls = params.pop("parallel_tool_calls", True)
        if parallel_tool_calls is False:
            raise ValueError(
                "Native Gemini Interactions does not support disabling "
                "parallel tool calls."
            )

        generation_config: Dict[str, Any] = {}
        max_tokens = params.pop("max_tokens", None)
        if max_tokens is not None:
            generation_config["max_output_tokens"] = max_tokens
        for parameter, target in (
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("seed", "seed"),
        ):
            value = params.pop(parameter, None)
            if value is not None:
                generation_config[target] = value
        stop = params.pop("stop", None)
        if stop is not None:
            generation_config["stop_sequences"] = (
                [stop] if isinstance(stop, str) else list(stop)
            )

        reasoning_effort = params.pop("reasoning_effort", None)
        if reasoning_effort is not None:
            generation_config["thinking_level"] = self._thinking_level(reasoning_effort)
            summaries = self._thinking_summaries_for(reasoning_effort)
            if summaries is not None:
                generation_config["thinking_summaries"] = summaries
        elif self.thinking_summaries is not None:
            generation_config["thinking_summaries"] = self.thinking_summaries

        tool_choice = params.pop("tool_choice", None)
        if tool_choice is not None:
            generation_config["tool_choice"] = self._native_tool_choice(tool_choice)

        response_format = params.pop("response_format", None)
        if response_format is not None:
            body["response_format"] = self._native_response_format(response_format)

        extra_body = params.pop("extra_body", None)
        if extra_body:
            if not isinstance(extra_body, Mapping):
                raise TypeError("`extra_body` must be a mapping")
            extra = deepcopy(dict(extra_body))
            extra_generation = extra.pop("generation_config", None)
            if extra_generation is not None:
                if not isinstance(extra_generation, Mapping):
                    raise TypeError("`generation_config` must be a mapping")
                explicit_generation = deepcopy(dict(extra_generation))
                self._validate_generation_config(explicit_generation)
                generation_config.update(explicit_generation)
            unknown = set(extra) - _NATIVE_EXTRA_FIELDS
            if unknown:
                formatted = ", ".join(f"`{name}`" for name in sorted(unknown))
                raise ValueError(
                    "Unsupported native Gemini request fields in `extra_body`: "
                    f"{formatted}."
                )
            body.update(extra)

        if generation_config:
            body["generation_config"] = generation_config
        body["stream"] = bool(stream)

        params.pop("stream_options", None)
        params.pop("provider_tools", None)
        params.pop("tool_catalog", None)
        params.pop("prefilling", None)
        rejected = [
            f"`{name}`"
            for name in _UNSUPPORTED_REQUEST_PARAMETERS
            if params.get(name) is not None
        ]
        if rejected:
            joined = ", ".join(rejected)
            raise ValueError(
                "Native Gemini Interactions does not support these parameters: "
                f"{joined}."
            )
        if params:
            unknown = ", ".join(f"`{name}`" for name in sorted(params))
            raise ValueError(
                f"Unsupported native Gemini request parameters: {unknown}."
            )
        return body

    @staticmethod
    def _validate_generation_config(config: Mapping[str, Any]) -> None:
        level = config.get("thinking_level")
        if level is not None:
            if level == "none":
                raise ValueError(
                    "Gemini Interactions cannot disable thinking with "
                    "`thinking_level='none'`; use 'minimal' as the lowest level."
                )
            if level not in set(_REASONING_LEVELS.values()):
                formatted = ", ".join(sorted(set(_REASONING_LEVELS.values())))
                raise ValueError(
                    f"Unsupported Gemini thinking level `{level}`; "
                    f"supported values: {formatted}."
                )
        summaries = config.get("thinking_summaries")
        if summaries is not None and summaries not in _SUPPORTED_THINKING_SUMMARIES:
            raise ValueError(
                "`thinking_summaries` must be 'auto' or 'none' in native "
                "Gemini `generation_config`."
            )

    @staticmethod
    def _native_response_format(response_format: Any) -> dict[str, Any]:
        if not isinstance(response_format, Mapping):
            raise TypeError("`response_format` must be a mapping")
        format_type = response_format.get("type")
        if format_type == "json_schema":
            config = response_format.get("json_schema")
            schema = config.get("schema") if isinstance(config, Mapping) else None
            if not isinstance(schema, Mapping):
                raise ValueError("JSON schema response format requires `schema`")
            return {
                "type": "text",
                "mime_type": "application/json",
                "schema": deepcopy(dict(schema)),
            }
        if format_type == "json_object":
            return {"type": "text", "mime_type": "application/json"}
        raise ValueError(f"Unsupported Gemini response format `{format_type}`")

    # -- native transport --------------------------------------------------

    def _execute_model(self, **kwargs):
        request_metadata = kwargs.pop("_request_model_metadata", None)
        self._raise_if_aborted()
        body = self._adapt_interactions_params(
            {**self.sampling_run_params, **kwargs},
            stream=bool(kwargs.get("stream")),
        )
        if request_metadata is not None:
            self._capture_reasoning_metadata(
                PreparedChatRequest(
                    api=self.api_mode,
                    endpoint=self._native_endpoint(),
                    params=body,
                ),
                request_metadata,
            )
        if body.get("stream"):
            return self._stream_interactions(body)
        response = self.http_transport.request(
            self,
            self._native_endpoint(),
            headers=self._native_headers(),
            json=body,
        )
        self._raise_if_aborted()
        return response.json()

    async def _aexecute_model(self, **kwargs):
        request_metadata = kwargs.pop("_request_model_metadata", None)
        self._raise_if_aborted()
        body = self._adapt_interactions_params(
            {**self.sampling_run_params, **kwargs},
            stream=bool(kwargs.get("stream")),
        )
        if request_metadata is not None:
            self._capture_reasoning_metadata(
                PreparedChatRequest(
                    api=self.api_mode,
                    endpoint=self._native_endpoint(),
                    params=body,
                ),
                request_metadata,
            )
        if body.get("stream"):
            return self._astream_interactions(body)
        response = await self.http_transport.arequest(
            self,
            self._native_endpoint(),
            headers=self._native_headers(),
            json=body,
        )
        self._raise_if_aborted()
        return response.json()

    def _stream_interactions(self, body: Dict[str, Any]):
        stream = self.http_transport.stream(
            self,
            self._native_endpoint(),
            headers=self._native_headers(),
            json=body,
            iterate=lambda response: self._expand_interaction_stream(
                iter_sse_json(response.iter_lines())
            ),
        )
        with closing(stream):
            yield from stream

    async def _astream_interactions(self, body: Dict[str, Any]):
        stream = self.http_transport.astream(
            self,
            self._native_endpoint(),
            headers=self._native_headers(),
            json=body,
            iterate=lambda response: self._expand_ainteraction_stream(
                aiter_sse_json(response.aiter_lines())
            ),
        )
        async with aclosing(stream):
            async for item in stream:
                yield item

    # -- native response processing ---------------------------------------

    @staticmethod
    def _step_text(step: Mapping[str, Any]) -> str:
        content = step.get("content")
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        chunks: list[str] = []
        for part in content:
            if isinstance(part, Mapping) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks)

    def _raise_interaction_failure(self, payload: Mapping[str, Any], *, status=None):
        errors = _value(payload, "errors")
        description = f"Gemini interaction failed (status={status})"
        if errors:
            description = f"{description}: {json.dumps(errors, default=str)}"
        raise ModelProviderHTTPError(
            status_code=500,
            description=description[:4000],
            provider=self.provider,
            model_id=self.model_id,
            response=payload,
        )

    def _process_interaction_output(  # noqa: C901
        self,
        model_output: Mapping[str, Any],
        generation_schema=None,
        transport_generation_schema=None,
    ):
        payload = (
            dotdict(model_output) if isinstance(model_output, Mapping) else model_output
        )
        status = _value(payload, "status")
        steps = _value(payload, "steps", []) or []
        errors = _value(payload, "errors")
        if status == "failed" or errors:
            self._raise_interaction_failure(payload, status=status)
        response = ModelResponse()

        history_items: list[dict[str, Any]] = []
        reasoning_texts: list[str] = []
        text_chunks: list[str] = []
        calls: list[dict[str, Any]] = []
        has_tool_calls = False

        for source in steps:
            if not isinstance(source, Mapping):
                continue
            step = _plain(dict(source))
            step_type = step.get("type")
            if step_type == "thought":
                text = GeminiInteractionsReasoningCodec.summary_text(step)
                if text:
                    reasoning_texts.append(text)
                item: dict[str, Any] = {"type": "reasoning", "role": "assistant"}
                if text and (self.return_reasoning or self.reasoning_in_tool_call):
                    item["text"] = text
                item["provider_state"] = {
                    **self.reasoning_codec.state_identity(
                        provider=self.provider,
                        api_mode=self.api_mode,
                    ),
                    "data": step,
                }
                history_items.append(item)
                continue
            if step_type == "model_output":
                text = self._step_text(step)
                if text:
                    text_chunks.append(text)
                native_content = self._native_content(step.get("content"))
                canonical_content = (
                    text
                    if all(part.get("type") == "text" for part in native_content)
                    else native_content
                )
                history_items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": canonical_content,
                        "provider_state": {
                            **self.reasoning_codec.state_identity(
                                provider=self.provider,
                                api_mode=self.api_mode,
                            ),
                            "data": step,
                        },
                    }
                )
                continue
            if step_type == "function_call":
                has_tool_calls = True
                call_id = step.get("id")
                name = step.get("name")
                arguments = step.get("arguments")
                if not isinstance(call_id, str) or not call_id:
                    raise ValueError("Gemini function calls require a non-empty id")
                if not isinstance(name, str) or not name:
                    raise ValueError("Gemini function calls require a name")
                arguments_json = self._arguments_json(arguments)
                calls.append(
                    {
                        "call_id": call_id,
                        "name": name,
                        "arguments": arguments_json,
                    }
                )
                history_items.append(
                    {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": name,
                        "arguments": arguments_json,
                        "provider_state": {
                            **self.reasoning_codec.state_identity(
                                provider=self.provider,
                                api_mode=self.api_mode,
                            ),
                            "data": step,
                        },
                    }
                )
                continue
            if step_type == "user_input":
                continue
            raise ValueError(
                f"Unsupported native Gemini output step type `{step_type}`."
            )

        has_commentary = bool(text_chunks) and has_tool_calls
        if has_commentary:
            for item in history_items:
                if item.get("type") == "message" and item.get("role") == "assistant":
                    item["visibility"] = "commentary"
            response.commentary.append("".join(text_chunks))

        reasoning_text = "\n\n".join(reasoning_texts) or None
        response.reasoning = reasoning_text if self.return_reasoning else None
        response.history_items = history_items

        if has_tool_calls:
            aggregator = ToolCallAggregator(
                reasoning_text if self.reasoning_in_tool_call else None,
                api_mode="responses",
            )
            for index, call in enumerate(calls):
                aggregator.process(
                    index,
                    call["call_id"],
                    call["name"],
                    call["arguments"],
                )
            response.set_response_type("tool_call")
            response.add(aggregator)
        else:
            text = "".join(text_chunks)
            if generation_schema is not None:
                response.set_response_type("structured")
                response.add(
                    self._decode_structured_output(
                        text,
                        generation_schema,
                        transport_generation_schema,
                    )
                )
            else:
                response.set_response_type("text_generation")
                response.add(text)

        metadata = self._build_response_metadata(payload)
        finish_reason = (
            "tool_calls" if has_tool_calls else _FINISH_REASONS.get(status, status)
        )
        self._set_stop_metadata(
            metadata,
            finish_reason=finish_reason,
            stop_reason=status,
        )
        response_id = _value(payload, "id")
        if isinstance(response_id, str) and response_id:
            metadata.response_id = response_id
        errors = _value(payload, "errors")
        if errors:
            metadata.errors = deepcopy(errors)
        response.set_metadata(metadata)
        return response

    def _decode_structured_output(
        self,
        text: str,
        generation_schema,
        transport_generation_schema,
    ):
        transport_info = transport_generation_schema or {}
        decoder_schema = transport_info.get("decoder_schema", generation_schema)
        normalize = transport_info.get("normalize")
        if decoder_schema is None:
            decoded = msgspec.json.decode(text)
        else:
            decoder = self._get_decoder(decoder_schema)
            decoded = struct_to_dict(decoder.decode(text))
        if normalize is not None:
            decoded = normalize(decoded)
        decoder = self._get_decoder(generation_schema)
        return dotdict(struct_to_dict(decoder.decode(self._encoder.encode(decoded))))

    # -- streaming ---------------------------------------------------------

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
        delta: Dict[str, Any] | None = None,
        finish_reason: Any = None,
        usage: Any = None,
        model: str | None = None,
        status: str | None = None,
        interaction_id: str | None = None,
    ):
        chunk: Dict[str, Any] = {
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "logprobs": None,
                    "delta": delta or cls._blank_delta(),
                }
            ]
        }
        if usage is not None:
            chunk["usage"] = usage
        if model is not None:
            chunk["model"] = model
        if status is not None:
            chunk["status"] = status
        if interaction_id is not None:
            chunk["id"] = interaction_id
        return dotdict(chunk)

    @staticmethod
    def _new_stream_state() -> Dict[str, Any]:
        return {
            "steps": {},
            "usage": None,
            "status": None,
            "interaction_id": None,
            "model": None,
            "finished": False,
        }

    def _process_stream_tool_calls(self, delta, stream_response, aggregator):
        # Canonical history already contains the assistant function_call step.
        # Responses-style rendering returns only function_call_output items,
        # avoiding the Chat Completions assistant/tool prelude duplicate that
        # Google rejects on native replay.
        aggregator.api_mode = "responses"
        super()._process_stream_tool_calls(delta, stream_response, aggregator)

    @staticmethod
    def _unwrap_stream_event(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if "event_type" in payload:
            return payload
        nested = payload.get("data")
        if isinstance(nested, Mapping):
            return nested
        return payload

    @staticmethod
    def _event_type(payload: Mapping[str, Any]) -> str | None:
        event_type = payload.get("event_type")
        if isinstance(event_type, str):
            return event_type
        if "delta" in payload and "index" in payload:
            return "step.delta"
        if "step" in payload and "index" in payload:
            return "step.start"
        interaction = payload.get("interaction")
        if isinstance(interaction, Mapping) and interaction.get("status"):
            return "interaction.completed"
        return None

    @staticmethod
    def _new_step_state(index: int, step: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "index": index,
            "type": step.get("type"),
            "step": _plain(dict(step)),
            "text_parts": [],
            "summary_parts": [],
            "summary_content": (
                deepcopy(step["summary"])
                if isinstance(step.get("summary"), list)
                else []
            ),
            "signature": None,
            "arguments_parts": [],
            "initial_arguments": deepcopy(step.get("arguments")),
            "id": step.get("id"),
            "name": step.get("name"),
            "text_seen": False,
            "summary_seen": False,
            "arguments_seen": False,
            "emitted": False,
        }

    def _thought_step_from_state(self, state: Mapping[str, Any]) -> Dict[str, Any]:
        step = _plain(dict(state.get("step") or {"type": "thought"}))
        step["type"] = "thought"
        summary = step.get("summary")
        summary_parts = deepcopy(summary) if isinstance(summary, list) else []
        streamed_summary = state.get("summary_content")
        if isinstance(streamed_summary, list) and streamed_summary:
            summary_parts = deepcopy(streamed_summary)
        elif state.get("summary_seen") and state.get("summary_parts"):
            summary_parts = [{"type": "text", "text": "".join(state["summary_parts"])}]
        step["summary"] = summary_parts
        signature = state.get("signature")
        if signature is None:
            signature = step.get("signature")
        if isinstance(signature, str) and signature:
            step["signature"] = signature
        else:
            step.pop("signature", None)
        return step

    @staticmethod
    def _thought_summary_text(step: Mapping[str, Any]) -> str:
        summary = step.get("summary")
        if isinstance(summary, str):
            return summary
        if not isinstance(summary, list):
            return ""
        chunks: list[str] = []
        for part in summary:
            if isinstance(part, Mapping) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks)

    def _handle_stream_event(  # noqa: C901
        self,
        payload: Any,
        state: Dict[str, Any],
    ) -> list[Any]:
        if not isinstance(payload, Mapping):
            return []
        payload = self._unwrap_stream_event(payload)
        event_type = self._event_type(payload)
        chunks: list[Any] = []

        if event_type == "error":
            error = payload.get("error") or {}
            message = error.get("message") if isinstance(error, Mapping) else None
            raise ModelProviderHTTPError(
                status_code=400,
                description=str(message or payload),
                provider=self.provider,
                model_id=self.model_id,
                error_type=(
                    error.get("status") or error.get("type")
                    if isinstance(error, Mapping)
                    else None
                ),
            )

        if event_type == "interaction.created":
            interaction = payload.get("interaction")
            if isinstance(interaction, Mapping):
                state["interaction_id"] = interaction.get("id")
                if interaction.get("model"):
                    state["model"] = interaction.get("model")
                if interaction.get("status"):
                    state["status"] = interaction.get("status")
            return chunks

        if event_type == "interaction.status_update":
            state["status"] = payload.get("status") or state["status"]
            if isinstance(payload.get("usage"), Mapping):
                state["usage"] = payload["usage"]
            return chunks

        if event_type == "step.start":
            index = payload.get("index")
            step = payload.get("step")
            if isinstance(index, int) and isinstance(step, Mapping):
                state["steps"][index] = self._new_step_state(index, step)
            return chunks

        if event_type == "step.delta":
            index = payload.get("index")
            delta = payload.get("delta")
            if not isinstance(index, int) or not isinstance(delta, Mapping):
                return chunks
            step_state = state["steps"].get(index)
            if step_state is None:
                step_state = self._new_step_state(index, {"type": None})
                state["steps"][index] = step_state
            delta_type = delta.get("type")
            if delta_type == "text":
                text = delta.get("text") or ""
                if text:
                    step_state["text_seen"] = True
                    step_state["text_parts"].append(text)
                    step_state["emitted"] = True
                    chunks.append(
                        self._delta_chunk(
                            delta={**self._blank_delta(), "content": text}
                        )
                    )
                return chunks
            if delta_type == "thought_summary":
                content = _plain(delta.get("content"))
                if isinstance(content, Mapping):
                    step_state["summary_content"].append(content)
                    summary = [content]
                elif isinstance(content, str):
                    part = {"type": "text", "text": content}
                    step_state["summary_content"].append(part)
                    summary = [part]
                else:
                    summary = []
                text = self._thought_summary_text({"summary": summary})
                if text:
                    step_state["summary_seen"] = True
                    step_state["summary_parts"].append(text)
                    step_state["emitted"] = True
                    chunks.append(
                        self._delta_chunk(
                            delta={
                                **self._blank_delta(),
                                "reasoning_content": text,
                            }
                        )
                    )
                elif summary:
                    step_state["summary_seen"] = True
                return chunks
            if delta_type == "thought_signature":
                signature = delta.get("signature")
                if isinstance(signature, str) and signature:
                    step_state["signature"] = signature
                return chunks
            if delta_type == "arguments_delta":
                partial = delta.get("arguments")
                if isinstance(partial, str) and partial:
                    step_state["arguments_seen"] = True
                    step_state["arguments_parts"].append(partial)
                    step_state["emitted"] = True
                    call_id = step_state.get("id")
                    name = step_state.get("name")
                    chunks.append(
                        self._delta_chunk(
                            delta={
                                **self._blank_delta(),
                                "tool_calls": [
                                    {
                                        "index": index,
                                        "id": call_id,
                                        "type": "function",
                                        "function": {
                                            "name": name,
                                            "arguments": partial,
                                        },
                                    }
                                ],
                            }
                        )
                    )
                return chunks
            raise ValueError(
                f"Unsupported native Gemini stream delta type `{delta_type}`."
            )

        if event_type == "step.stop":
            index = payload.get("index")
            if isinstance(payload.get("usage"), Mapping):
                state["usage"] = payload["usage"]
            if not isinstance(index, int):
                return chunks
            step_state = state["steps"].get(index)
            if step_state is None:
                return chunks

            if step_state["type"] == "thought":
                if not step_state["summary_seen"]:
                    text = self._thought_summary_text(step_state["step"])
                    if text:
                        step_state["summary_parts"].append(text)
                        chunks.append(
                            self._delta_chunk(
                                delta={
                                    **self._blank_delta(),
                                    "reasoning_content": text,
                                }
                            )
                        )
                thought_step = self._thought_step_from_state(step_state)
                chunks.append(
                    self._delta_chunk(
                        delta={
                            **self._blank_delta(),
                            "thought_step": thought_step,
                        },
                        usage=state["usage"] if not chunks else None,
                    )
                )
            elif step_state["type"] == "model_output":
                if not step_state["text_seen"]:
                    text = self._step_text(step_state["step"])
                    if text:
                        chunks.append(
                            self._delta_chunk(
                                delta={**self._blank_delta(), "content": text},
                                usage=state["usage"],
                            )
                        )
            elif step_state["type"] == "function_call":
                if not step_state["arguments_seen"]:
                    arguments = step_state.get("initial_arguments")
                    if arguments is None:
                        arguments = {}
                    arguments_json = self._arguments_json(arguments)
                    chunks.append(
                        self._delta_chunk(
                            delta={
                                **self._blank_delta(),
                                "tool_calls": [
                                    {
                                        "index": index,
                                        "id": step_state.get("id"),
                                        "type": "function",
                                        "function": {
                                            "name": step_state.get("name"),
                                            "arguments": arguments_json,
                                        },
                                    }
                                ],
                            },
                            usage=state["usage"] if not chunks else None,
                        )
                    )
            else:
                raise ValueError(
                    "Unsupported native Gemini stream step type "
                    f"`{step_state['type']}`."
                )
            return chunks

        if event_type == "interaction.completed":
            interaction = payload.get("interaction")
            if isinstance(interaction, Mapping):
                state["interaction_id"] = (
                    interaction.get("id") or state["interaction_id"]
                )
                if interaction.get("model"):
                    state["model"] = interaction.get("model")
                if interaction.get("status"):
                    state["status"] = interaction.get("status")
                if isinstance(interaction.get("usage"), Mapping):
                    state["usage"] = interaction["usage"]
            status = state["status"]
            errors = (
                interaction.get("errors") if isinstance(interaction, Mapping) else None
            )
            if status == "failed" or errors:
                self._raise_interaction_failure(
                    interaction if isinstance(interaction, Mapping) else {},
                    status=status,
                )
            finish_reason = _FINISH_REASONS.get(status)
            state["finished"] = True
            chunks.append(
                self._delta_chunk(
                    finish_reason=finish_reason,
                    usage=state["usage"],
                    model=state["model"],
                    status=status,
                    interaction_id=state["interaction_id"],
                )
            )
            return chunks

        return chunks

    def _expand_interaction_stream(self, events: Iterator[Mapping[str, Any]]):
        state = self._new_stream_state()
        for payload in events:
            yield from self._handle_stream_event(payload, state)
        if not state["finished"]:
            raise ModelProviderHTTPError(
                status_code=500,
                description=(
                    "Gemini interaction stream ended before interaction.completed."
                ),
                provider=self.provider,
                model_id=self.model_id,
            )

    async def _expand_ainteraction_stream(self, events):
        state = self._new_stream_state()
        async for payload in events:
            for item in self._handle_stream_event(payload, state):
                yield item
        if not state["finished"]:
            raise ModelProviderHTTPError(
                status_code=500,
                description=(
                    "Gemini interaction stream ended before interaction.completed."
                ),
                provider=self.provider,
                model_id=self.model_id,
            )

    def _get_reasoning_effort_metadata(self) -> str | None:
        reasoning_effort = self.sampling_run_params.get("reasoning_effort")
        if not isinstance(reasoning_effort, str):
            return None
        try:
            return self._thinking_level(reasoning_effort)
        except ValueError:
            return reasoning_effort

    def warmup_system_prompt(self, *, system_prompt, tool_catalog=None):
        generation_params = self._build_generation_params(
            "Warmup",
            system_prompt,
            None,
            tool_catalog,
        )
        generation_params["max_tokens"] = self.warmup_max_tokens
        generation_params.pop("prefilling", None)
        return self._execute_model(**generation_params)

    async def awarmup_system_prompt(self, *, system_prompt, tool_catalog=None):
        generation_params = self._build_generation_params(
            "Warmup",
            system_prompt,
            None,
            tool_catalog,
        )
        generation_params["max_tokens"] = self.warmup_max_tokens
        generation_params.pop("prefilling", None)
        return await self._aexecute_model(**generation_params)

    def close(self) -> None:
        self.http_transport.close()
        super().close()

    async def aclose(self) -> None:
        await self.http_transport.aclose()
        await super().aclose()
