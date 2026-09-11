import json
from base64 import b64encode
from os import getenv
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping
from uuid import uuid4

import httpx2

from msgflux.chat_messages import ChatMessages
from msgflux.core.dotdict import dotdict
from msgflux.models.cache import ResponseCache
from msgflux.models.chat_capabilities import (
    ChatAPIModeCapabilities,
    ChatProviderCapabilities,
)
from msgflux.models.openai_compatible import (
    OpenAIChatCompletionsAPI,
    OpenAICompatibleChatCompletion,
)
from msgflux.models.profiles import get_model_profile
from msgflux.models.providers.openai import (
    OpenAITextEmbedder,
)
from msgflux.models.reasoning import (
    OllamaReasoningCodec,
    OpenAICompatibleReasoningCodec,
)
from msgflux.models.registry import register_model
from msgflux.utils.tenacity import apply_retry, default_model_retry


class _BaseOllama:
    """Configurations to use Ollama models."""

    provider: str = "ollama"

    def _get_base_url(self):
        base_url = getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        if base_url is None:
            raise ValueError("Please set `OLLAMA_BASE_URL`")
        return base_url

    def _get_api_key(self):
        """Load API keys from environment variable."""
        key = getenv("OLLAMA_API_KEY", "ollama")
        return key

    @property
    def profile(self):
        """Get model profile from registry.

        Returns:
            ModelProfile if found, None otherwise
        """
        return get_model_profile(self.model_id, provider_id=self.provider)


class OllamaChatAPI(OpenAIChatCompletionsAPI):
    """Ollama native chat protocol using the completion-shaped decoder."""

    name = "ollama_chat"
    canonical_history = True


@register_model
class OllamaChatCompletion(_BaseOllama, OpenAICompatibleChatCompletion):
    """Ollama chat model with native and OpenAI-compatible transports."""

    capabilities = ChatProviderCapabilities(
        default_api_mode="ollama_chat",
        api_modes=(
            ChatAPIModeCapabilities(
                name="ollama_chat",
                adapter=OllamaChatAPI(),
                reasoning_codec=OllamaReasoningCodec(),
            ),
            ChatAPIModeCapabilities(
                name="chat_completions",
                adapter=OpenAIChatCompletionsAPI(),
            ),
        ),
        default_reasoning_codec=OpenAICompatibleReasoningCodec(),
    )

    def _get_reasoning_effort_metadata(self) -> None:
        return None

    def _build_generation_params(self, messages, *args, **kwargs):
        if (
            self.api_mode == "ollama_chat"
            and isinstance(messages, list)
            and any(
                isinstance(item, Mapping) and item.get("type") is not None
                for item in messages
            )
        ):
            messages = ChatMessages(messages)
        return super()._build_generation_params(messages, *args, **kwargs)

    def _get_base_url(self):
        base_url = getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        if self.api_mode == "chat_completions":
            return (
                base_url
                if base_url.rstrip("/").endswith("/v1")
                else f"{base_url.rstrip('/')}/v1"
            )
        if base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/")[:-3]
        return base_url.rstrip("/")

    def _initialize(self):
        if self.api_mode == "chat_completions":
            return super()._initialize()
        self.current_key_index = 0
        timeout_value = getenv("OLLAMA_TIMEOUT")
        timeout = float(timeout_value) if timeout_value else None
        self.client = httpx2.Client(timeout=timeout)
        self.aclient = httpx2.AsyncClient(timeout=timeout)
        self._response_cache = (
            ResponseCache(maxsize=self.cache_size) if self.enable_cache else None
        )
        self.__call__ = apply_retry(
            self.__call__, self.retry, default=default_model_retry
        )
        self.acall = apply_retry(self.acall, self.retry, default=default_model_retry)

    def _adapt_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.api_mode == "ollama_chat":
            return self._adapt_native_params(params)
        extra_body = dict(params.get("extra_body") or {})

        if self.enable_thinking is not None:
            extra_body["think"] = self.enable_thinking

        params["extra_body"] = extra_body
        return params

    @staticmethod
    def _native_image_value(value: Any) -> str:
        if isinstance(value, bytes):
            return b64encode(value).decode()
        if not isinstance(value, str):
            raise TypeError("Ollama image inputs must be bytes or strings")
        if value.startswith("data:"):
            _, separator, encoded = value.partition(",")
            if not separator:
                raise ValueError("Invalid image data URL")
            return encoded
        path = Path(value).expanduser()
        if path.is_file():
            return b64encode(path.read_bytes()).decode()
        if value.startswith(("http://", "https://")):
            raise ValueError(
                "Native Ollama image inputs must be data URLs, base64 strings, "
                "bytes, or local paths; remote URLs are not accepted by /api/chat."
            )
        return value

    @classmethod
    def _native_content(cls, content: Any) -> tuple[str, list[str]]:
        if not isinstance(content, list):
            return content or "", []
        text_parts: list[str] = []
        images: list[str] = []
        for part in content:
            if not isinstance(part, Mapping):
                continue
            if part.get("type") in {"text", "input_text"}:
                text_parts.append(str(part.get("text", "")))
            elif part.get("type") in {"image_url", "input_image"}:
                image = part.get("image_url") or part.get("image")
                if isinstance(image, Mapping):
                    image = image.get("url")
                images.append(cls._native_image_value(image))
        return "".join(text_parts), images

    @staticmethod
    def _native_tool_calls(
        tool_calls: Any, call_names: dict[str, str]
    ) -> list[dict[str, Any]]:
        native_calls = []
        if not isinstance(tool_calls, list):
            return native_calls
        for call in tool_calls:
            function = dict(call.get("function") or {})
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments or "{}")
            call_id = call.get("id")
            name = function.get("name")
            if call_id and name:
                call_names[call_id] = name
            native_calls.append(
                {
                    "type": "function",
                    **({"id": call_id} if call_id else {}),
                    "function": {"name": name, "arguments": arguments},
                }
            )
        return native_calls

    @classmethod
    def _to_native_messages(
        cls, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        native_messages: list[dict[str, Any]] = []
        call_names: dict[str, str] = {}
        for source in messages:
            message = dict(source)
            role = message.get("role")
            native: dict[str, Any] = {"role": role}
            content, images = cls._native_content(message.get("content", ""))
            native["content"] = content or ""
            if images:
                native["images"] = images

            thinking = message.get("thinking")
            if isinstance(thinking, str) and thinking:
                native["thinking"] = thinking

            native_calls = cls._native_tool_calls(message.get("tool_calls"), call_names)
            if native_calls:
                native["tool_calls"] = native_calls

            if role == "tool":
                call_id = message.get("tool_call_id")
                tool_name = message.get("tool_name") or call_names.get(call_id)
                if tool_name:
                    native["tool_name"] = tool_name
            native_messages.append(native)
        return native_messages

    def _adapt_native_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params.pop("provider_tools", None)
        params.pop("tool_catalog", None)
        params.pop("prefilling", None)
        params.pop("stream_options", None)
        params.pop("parallel_tool_calls", None)
        params.pop("logprobs", None)
        params.pop("top_logprobs", None)
        params.pop("reasoning_effort", None)
        params.pop("verbosity", None)
        params.pop("web_search_options", None)

        extra_body = dict(params.pop("extra_body", None) or {})
        options = dict(extra_body.pop("options", None) or {})
        for source, target in (
            ("max_tokens", "num_predict"),
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("stop", "stop"),
        ):
            value = params.pop(source, None)
            if value is not None:
                options[target] = value
        if options:
            params["options"] = options

        response_format = params.pop("response_format", None)
        if isinstance(response_format, Mapping):
            if response_format.get("type") == "json_schema":
                params["format"] = dict(response_format.get("json_schema") or {}).get(
                    "schema", "json"
                )
            elif response_format.get("type") == "json_object":
                params["format"] = "json"

        tool_choice = params.pop("tool_choice", None)
        if tool_choice not in {None, "auto"}:
            raise ValueError(
                "Native Ollama chat does not support explicit `tool_choice`; "
                "use the default automatic selection."
            )
        params["messages"] = self._to_native_messages(params["messages"])
        params["stream"] = bool(params.get("stream", False))
        if self.enable_thinking is not None:
            params["think"] = self.enable_thinking
        params.update(extra_body)
        return params

    @staticmethod
    def _native_usage(payload: Mapping[str, Any]) -> dict[str, Any] | None:
        usage = {
            key: payload[key]
            for key in ("prompt_eval_count", "eval_count")
            if isinstance(payload.get(key), int)
        }
        return usage or None

    @classmethod
    def _native_to_completion(cls, payload: Mapping[str, Any], *, stream: bool):
        message = dict(payload.get("message") or {})
        calls = []
        for index, call in enumerate(message.get("tool_calls") or []):
            function = dict(call.get("function") or {})
            arguments = function.get("arguments", {})
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, separators=(",", ":"))
            calls.append(
                {
                    "index": index,
                    "id": call.get("id") or f"ollama_call_{uuid4().hex}",
                    "type": "function",
                    "function": {"name": function.get("name"), "arguments": arguments},
                }
            )
        converted_message = {
            "content": message.get("content") or None,
            "thinking": message.get("thinking") or None,
            "tool_calls": calls or None,
            "audio": None,
            "annotations": None,
        }
        choice = {
            "finish_reason": payload.get("done_reason")
            if payload.get("done")
            else None,
            ("delta" if stream else "message"): converted_message,
            "logprobs": None,
        }
        return dotdict(
            {
                "model": payload.get("model"),
                "created_at": payload.get("created_at"),
                "usage": cls._native_usage(payload),
                "choices": [choice],
            }
        )

    def _native_url(self) -> str:
        base_url = self.sampling_params["base_url"].rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        return f"{base_url}/api/chat"

    def _native_headers(self) -> dict[str, str]:
        api_key = getenv("OLLAMA_API_KEY")
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def _native_stream(self, params: dict[str, Any]) -> Iterator[dotdict]:
        with self.client.stream(
            "POST",
            self._native_url(),
            headers=self._native_headers(),
            json=params,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    yield self._native_to_completion(json.loads(line), stream=True)

    async def _anative_stream(self, params: dict[str, Any]):
        async with self.aclient.stream(
            "POST",
            self._native_url(),
            headers=self._native_headers(),
            json=params,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    yield self._native_to_completion(json.loads(line), stream=True)

    def _execute_model(self, **kwargs):
        if self.api_mode == "chat_completions":
            return super()._execute_model(**kwargs)
        self._raise_if_aborted()
        prefilling = kwargs.pop("prefilling", None)
        if prefilling:
            kwargs["messages"] = [
                *kwargs["messages"],
                {"role": "assistant", "content": prefilling},
            ]
        params = self._adapt_native_params({**self.sampling_run_params, **kwargs})
        if params.get("stream"):
            return self._native_stream(params)
        response = self.client.post(
            self._native_url(), headers=self._native_headers(), json=params
        )
        response.raise_for_status()
        self._raise_if_aborted()
        return self._native_to_completion(response.json(), stream=False)

    async def _aexecute_model(self, **kwargs):
        if self.api_mode == "chat_completions":
            return await super()._aexecute_model(**kwargs)
        self._raise_if_aborted()
        prefilling = kwargs.pop("prefilling", None)
        if prefilling:
            kwargs["messages"] = [
                *kwargs["messages"],
                {"role": "assistant", "content": prefilling},
            ]
        params = self._adapt_native_params({**self.sampling_run_params, **kwargs})
        if params.get("stream"):
            return self._anative_stream(params)
        response = await self.aclient.post(
            self._native_url(), headers=self._native_headers(), json=params
        )
        response.raise_for_status()
        self._raise_if_aborted()
        return self._native_to_completion(response.json(), stream=False)

    def warmup_system_prompt(self, *, system_prompt, tool_catalog=None):
        if self.api_mode == "chat_completions":
            return super().warmup_system_prompt(
                system_prompt=system_prompt, tool_catalog=tool_catalog
            )
        params = self._build_generation_params([], system_prompt, None, tool_catalog)
        params["max_tokens"] = self.warmup_max_tokens
        return self._execute_model(**params)

    async def awarmup_system_prompt(self, *, system_prompt, tool_catalog=None):
        if self.api_mode == "chat_completions":
            return await super().awarmup_system_prompt(
                system_prompt=system_prompt, tool_catalog=tool_catalog
            )
        params = self._build_generation_params([], system_prompt, None, tool_catalog)
        params["max_tokens"] = self.warmup_max_tokens
        return await self._aexecute_model(**params)


@register_model
class OllamaTextEmbedder(_BaseOllama, OpenAITextEmbedder):
    """Ollama Text Embedder."""
