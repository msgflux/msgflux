"""Opt-in durable offload for text and JSON-compatible tool results."""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import chain
from typing import Iterator

import msgspec

from msgflux.nn.extensions.base import AgentExtension
from msgflux.nn.extensions.prompt import _append_section
from msgflux.nn.extensions.tool_library import ToolLibraryExtension
from msgflux.nn.hooks import Hook
from msgflux.nn.hooks.events import AfterTool
from msgflux.nn.modules.tool.extensions import ToolContextProvider
from msgflux.runtime.shell_capture import ShellOutputCapture
from msgflux.runtime.tool_results import (
    ToolOutputOffloadConfig,
    ToolResultStore,
    get_tool_result_reference,
)
from msgflux.tools.shell import ShellResult

_TEXT_CHARS = 8192
_MAX_DEPTH = 64


class _ShellCaptureProvider(ToolContextProvider):
    def __init__(self, capture):
        super().__init__("context_shell_capture", sources=("shell_capture",))
        self.capture = capture

    async def resolve(self, _request):
        return self.capture


def _string_chunks(value: str, *, json_string: bool) -> Iterator[bytes]:
    if json_string:
        yield b'"'
    for index in range(0, len(value), _TEXT_CHARS):
        part = value[index : index + _TEXT_CHARS]
        yield msgspec.json.encode(part)[1:-1] if json_string else part.encode("utf-8")
    if json_string:
        yield b'"'


def _json_chunks(  # noqa: C901 - one bounded encoder for JSON primitives
    value, ancestors: set[int], depth: int = 0
) -> Iterator[bytes]:
    """Encode JSON primitives without allocating a full serialized large string."""
    if depth > _MAX_DEPTH:
        raise ValueError("Tool result exceeds JSON nesting limit")
    if isinstance(value, str):
        yield from _string_chunks(value, json_string=True)
    elif value is None or type(value) in (int, bool, float):
        if type(value) is float and not math.isfinite(value):
            raise ValueError("Nonfinite numbers are not supported in tool output JSON")
        yield msgspec.json.encode(value)
    elif isinstance(value, (dict, list)):
        identity = id(value)
        if identity in ancestors:
            raise ValueError("Cyclic tool result cannot be encoded as JSON")
        ancestors.add(identity)
        try:
            is_object = isinstance(value, dict)
            yield b"{" if is_object else b"["
            for index, entry in enumerate(value.items() if is_object else value):
                if index:
                    yield b","
                if is_object:
                    key, item = entry
                    if not isinstance(key, str):
                        raise TypeError("Tool output JSON object keys must be strings")
                    yield from _string_chunks(key, json_string=True)
                    yield b":"
                else:
                    item = entry
                yield from _json_chunks(item, ancestors, depth + 1)
            yield b"}" if is_object else b"]"
        finally:
            ancestors.remove(identity)
    else:
        raise TypeError("Tool output contains a value that is not a JSON primitive")


class ToolOutputOffloadExtension(ToolLibraryExtension):
    """Replace large supported results with a JSON descriptor and text preview.

    Builtin Bash uses optional incremental capture when registered. Other tools
    are transformed after return: their allocations are not bounded here. This
    does not authorize reads or transform other binary/provider-native types.
    """

    def __init__(
        self,
        store: ToolResultStore,
        *,
        max_inline_bytes: int = 32 * 1024,
        preview_bytes: int = 2048,
        max_capture_bytes: int = 1_000_000,
    ) -> None:
        super().__init__("tool_output_offload")
        if not isinstance(store, ToolResultStore):
            raise TypeError("store must implement ToolResultStore")
        if type(max_inline_bytes) is not int or max_inline_bytes <= 0:
            raise ValueError("max_inline_bytes must be a positive integer")
        if type(preview_bytes) is not int or not 0 <= preview_bytes <= max_inline_bytes:
            raise ValueError("preview_bytes must be between zero and max_inline_bytes")
        self.store = store
        self.max_inline_bytes = max_inline_bytes
        self.preview_bytes = preview_bytes
        if type(max_capture_bytes) is not int or max_capture_bytes <= 0:
            raise ValueError("max_capture_bytes must be a positive integer")
        self.max_capture_bytes = max_capture_bytes
        self._capture_handle = None

    def on_register(self, library):
        self._capture_handle = library.runtime_extensions.register(
            _ShellCaptureProvider(
                ShellOutputCapture(
                    self.store,
                    max_inline_bytes=self.max_inline_bytes,
                    preview_bytes=self.preview_bytes,
                    max_capture_bytes=self.max_capture_bytes,
                )
            )
        )

    def on_remove(self, _library):
        if self._capture_handle is not None:
            self._capture_handle.remove()
            self._capture_handle = None

    def hooks(self):
        return (Hook(event="transform_tool_output", handler=self._transform),)

    def _transform(self, outcome: AfterTool) -> AfterTool:
        if outcome.error is not None:
            return outcome
        if isinstance(outcome.result, ShellResult):
            return self._transform_shell(outcome)
        if not isinstance(outcome.result, (str, dict, list)):
            return outcome
        offloaded = self._offload(outcome.result)
        if offloaded is None:
            return outcome
        reference, preview = offloaded
        return replace(
            outcome,
            result={
                "type": "tool_result_reference",
                "reference": reference.to_dict(),
                "preview": preview,
                "truncated": True,
            },
        )

    def _transform_shell(self, outcome: AfterTool) -> AfterTool:
        result = outcome.result
        if result.output_reference is not None:
            return outcome
        # Shallow containers reuse the original strings; the encoder never
        # materializes a second complete serialized copy of stdout/stderr.
        value = {
            "results": [
                {
                    "status": part.status,
                    "stdout": part.stdout,
                    "stderr": part.stderr,
                    "returncode": part.returncode,
                }
                for part in result.results
            ]
        }
        offloaded = self._offload(value)
        if offloaded is None:
            return outcome
        reference, _ = offloaded
        per_field = self.preview_bytes // (2 * len(result.results))

        def preview(text):
            return (
                text[:per_field]
                .encode("utf-8")[:per_field]
                .decode("utf-8", errors="ignore")
            )

        return replace(
            outcome,
            result=ShellResult(
                results=tuple(
                    msgspec.structs.replace(
                        part, stdout=preview(part.stdout), stderr=preview(part.stderr)
                    )
                    for part in result.results
                ),
                output_reference=reference,
            ),
        )

    def _offload(self, value):
        text = isinstance(value, str)
        chunks = iter(
            _string_chunks(value, json_string=False)
            if text
            else _json_chunks(value, set())
        )
        prefix = bytearray()
        try:
            for chunk in chunks:
                if len(prefix) + len(chunk) > self.max_inline_bytes:
                    preview = bytes(prefix[: self.preview_bytes])
                    preview += chunk[: max(0, self.preview_bytes - len(preview))]
                    reference = self.store.put(
                        chain((bytes(prefix), chunk), chunks),
                        media_type="text/plain; charset=utf-8"
                        if text
                        else "application/json",
                    )
                    return reference, preview.decode("utf-8", errors="ignore")
                prefix.extend(chunk)
        finally:
            chunks.close()
        return None


class _ManagedToolResultStore(ToolResultStore):
    """Resolve a store at use time, including background invocation contexts."""

    def __init__(self, config):
        self.config = config

    def _resolve(self):
        from msgflux.nn.modules.agent.resources import (  # noqa: PLC0415
            _get_tool_result_store,
        )

        return _get_tool_result_store(config=self.config)

    def put(self, chunks, *, media_type="application/octet-stream"):
        return self._resolve().put(chunks, media_type=media_type)

    def get(self, result_id):
        return self._resolve().get(result_id)

    def iter_bytes(self, reference, *, offset=0, limit=None, chunk_size=65536):
        yield from self._resolve().iter_bytes(
            reference, offset=offset, limit=limit, chunk_size=chunk_size
        )


class _ManagedToolOutputOffloadExtension(ToolOutputOffloadExtension):
    def __init__(self, config):
        super().__init__(
            _ManagedToolResultStore(config),
            max_inline_bytes=config.max_inline_bytes,
            preview_bytes=config.preview_bytes,
            max_capture_bytes=config.max_capture_bytes,
        )

    def _transform(self, outcome):
        # Preserve existing references (including background task retrieval).
        # Read bounds its excerpts; offloading them again prevents inspection.
        if outcome.tool_name == "read":
            return outcome
        if (
            isinstance(outcome.result, dict)
            and outcome.result.get("type") == "tool_result_reference"
            and get_tool_result_reference(outcome.result) is not None
        ):
            return outcome
        transformed = super()._transform(outcome)
        if transformed is not outcome and isinstance(transformed.result, dict):
            reference = transformed.result["reference"]
            store = self.store._resolve()
            return replace(
                transformed,
                result={
                    **transformed.result,
                    "path": str(store.root / reference["result_id"] / "content"),
                },
            )
        return transformed


class ManagedToolOutputOffloadExtension(AgentExtension):
    """Offload large outputs into the active managed Agent thread directory.

    Register through Agent.extensions. The extension owns its library hooks and
    shell capture; stores belong to managed resources, never the extension.
    """

    def __init__(self, config: ToolOutputOffloadConfig | None = None):
        super().__init__("tool_output_offload")
        if config is not None and not isinstance(config, ToolOutputOffloadConfig):
            raise TypeError("config must be ToolOutputOffloadConfig or None")
        self.config = config or ToolOutputOffloadConfig()
        self._library_handle = None
        self._inherited = False

    def on_register(self, agent):
        if agent.tool_library.has_extension("tool_output_offload"):
            raise ValueError(
                "Managed offload conflicts with explicit tool output storage"
            )
        self._library_handle = agent.tool_library.register_extension(
            "tool_output_offload", _ManagedToolOutputOffloadExtension(self.config)
        )

    def on_remove(self, _agent):
        if self._library_handle is not None:
            self._library_handle.remove()
            self._library_handle = None

    def hooks(self):
        return (Hook(event="transform_system_prompt", handler=self._guidance),)

    def _guidance(self, ctx):
        from msgflux.nn.modules.agent.resources import (  # noqa: PLC0415
            _CURRENT_RESOURCES,
        )
        from msgflux.runtime.context import get_execution_scope  # noqa: PLC0415

        owned = _CURRENT_RESOURCES.get()
        if owned is None or get_execution_scope().workspace is None:
            return ctx
        guidance = {
            "agent_dir": str(owned.resources.thread_dir.parent.parent),
            "thread_id": owned.resources.thread_id,
            "tool_results": str(owned.resources.thread_dir / "tool-results"),
            "instructions": (
                "Large tool outputs are saved as files. Use read with the returned "
                "path and offset/limit to retrieve more lines. For a shell "
                "output_reference, read <tool_results>/<result_id>/content. "
                "Previews are excerpts; do not invent result IDs. These paths "
                "are read-only artifacts, not project files."
            ),
        }
        return _append_section(
            ctx,
            "<tool_output_context>\n"
            + msgspec.json.encode(guidance).decode().replace("<", "\\u003c")
            + "\n</tool_output_context>",
        )


def _bind_managed_offload(agent, owned, *, inherit=False):
    configured = [
        extension
        for extension in agent.extensions.values()
        if isinstance(extension, ManagedToolOutputOffloadExtension)
    ]
    if len(configured) > 1:
        raise ValueError("Only one managed tool output offload extension is allowed")
    extension = configured[0] if configured else None
    if (
        extension is not None
        and extension._inherited
        and (not inherit or owned is None or owned.resources.offload_config is None)
    ):
        agent.remove_extension(extension.name)
        extension = None
    if owned is None:
        if extension is not None:
            raise ValueError(
                "Managed tool offload requires agent_dir or inherited Agent resources"
            )
        return
    with owned.lock, agent._resource_lock:
        inherited = owned.resources.offload_config
        config = (
            extension.config
            if extension is not None
            else inherited
            if inherit
            else None
        )
        if config is None:
            if not inherit:
                owned.resources.offload_config = None
            return
        if inherited is not None and inherited != config:
            raise ValueError(
                "Nested Agent cannot replace tool output offload configuration"
            )
        if extension is None:
            extension = ManagedToolOutputOffloadExtension(config)
            extension._inherited = True
            agent.register_extension("tool_output_offload", extension)
        owned.resources.offload_config = config
