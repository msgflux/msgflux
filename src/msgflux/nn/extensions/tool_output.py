"""Opt-in durable offload for text and JSON-compatible tool results."""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import chain
from typing import Iterator

import msgspec

from msgflux.nn.extensions.tool_library import ToolLibraryExtension
from msgflux.nn.hooks import Hook
from msgflux.nn.hooks.events import AfterTool
from msgflux.nn.modules.tool.extensions import ToolContextProvider
from msgflux.runtime.shell_capture import ShellOutputCapture
from msgflux.runtime.tool_results import ToolResultStore
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
