"""Restore provider continuation codecs without restoring execution authority."""

from collections.abc import Mapping

from msgflux.models.tool_adapters.base import ToolTransportAdapter
from msgflux.models.tool_adapters.openai_patch import OpenAIApplyPatchAdapter
from msgflux.models.tool_adapters.openai_shell import OpenAIShellAdapter

_ADAPTERS: tuple[ToolTransportAdapter, ...] = (
    OpenAIShellAdapter(),
    OpenAIApplyPatchAdapter(),
)


def native_item_types(*, output=False):
    return {
        adapter.output_type if output else adapter.item_type for adapter in _ADAPTERS
    }


def pending_tool_calls(items) -> dict[str, Mapping]:
    """Return call IDs awaiting an output mapped to their latest call item."""
    call_types = {"function_call", *native_item_types()}
    output_types = {"function_call_output", *native_item_types(output=True)}
    pending: dict[str, Mapping] = {}
    for item in items:
        if not isinstance(item, Mapping):
            continue
        call_id = item.get("call_id") or item.get("id")
        if not isinstance(call_id, str) or not call_id:
            continue
        if item.get("type") in call_types:
            pending[call_id] = item
        elif item.get("type") in output_types:
            pending.pop(call_id, None)
    return pending


def history_adapter(item) -> ToolTransportAdapter | None:
    for adapter in _ADAPTERS:
        if item.get("type") in {adapter.item_type, adapter.output_type}:
            metadata = item.get("metadata", {}).get("tool_transport")
            if (
                metadata is not None
                and transport_adapter(metadata).codec != adapter.codec
            ):
                raise ValueError("History item does not match tool transport")
            return adapter
    return None


def transport_adapter(metadata) -> ToolTransportAdapter:
    # Deliberately closed registry: checkpoints cannot name importable code.
    if not isinstance(metadata, Mapping):
        raise ValueError("Tool transport metadata must be a mapping")
    adapter = next(
        (adapter for adapter in _ADAPTERS if metadata.get("codec") == adapter.codec),
        None,
    )
    if (
        adapter is None
        or type(metadata.get("version")) is not int
        or metadata["version"] != adapter.version
    ):
        raise ValueError("Unknown tool transport codec or version")
    if not isinstance(metadata.get("name"), str) or not metadata["name"]:
        raise ValueError("Tool transport requires a logical name")
    adapter.validate_metadata(metadata)
    return adapter


def validate_native_calls(native_calls, intents):
    if not isinstance(native_calls, Mapping):
        raise ValueError("Native calls must be a mapping")
    by_id = {intent["id"]: intent for intent in intents}
    for call_id, metadata in native_calls.items():
        transport_adapter(metadata)
        if call_id not in by_id or by_id[call_id]["name"] != metadata["name"]:
            raise ValueError("Tool transport does not match the pending intent")


def render_native_output(call_id, result, metadata, *, error=None):
    return transport_adapter(metadata).render(call_id, result, metadata, error=error)
