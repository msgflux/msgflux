"""Responses apply_patch transport; execution belongs to workspace tools."""

from collections.abc import Mapping

from msgflux.models.tool_adapters.base import ToolTransportAdapter
from msgflux.utils.msgspec import msgspec_dumps


def patch_arguments(item):
    operation = item.get("operation")
    if not isinstance(operation, Mapping):
        raise ValueError("Patch call requires an operation")
    kinds = {"create_file": "create", "update_file": "update", "delete_file": "delete"}
    kind = operation.get("type")
    if not isinstance(kind, str) or kind not in kinds:
        raise ValueError("Unknown patch operation")
    if set(operation) - {"type", "path", "diff"}:
        raise ValueError("Unsupported patch operation fields")
    path, diff = operation.get("path"), operation.get("diff")
    if not isinstance(path, str) or not path or "\0" in path:
        raise ValueError("Patch operation requires a path")
    if kind == "delete_file":
        if diff is not None:
            raise ValueError("Delete operation must not contain a diff")
    elif not isinstance(diff, str):
        raise ValueError("Create and update require a text diff")
    return {"operation": kinds[kind], "path": path, "diff": diff}


class OpenAIApplyPatchAdapter(ToolTransportAdapter):
    provider = "openai"
    api_mode = "responses"
    codec = "openai.responses.apply_patch"
    version = 1
    kind = "apply_patch"
    item_type = "apply_patch_call"
    output_type = "apply_patch_call_output"

    def declaration(self):
        return {"type": "apply_patch"}

    def supports(self, entry):
        schema = (
            getattr(entry, "input_schema", None)
            or getattr(entry, "parameters", None)
            or {}
        )
        return set(schema.get("properties", {})) == {"operation", "path", "diff"}

    def validate_metadata(self, metadata):
        if set(metadata) != {"codec", "version", "name"}:
            raise ValueError("Unsupported patch transport metadata")

    def decode(self, item, name):
        if item.get("status") != "completed":
            raise ValueError("Patch call must be complete before dispatch")
        return patch_arguments(item), {
            "codec": self.codec,
            "version": self.version,
            "name": name,
        }

    def render(self, call_id, result, metadata, *, error=None):  # noqa: ARG002
        if error is None and (
            not isinstance(result, Mapping)
            or result.get("status") not in {"completed", "failed"}
        ):
            raise ValueError("Invalid canonical patch result")
        item = {
            "type": self.output_type,
            "call_id": call_id,
            "status": "failed" if error is not None else result["status"],
        }
        output = error if error is not None else result.get("output")
        if output is not None:
            if not isinstance(output, str):
                raise ValueError("Patch output must be text")
            item["output"] = output
        return item

    def project_history(self, item):
        if item["type"] == self.item_type:
            metadata = item.get("metadata", {}).get("tool_transport", {})
            return {
                "type": "function_call",
                "call_id": item["call_id"],
                "name": metadata.get("name", "apply_patch"),
                "arguments": msgspec_dumps(patch_arguments(item)),
            }
        result = {"status": item["status"]}
        if "output" in item:
            result["output"] = item["output"]
        return {
            "type": "function_call_output",
            "call_id": item["call_id"],
            "output": result,
        }

    def interrupted(self, item, reason):
        return self.render(item["call_id"], None, {}, error=reason)
