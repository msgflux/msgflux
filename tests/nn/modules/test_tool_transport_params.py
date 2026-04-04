"""Unit tests for transport-lowered tool params in LocalTool."""

import pytest
from typing import Optional, Union

from msgflux.nn.modules.tool import LocalTool, ToolLibrary, _convert_module_to_nn_tool


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_local_tool(fn, transport_params=None):
    """Wrap a plain function as LocalTool without going through the agent machinery."""
    return LocalTool(
        name=fn.__name__,
        description=fn.__doc__ or fn.__name__,
        annotations=fn.__annotations__,
        tool_config={},
        impl=fn,
        transport_params=transport_params,
    )


# ── _restore_transport_params ─────────────────────────────────────────────────

class TestRestoreTransportParams:

    def test_scalar_params_are_left_unchanged(self):
        def fn(x: str) -> str:
            """fn"""
            return x

        tool = _make_local_tool(fn, transport_params=None)
        kwargs = {"x": "hello", "tool_call_id": "abc"}
        result = tool._restore_transport_params(kwargs)
        assert result == {"x": "hello", "tool_call_id": "abc"}

    def test_entries_are_converted_to_dict(self):
        def fn(mapping: dict[str, str]) -> str:
            """fn"""
            return ""

        tool = _make_local_tool(fn, transport_params={"mapping": dict[str, str]})
        wire = {
            "mapping": {
                "entries": [
                    {"key": "a", "value": "1"},
                    {"key": "b", "value": "2"},
                ]
            }
        }
        restored = tool._restore_transport_params(wire)
        assert restored["mapping"] == {"a": "1", "b": "2"}

    def test_non_transport_params_are_untouched(self):
        def fn(mapping: dict[str, str], name: str) -> str:
            """fn"""
            return ""

        tool = _make_local_tool(fn, transport_params={"mapping": dict[str, str]})
        wire = {
            "mapping": {"entries": [{"key": "x", "value": "y"}]},
            "name": "Alice",
        }
        restored = tool._restore_transport_params(wire)
        assert restored["name"] == "Alice"
        assert restored["mapping"] == {"x": "y"}

    def test_missing_param_in_kwargs_is_ignored(self):
        def fn(mapping: dict[str, str]) -> str:
            """fn"""
            return ""

        tool = _make_local_tool(fn, transport_params={"mapping": dict[str, str]})
        wire = {"other": "value"}
        restored = tool._restore_transport_params(wire)
        assert restored == {"other": "value"}

    def test_plain_dict_is_restored_recursively_without_entries_wrapper(self):
        def fn(mapping: dict[str, dict[str, str]]) -> str:
            """fn"""
            return ""

        tool = _make_local_tool(
            fn,
            transport_params={"mapping": dict[str, dict[str, str]]},
        )
        wire = {
            "mapping": {
                "profile": {
                    "entries": [{"key": "city", "value": "Austin"}],
                }
            }
        }
        restored = tool._restore_transport_params(wire)
        assert restored["mapping"] == {"profile": {"city": "Austin"}}

    def test_empty_entries_list_produces_empty_dict(self):
        def fn(mapping: dict[str, str]) -> str:
            """fn"""
            return ""

        tool = _make_local_tool(fn, transport_params={"mapping": dict[str, str]})
        wire = {"mapping": {"entries": []}}
        restored = tool._restore_transport_params(wire)
        assert restored["mapping"] == {}

    def test_list_value_type_preserved(self):
        def fn(mapping: dict[str, list[str]]) -> str:
            """fn"""
            return ""

        tool = _make_local_tool(fn, transport_params={"mapping": dict[str, list[str]]})
        wire = {
            "mapping": {
                "entries": [
                    {"key": "participants", "value": ["Alice", "Bob"]},
                ]
            }
        }
        restored = tool._restore_transport_params(wire)
        assert restored["mapping"] == {"participants": ["Alice", "Bob"]}

    def test_int_keys_are_restored_from_entries(self):
        def fn(labels: dict[int, str]) -> str:
            """fn"""
            return ""

        tool = _make_local_tool(fn, transport_params={"labels": dict[int, str]})
        wire = {
            "labels": {
                "entries": [
                    {"key": 1, "value": "one"},
                    {"key": 2, "value": "two"},
                ]
            }
        }
        restored = tool._restore_transport_params(wire)
        assert restored["labels"] == {1: "one", 2: "two"}


# ── annotations drive restoration ────────────────────────────────────────────

class TestToolAnnotations:

    def test_convert_module_preserves_dict_annotation(self):
        def fn(updates: dict[str, str]) -> str:
            """A tool with a dict param."""
            return ""

        local = _convert_module_to_nn_tool(fn)
        assert local.get_module_annotations()["updates"] == dict[str, str]

    def test_tool_library_exposes_annotations_by_tool_name(self):
        def fn(updates: dict[str, str], name: str, count: int) -> str:
            """A tool with mixed params."""
            return ""

        library = ToolLibrary("test", [fn])
        annotations = library.get_tool_annotations()
        assert annotations["fn"]["updates"] == dict[str, str]
        assert annotations["fn"]["name"] is str
        assert annotations["fn"]["count"] is int


# ── forward calls impl with restored dict ────────────────────────────────────

class TestForwardRestores:

    def test_forward_passes_restored_dict_to_impl(self):
        received = {}

        def fn(mapping: dict[str, str], **kwargs) -> str:
            """fn"""
            received.update(mapping)
            return "ok"

        local = _convert_module_to_nn_tool(fn)
        wire_mapping = {"entries": [{"key": "city", "value": "Austin"}]}
        local(mapping=wire_mapping)

        assert received == {"city": "Austin"}

    def test_forward_without_transport_params_unchanged(self):
        received = {}

        def fn(name: str) -> str:
            """fn"""
            received["name"] = name
            return "ok"

        local = _convert_module_to_nn_tool(fn)
        local(name="Alice")

        assert received["name"] == "Alice"


# ── JSON schema shape ─────────────────────────────────────────────────────────

class TestDictToolSchema:

    def test_schema_uses_entries_shape(self):
        def fn(updates: dict[str, str]) -> str:
            """A tool."""
            return ""

        local = _convert_module_to_nn_tool(fn)
        schema = local.get_json_schema()
        params = schema["function"]["parameters"]
        prop = params["properties"]["updates"]
        assert prop["type"] == "object"
        assert "entries" in prop["properties"]

    def test_schema_entry_has_key_and_value(self):
        def fn(updates: dict[str, str]) -> str:
            """A tool."""
            return ""

        local = _convert_module_to_nn_tool(fn)
        schema = local.get_json_schema()
        entry = (
            schema["function"]["parameters"]["properties"]["updates"]
            ["properties"]["entries"]["items"]
        )
        assert "key" in entry["properties"]
        assert "value" in entry["properties"]
        assert entry["additionalProperties"] is False

    def test_schema_strict_true(self):
        def fn(updates: dict[str, str]) -> str:
            """A tool."""
            return ""

        local = _convert_module_to_nn_tool(fn)
        schema = local.get_json_schema()
        assert schema["function"]["strict"] is True
