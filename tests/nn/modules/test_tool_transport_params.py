"""Unit tests for transport_params dict-lowering in LocalTool."""

import pytest
from typing import Optional, Union
from unittest.mock import MagicMock

from msgflux.nn.modules.tool import LocalTool, _convert_module_to_nn_tool
from msgflux.tools.config import tool_config


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

    def test_no_transport_params_returns_kwargs_unchanged(self):
        def fn(x: str) -> str:
            """fn"""
            return x

        tool = _make_local_tool(fn, transport_params=None)
        kwargs = {"x": "hello", "tool_call_id": "abc"}
        assert tool._restore_transport_params(kwargs) is kwargs

    def test_empty_transport_params_returns_kwargs_unchanged(self):
        def fn(x: str) -> str:
            """fn"""
            return x

        tool = _make_local_tool(fn, transport_params={})
        kwargs = {"x": "hello"}
        result = tool._restore_transport_params(kwargs)
        assert result == {"x": "hello"}

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

    def test_param_without_entries_key_is_not_touched(self):
        def fn(mapping: dict[str, str]) -> str:
            """fn"""
            return ""

        tool = _make_local_tool(fn, transport_params={"mapping": dict[str, str]})
        wire = {"mapping": {"not_entries": []}}
        restored = tool._restore_transport_params(wire)
        assert restored["mapping"] == {"not_entries": []}

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


# ── _convert_module_to_nn_tool detects transport params ──────────────────────

class TestConvertDetectsTransportParams:

    def test_dict_param_detected(self):
        def fn(updates: dict[str, str]) -> str:
            """A tool with a dict param."""
            return ""

        local = _convert_module_to_nn_tool(fn)
        assert "updates" in local._buffers["transport_params"]

    def test_non_dict_param_not_detected(self):
        def fn(name: str, count: int) -> str:
            """A tool without dict params."""
            return ""

        local = _convert_module_to_nn_tool(fn)
        assert local._buffers["transport_params"] == {}

    def test_return_annotation_not_in_transport_params(self):
        def fn(updates: dict[str, str]) -> dict:
            """A tool."""
            return {}

        local = _convert_module_to_nn_tool(fn)
        assert "return" not in local._buffers["transport_params"]

    def test_mixed_params_only_dict_detected(self):
        def fn(updates: dict[str, str], name: str, count: int) -> str:
            """A tool with mixed params."""
            return ""

        local = _convert_module_to_nn_tool(fn)
        tp = local._buffers["transport_params"]
        assert "updates" in tp
        assert "name" not in tp
        assert "count" not in tp


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
