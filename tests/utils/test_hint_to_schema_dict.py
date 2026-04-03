"""Unit tests for dict[K, V] lowering in hint_to_schema."""

import pytest
from typing import Any, Dict, List, Optional, Union

from msgflux.utils.chat import hint_to_schema


class TestDictLowering:
    """hint_to_schema converts dict[K, V] to {entries: [{key, value}]} wire format."""

    def test_dict_str_str(self):
        schema = hint_to_schema(dict[str, str])
        assert schema == {
            "type": "object",
            "properties": {
                "entries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string"},
                            "value": {"type": "string"},
                        },
                        "required": ["key", "value"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["entries"],
            "additionalProperties": False,
        }

    def test_dict_str_int(self):
        schema = hint_to_schema(dict[str, int])
        value_schema = schema["properties"]["entries"]["items"]["properties"]["value"]
        assert value_schema == {"type": "integer"}

    def test_dict_str_list_str(self):
        schema = hint_to_schema(dict[str, list[str]])
        value_schema = schema["properties"]["entries"]["items"]["properties"]["value"]
        assert value_schema == {"type": "array", "items": {"type": "string"}}

    def test_dict_str_optional_str(self):
        schema = hint_to_schema(dict[str, Optional[str]])
        value_schema = schema["properties"]["entries"]["items"]["properties"]["value"]
        assert value_schema == {"anyOf": [{"type": "string"}, {"type": "null"}]}

    def test_dict_str_union(self):
        schema = hint_to_schema(dict[str, Union[str, list[str], None]])
        value_schema = schema["properties"]["entries"]["items"]["properties"]["value"]
        assert value_schema == {
            "anyOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}},
                {"type": "null"},
            ]
        }

    def test_dict_str_any_value_schema_is_empty(self):
        # Any as value → {} (no constraint) instead of TypeError
        schema = hint_to_schema(dict[str, Any])
        value_schema = schema["properties"]["entries"]["items"]["properties"]["value"]
        assert value_schema == {}

    def test_bare_dict_raises(self):
        with pytest.raises(TypeError, match="bare `dict`"):
            hint_to_schema(dict)

    def test_bare_Dict_raises(self):
        with pytest.raises(TypeError, match="bare `dict`"):
            hint_to_schema(Dict)

    def test_dict_without_args_raises(self):
        # Dict without args at runtime behaves like bare dict
        with pytest.raises(TypeError):
            hint_to_schema(Dict)

    def test_nested_dict_in_list(self):
        # list[dict[str, int]] — the list item is a dict, each item gets lowered
        schema = hint_to_schema(list[dict[str, int]])
        assert schema["type"] == "array"
        item = schema["items"]
        assert item["type"] == "object"
        assert "entries" in item["properties"]

    def test_entries_required(self):
        schema = hint_to_schema(dict[str, str])
        assert "entries" in schema["required"]

    def test_entry_key_and_value_required(self):
        schema = hint_to_schema(dict[str, str])
        entry = schema["properties"]["entries"]["items"]
        assert "key" in entry["required"]
        assert "value" in entry["required"]

    def test_additionalProperties_false_at_root(self):
        schema = hint_to_schema(dict[str, str])
        assert schema["additionalProperties"] is False

    def test_additionalProperties_false_in_entry(self):
        schema = hint_to_schema(dict[str, str])
        entry = schema["properties"]["entries"]["items"]
        assert entry["additionalProperties"] is False
