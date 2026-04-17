import os
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

import msgspec
import pytest

from msgflux.utils.msgspec import (
    StructFactory,
    export_to_json,
    is_optional_field,
    lower_msgspec_struct_for_openai,
    load,
    msgspec_dumps,
    read_json,
    restore_openai_structured_output,
    restore_transport_value,
    save,
    struct_to_dict,
)


class MyStruct(msgspec.Struct):
    a: int
    b: str
    c: Optional[int] = None


class DictOutput(msgspec.Struct):
    entities: List[Dict[str, str]]
    metadata: Optional[Dict[str, int]] = None


class PlainOutput(msgspec.Struct):
    title: str
    score: int


class AnyOutput(msgspec.Struct):
    payload: Any


class UnionOutput(msgspec.Struct):
    payload: Union[str, int]


class SetOutput(msgspec.Struct):
    tags: Set[str]


class BareDictOutput(msgspec.Struct):
    payload: dict


class Label(Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"


JSON_SCHEMA = {
    "$defs": {
        "MyStruct": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "string"}},
            "required": ["a", "b"],
        }
    },
    "$ref": "#/definitions/MyStruct",
}


class TestStructFactory:
    def test_from_json_schema(self):
        struct = StructFactory.from_json_schema(JSON_SCHEMA)
        assert issubclass(struct, msgspec.Struct)
        instance = struct(a=1, b="2")
        assert instance.a == 1

    def test_from_signature(self):
        struct = StructFactory.from_signature("a: int, b: str")
        assert issubclass(struct, msgspec.Struct)
        instance = struct(a=1, b="2")
        assert instance.a == 1


def test_msgspec_dumps():
    instance = MyStruct(a=1, b="2")
    assert msgspec_dumps(instance) == '{"a":1,"b":"2","c":null}'


@pytest.fixture
def temp_file(tmp_path):
    return tmp_path / "test.json"


def test_save_and_load(temp_file):
    data = {"a": 1, "b": "2"}
    save(data, str(temp_file))
    loaded_data = load(str(temp_file))
    assert data == loaded_data


def test_export_and_read_json(temp_file):
    data = {"a": 1, "b": "2"}
    export_to_json(data, str(temp_file))
    loaded_data = read_json(str(temp_file))
    assert data == loaded_data


def test_struct_to_dict():
    instance = MyStruct(a=1, b="2")
    d = struct_to_dict(instance)
    assert d == {"a": 1, "b": "2", "c": None}


def test_is_optional_field():
    assert not is_optional_field(MyStruct, "a")
    assert is_optional_field(MyStruct, "c")


def test_lower_msgspec_struct_for_openai_preserves_plain_structs():
    lowered = lower_msgspec_struct_for_openai(PlainOutput)
    assert lowered is PlainOutput


def test_lower_msgspec_struct_for_openai_lowers_dict_fields():
    lowered = lower_msgspec_struct_for_openai(DictOutput)

    assert lowered is not DictOutput

    schema = msgspec.json.schema(lowered)
    entities_items = schema["$defs"][lowered.__name__]["properties"]["entities"][
        "items"
    ]
    metadata_anyof = schema["$defs"][lowered.__name__]["properties"]["metadata"][
        "anyOf"
    ]
    metadata_value = next(item for item in metadata_anyof if "$ref" in item)

    assert entities_items["$ref"].endswith("Map")
    assert metadata_value["$ref"].endswith("Map")


def test_restore_openai_structured_output_restores_dict_fields():
    transport_output = {
        "entities": [
            {
                "entries": [
                    {"key": "name", "value": "Apple"},
                    {"key": "type", "value": "Organization"},
                ]
            },
            {
                "entries": [
                    {"key": "name", "value": "Tim Cook"},
                    {"key": "type", "value": "Person"},
                ]
            },
        ],
        "metadata": {"entries": [{"key": "count", "value": 2}]},
    }

    restored = restore_openai_structured_output(transport_output, DictOutput)

    assert restored == {
        "entities": [
            {"name": "Apple", "type": "Organization"},
            {"name": "Tim Cook", "type": "Person"},
        ],
        "metadata": {"count": 2},
    }


def test_restore_openai_structured_output_rejects_invalid_mapping_wrapper():
    with pytest.raises(ValueError, match="required `entries` field"):
        restore_openai_structured_output({"metadata": {}}, Dict[str, str])


def test_restore_openai_structured_output_rejects_non_optional_union():
    with pytest.raises(TypeError, match="Only Optional\\[T\\] unions are supported"):
        restore_openai_structured_output("value", Union[str, int])


def test_restore_transport_value_accepts_plain_mapping_when_not_strict():
    restored = restore_transport_value(
        {"profile": {"entries": [{"key": "city", "value": "Austin"}]}},
        Dict[str, Dict[str, str]],
    )
    assert restored == {"profile": {"city": "Austin"}}


def test_restore_transport_value_restores_enum_keys():
    restored = restore_transport_value(
        {"entries": [{"key": "primary", "value": "Alice"}]},
        Dict[Label, str],
    )
    assert restored == {Label.PRIMARY: "Alice"}


@pytest.mark.parametrize(
    ("schema_type", "path", "dtype"),
    [
        (AnyOutput, "payload", "Any"),
        (UnionOutput, "payload", "Union[str, int]"),
        (SetOutput, "tags", "Set[str]"),
        (BareDictOutput, "payload", "dict"),
    ],
)
def test_lower_msgspec_struct_for_openai_rejects_unsupported_types(
    schema_type, path, dtype
):
    with pytest.raises(TypeError, match=rf"`{path}`.*`{re.escape(dtype)}`"):
        lower_msgspec_struct_for_openai(schema_type)
