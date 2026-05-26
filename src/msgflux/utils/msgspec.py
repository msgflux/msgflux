import ast
import os
import re
from collections import OrderedDict
from collections.abc import Mapping as ABCMapping
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
)

import msgspec
from msgspec import Meta, Struct, defstruct
from typing_extensions import Annotated

from msgflux.core.dotdict import dotdict
from msgflux.logger import logger
from msgflux.utils.common import type_mapping


class StructFactory:
    """Rebuild msgspec.Struct from a JSON-schema or str signature."""

    def __init__(self):
        self.reconstructed_classes = OrderedDict()

    @classmethod
    def from_json_schema(cls, json_schema: Dict[str, Any]) -> msgspec.Struct:
        self = cls()  # Temp instance
        if "$defs" not in json_schema:
            raise ValueError("json_schema must contain definitions in `$defs`")

        definitions = json_schema["$defs"]

        dependency_order = self._get_dependency_order(definitions)

        for class_name in dependency_order:
            class_def = definitions[class_name]
            self._reconstruct_class(class_name, class_def, definitions)

        main_struct = self.reconstructed_classes.popitem(last=True)[1]
        return main_struct

    def _get_dependency_order(self, definitions: Dict[str, Any]) -> List[str]:
        dependencies = {}
        for class_name, class_def in definitions.items():
            deps = set()
            self._find_dependencies(class_def, deps, definitions.keys())
            dependencies[class_name] = deps

        ordered = []
        remaining = set(definitions.keys())

        while remaining:
            ready = [
                name for name in remaining if dependencies[name].issubset(set(ordered))
            ]
            if not ready:  # Circular dependency - uses alphabetical order as fallback
                ready = [min(remaining)]
            for name in ready:
                ordered.append(name)
                remaining.remove(name)

        return ordered

    def _find_dependencies(
        self, definition: Dict[str, Any], deps: Set[str], available_classes: Set[str]
    ):
        if isinstance(definition, dict):
            if "$ref" in definition:
                ref_name = definition["$ref"].split("/")[-1]
                if ref_name in available_classes:
                    deps.add(ref_name)

            if "properties" in definition:
                for prop_def in definition["properties"].values():
                    self._find_dependencies(prop_def, deps, available_classes)

            if "anyOf" in definition:
                for item in definition["anyOf"]:
                    self._find_dependencies(item, deps, available_classes)

            if "items" in definition:
                self._find_dependencies(definition["items"], deps, available_classes)

    def _reconstruct_class(
        self,
        class_name: str,
        class_def: Dict[str, Any],
        all_definitions: Dict[str, Any],
    ):
        if class_name in self.reconstructed_classes:
            return self.reconstructed_classes[class_name]

        properties = class_def.get("properties", {})
        required_fields = set(class_def.get("required", []))

        fields = []

        for field_name, field_def in properties.items():
            field_type = self._resolve_field_type(field_def, all_definitions)

            if field_name in required_fields:
                fields.append((field_name, field_type))
            else:
                fields.append((field_name, field_type, None))

        reconstructed_class = defstruct(
            class_name,
            fields,
            kw_only=True,
            module=__name__,
        )

        self.reconstructed_classes[class_name] = reconstructed_class
        return reconstructed_class

    def _resolve_field_type(
        self, field_def: Dict[str, Any], all_definitions: Dict[str, Any]
    ):
        if "anyOf" in field_def:
            types = []
            description = None

            for option in field_def["anyOf"]:
                if option.get("type") == "null":
                    continue

                if "description" in option:
                    description = option["description"]

                option_type = self._resolve_single_type(option, all_definitions)
                if option_type and option_type is not type(None):
                    types.append(option_type)

            if len(types) == 1:
                base_type = types[0]
                if description:
                    annotated_type = Annotated[base_type, Meta(description=description)]
                    return Optional[annotated_type]
                else:
                    return Optional[base_type]
            elif len(types) > 1:
                if description:
                    union_type = Union[tuple(types)]
                    annotated_type = Annotated[
                        union_type, Meta(description=description)
                    ]
                    return Optional[annotated_type]
                else:
                    return Optional[Union[tuple(types)]]
            else:
                return Optional[str]

        return self._resolve_single_type(field_def, all_definitions)

    def _resolve_single_type(
        self, type_def: Dict[str, Any], all_definitions: Dict[str, Any]
    ):
        if "$ref" in type_def:
            ref_name = type_def["$ref"].split("/")[-1]
            if ref_name in self.reconstructed_classes:
                return self.reconstructed_classes[ref_name]
            else:
                return ref_name

        if "enum" in type_def:
            enum_values = type_def["enum"]
            description = type_def.get("description")

            if len(enum_values) > 0:
                if len(enum_values) == 1:
                    literal_type = Literal[enum_values[0]]
                else:
                    literal_type = Literal[tuple(enum_values)]

                if description:
                    return Annotated[literal_type, Meta(description=description)]
                else:
                    return literal_type

        if "type" in type_def:
            type_name = type_def["type"]

            if type_name == "array":
                if "items" in type_def:
                    item_type = self._resolve_single_type(
                        type_def["items"], all_definitions
                    )
                    return List[item_type]
                else:
                    return List[Any]

            return type_mapping.get(type_name, str)

        return str

    @classmethod
    def from_signature(
        cls,
        signature: str,
        struct_name: Optional[str] = "DynamicStruct",
        field_descriptions: Optional[Dict[str, str]] = None,
    ) -> type:
        annotations = cls._parse_annotations(signature)
        struct_fields = []

        for info in annotations:
            try:
                parsed_type = cls._parse_type_string(info.dtype)

                if field_descriptions and info.name in field_descriptions:
                    annotated_type = Annotated[
                        parsed_type, Meta(description=field_descriptions[info.name])
                    ]
                    struct_fields.append((info.name, annotated_type))
                else:
                    struct_fields.append((info.name, parsed_type))

            except ValueError as e:
                raise ValueError(
                    f"Error parsing field `{info.name}` (type='{info.dtype}')"
                ) from e
            except Exception as e:
                raise RuntimeError(
                    f"Unexpected error parsing field `{info.name}`"
                ) from e

        if not struct_fields and signature.strip():
            raise ValueError("No valid fields parsed from the signature.")

        try:
            DynamicStruct = msgspec.defstruct(struct_name, struct_fields)  # noqa: N806
        except Exception as e:
            raise RuntimeError(f"Error creating struct `{struct_name}`") from e

        return DynamicStruct

    @classmethod
    def _parse_literal_args(cls, args_str: str) -> Tuple:
        """Parse arguments inside Literal[...] robustly, respecting quotes
        and nested structures.
        """
        try:
            # split respecting nested brackets and quotes
            parts = cls._split_args(args_str)
            values = []
            for p in parts:
                try:
                    values.append(ast.literal_eval(p))
                except Exception:
                    # fallback: an unquoted identifier - treat as string
                    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", p):
                        values.append(p)
                    else:
                        raise
            return tuple(values)
        except (SyntaxError, ValueError, TypeError) as e:
            logger.error(str(e))
            raise ValueError(f"Invalid literal arguments: `{args_str}`") from e

    @classmethod
    def _split_args(cls, args_str: str) -> list[str]:
        """Split by commas at top-level only. Respects nested brackets and quotes.

        This is an improved version of your previous _split_args but keeps the same
        method name to remain compatible with existing calls.
        """
        args = []
        level = 0
        current_arg_start = 0
        in_quotes = None

        if not args_str.strip():
            return []

        for i, char in enumerate(args_str):
            if char in ("[", "{", "(") and not in_quotes:
                level += 1
            elif char in ("]", "}", ")") and not in_quotes:
                level -= 1
                if level < 0:
                    raise ValueError("Unbalanced brackets in type arguments")
            elif char in ("'", '"'):
                if in_quotes == char:
                    in_quotes = None
                elif in_quotes is None:
                    in_quotes = char
            elif char == "," and level == 0 and not in_quotes:
                args.append(args_str[current_arg_start:i].strip())
                current_arg_start = i + 1

        args.append(args_str[current_arg_start:].strip())
        return [arg for arg in args if arg]

    @classmethod
    def _parse_type_string(cls, type_str: str) -> type:  # noqa: C901
        """Recursive parser for nested generics like 'list[dict[str, int]]'.

        Supported base names (case-insensitive): list, dict, tuple, union, optional,
        literal, str, int, float, bool, any, none.
        """
        type_str = type_str.strip()
        if not type_str:
            raise ValueError("The type string cannot be empty.")

        # map textual base names to typing objects
        generic_bases = {
            "list": List,
            "dict": Dict,
            "tuple": Tuple,
            "union": Union,
            "optional": Optional,
            "literal": Literal,
            "any": Any,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "none": type(None),
        }

        # simple (non-generic) form
        if "[" not in type_str:
            key = type_str.lower()
            if key in generic_bases:
                return generic_bases[key]
            # fallback to global type_mapping if it exists
            if key in globals().get("type_mapping", {}):
                return globals()["type_mapping"][key]
            raise ValueError(f"Unsupported or unknown type: `{type_str}`")

        # match Base[arg,...]
        m = re.match(r"^\s*([^\[\]]+)\s*\[(.*)\]\s*$", type_str, flags=re.DOTALL)
        if not m:
            raise ValueError(f"Malformed generic type: `{type_str}`")
        base_name, inner = m.groups()
        base_key = base_name.strip().lower()

        if base_key not in generic_bases:
            # try to resolve via type_mapping
            if base_key in globals().get("type_mapping", {}):
                base = globals()["type_mapping"][base_key]
            else:
                raise ValueError(
                    f"Base type not supported: `{base_name}` in `{type_str}`"
                )
        else:
            base = generic_bases[base_key]

        # Literal[...] special handling
        if base is Literal:
            lit_args = cls._parse_literal_args(inner)
            if len(lit_args) == 1:
                return Literal[lit_args[0]]
            return Literal[lit_args]

        # Optional[T] -> Union[T, NoneType]
        if base is Optional:
            parts = cls._split_args(inner)
            if len(parts) != 1:
                raise ValueError("Optional[...] requires exactly 1 argument")
            inner_t = cls._parse_type_string(parts[0])
            return Union[inner_t, type(None)]

        # Union[...] -> Union[T1, T2, ...]
        if base is Union:
            parts = cls._split_args(inner)
            if not parts:
                raise ValueError("Union[...] cannot be empty")
            parsed = tuple(cls._parse_type_string(p) for p in parts)
            if len(parsed) == 1:
                return parsed[0]
            return Union[parsed]

        # Tuple[...] handling
        if base is Tuple:
            parts = cls._split_args(inner)
            if len(parts) == 1 and parts[0].endswith("..."):
                item_type = cls._parse_type_string(parts[0][:-3].strip())
                return Tuple[item_type, ...]
            if not parts:
                return Tuple[()]
            parsed = tuple(cls._parse_type_string(p) for p in parts)
            return Tuple[parsed]

        # generic with positional args: List[T], Dict[K, V], etc.
        parts = cls._split_args(inner)
        parsed_parts = tuple(cls._parse_type_string(p) for p in parts) if parts else ()

        if base is List:
            if len(parsed_parts) != 1:
                raise ValueError("List requires exactly 1 argument")
            return List[parsed_parts[0]]
        if base is Dict:
            if len(parsed_parts) != 2:
                raise ValueError("Dict requires exactly 2 arguments")
            return Dict[parsed_parts[0], parsed_parts[1]]

        # Attempt to parameterize any other generic base
        try:
            if parsed_parts:
                return (
                    base[parsed_parts]
                    if len(parsed_parts) > 1
                    else base[parsed_parts[0]]
                )
        except Exception as e:
            raise ValueError(
                f"Could not create parameterized type for `{type_str}`: {e}"
            ) from e

        raise ValueError(f"Malformed or unsupported generic: `{type_str}`")

    @classmethod
    def _parse_annotations(cls, signature: str) -> List[Any]:  # noqa: C901
        from msgflux.dsl.signature import FieldInfo  # noqa: PLC0415

        fields = []
        current_pos = 0
        level = 0
        in_quotes = None
        current_field_start = 0
        signature = signature.strip()

        if not signature:
            return []

        while current_pos < len(signature):
            char = signature[current_pos]
            if char in ("[", "{", "(") and not in_quotes:
                level += 1
            elif char in ("]", "}", ")") and not in_quotes:
                level -= 1
                if level < 0:
                    raise ValueError(
                        "Unbalanced nesting near `{signature[current_pos:]}`"
                    )
            elif char in ("'", '"'):
                if in_quotes == char:
                    in_quotes = None
                elif in_quotes is None:
                    in_quotes = char

            if char == "," and level == 0 and not in_quotes:
                field_str = signature[current_field_start:current_pos].strip()
                if field_str:
                    fields.append(field_str)
                current_field_start = current_pos + 1
            current_pos += 1

        if level != 0:
            raise ValueError("Unbalanced brackets/parentheses in signature.")
        if in_quotes:
            raise ValueError("Unclosed quotation marks in signature.")

        last_field_str = signature[current_field_start:].strip()
        if last_field_str:
            fields.append(last_field_str)

        result: List[Any] = []
        for field_str in fields:
            parts = field_str.split(":", 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value_dtype = parts[1].strip()
                if not key:
                    raise ValueError(f"Field name cannot be empty in `{field_str}`")
                if not value_dtype:
                    raise ValueError(f"Type cannot be empty after ':' in `{field_str}`")
            else:
                key = field_str.strip()
                value_dtype = "str"
                if not key:
                    raise ValueError(f"Field name cannot be empty in `{field_str}`")

            # FieldInfo is expected to be a simple struct/dataclass with (name, dtype)
            result.append(FieldInfo(name=key, dtype=value_dtype))

        return result


def msgspec_dumps(obj: object) -> str:
    return msgspec.json.encode(obj).decode("utf-8")


def export_to_json(
    obj: object, filepath: Union[str, os.PathLike], indent: Optional[int] = 4
):
    with open(filepath, "wb") as f:
        obj_b = msgspec.json.encode(obj)
        formatted_obj_b = msgspec.json.format(obj_b, indent=indent)
        f.write(formatted_obj_b)


def save(obj: object, f: Union[str, os.PathLike]):
    """Save a Python object to a file in either JSON format.

    Args:
        data:
            Saved object.
        filepath:
            A string or os.PathLike object containing a file name.

    Raises:
        ValueError:
            If the file format is not "json".
        FileNotFoundError:
            If the directory of the provided filepath does not exist.

    !!! example
        ``` python
        data = {"name": "Satoshi", "age": 42}
        save(data, "output.json")
        ```
    """
    directory = os.path.dirname(f)
    if directory and not os.path.exists(directory):
        raise FileNotFoundError(f"The directory `{directory}` does not exist")

    if f.endswith("json"):
        export_to_json(obj, f)
    else:
        raise ValueError(f"Unsupported format: `{f}`. Use `json`.")


def read_json(filepath: Union[str, os.PathLike]) -> Mapping[str, Any]:
    with open(filepath, "rb") as f:
        return msgspec.json.decode(f.read())


def load(f: Union[str, os.PathLike]) -> Any:
    """Load data from a file in either JSON.

    Args:
        f: A string or os.PathLike object containing a file name.

    Returns:
        The Python object loaded from the file.

    Raises:
        FileNotFoundError:
            If the file does not exist.
        ValueError:
            If the file extension is not ".json".

    !!! example
        ``` python
        data = load("data.json")
        ```
    """
    if not os.path.exists(f):
        raise FileNotFoundError(f"The file `{f}` does not exist.")

    if f.endswith(".json"):
        return read_json(f)
    else:
        raise ValueError(f"Unsupported file extension: `{f}`. Use `.json`")


def struct_to_dict(obj: object):
    """Recursively converts a msgspec.Struct object to a pure Python dictionary."""
    if isinstance(obj, msgspec.Struct):
        # Convert the struct to a dictionary and recursively process each value
        return {k: struct_to_dict(v) for k, v in msgspec.structs.asdict(obj).items()}
    elif isinstance(obj, list):
        # Convert each item in the list recursively
        return [struct_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        # If it is a dictionary, recursively convert its values
        return dotdict({k: struct_to_dict(v) for k, v in obj.items()})
    else:  # Returns the value as is for simple types
        return obj


def lower_msgspec_struct_for_openai(  # noqa: C901
    struct_class: Type[Struct],
) -> Type[Struct]:
    """Build an OpenAI-compatible transport struct from a logical msgspec schema.

    OpenAI Structured Outputs reject open-ended JSON objects such as
    ``dict[str, T]``. This helper lowers those types to a closed object shape
    using ``entries: list[{key, value}]`` while keeping the logical schema
    untouched for the rest of msgflux.
    """

    def _is_struct_type(type_hint: Any) -> bool:
        return isinstance(type_hint, type) and issubclass(type_hint, msgspec.Struct)

    def _to_camel_case(name: str) -> str:
        sanitized = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_") or "Generated"
        parts = [part for part in sanitized.split("_") if part]
        camel = "".join(part[:1].upper() + part[1:] for part in parts) or "Generated"
        if camel[0].isdigit():
            camel = f"T{camel}"
        return camel

    def _wrap_annotated(base_type: Any, metadata: Tuple[Any, ...]) -> Any:
        if not metadata:
            return base_type
        return Annotated.__class_getitem__((base_type, *metadata))

    def _format_type_hint(type_hint: Any) -> str:
        if isinstance(type_hint, type):
            return type_hint.__name__
        return str(type_hint).replace("typing.", "")

    class _Compiler:
        def __init__(self):
            self.cache: Dict[Any, Any] = {}

        def _unsupported(self, path: str, type_hint: Any, reason: str) -> None:
            path_str = path or "<root>"
            msg = (
                "Unsupported OpenAI structured output type "
                f"at `{path_str}`: `{_format_type_hint(type_hint)}`. {reason}"
            )
            raise TypeError(msg)

        def _validate_dict_key_type(self, key_type: Any, path: str) -> None:  # noqa: C901
            origin = get_origin(key_type)

            if origin is Annotated:
                self._validate_dict_key_type(get_args(key_type)[0], path)
                return

            if key_type is Any:
                self._unsupported(
                    path,
                    key_type,
                    "`Any` is too broad for OpenAI structured outputs.",
                )

            if key_type in {str, int, float, bool, type(None)}:
                return

            if isinstance(key_type, type) and issubclass(key_type, Enum):
                return

            if origin is Literal:
                if all(
                    isinstance(item, (str, int, float, bool, type(None)))
                    for item in get_args(key_type)
                ):
                    return
                self._unsupported(
                    path,
                    key_type,
                    "Dict keys must resolve to hashable scalar values.",
                )

            if origin is Union:
                non_none_args = [
                    arg for arg in get_args(key_type) if arg is not type(None)
                ]
                if len(non_none_args) != 1:
                    self._unsupported(
                        path,
                        key_type,
                        "Dict keys only support Optional[T] style unions.",
                    )
                self._validate_dict_key_type(non_none_args[0], path)
                return

            if origin in (tuple, Tuple):
                tuple_args = get_args(key_type)
                if len(tuple_args) == 2 and tuple_args[1] is Ellipsis:
                    self._validate_dict_key_type(tuple_args[0], f"{path}[]")
                    return
                for index, item_type in enumerate(tuple_args):
                    self._validate_dict_key_type(item_type, f"{path}[{index}]")
                return

            self._unsupported(
                path,
                key_type,
                "Dict keys must be hashable scalar, Literal, Enum, or Tuple values.",
            )

        def lower_struct(
            self, schema_type: Type[Struct], name_hint: str, path_hint: str = ""
        ) -> Type[Struct]:
            cache_key = ("struct", schema_type)
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

            fields = []
            changed = False

            for field in msgspec.structs.fields(schema_type):
                field_path = f"{path_hint}.{field.name}" if path_hint else field.name
                lowered_type = self.lower_type(
                    field.type, f"{name_hint}_{field.name}", field_path
                )
                changed = changed or lowered_type is not field.type

                if field.default_factory is not msgspec.NODEFAULT:
                    fields.append(
                        (
                            field.name,
                            lowered_type,
                            msgspec.field(default_factory=field.default_factory),
                        )
                    )
                elif field.default is not msgspec.NODEFAULT:
                    fields.append((field.name, lowered_type, field.default))
                else:
                    fields.append((field.name, lowered_type))

            if not changed:
                self.cache[cache_key] = schema_type
                return schema_type

            lowered_struct = defstruct(
                _to_camel_case(name_hint),
                fields,
                kw_only=True,
                module=__name__,
            )
            self.cache[cache_key] = lowered_struct
            return lowered_struct

        def lower_type(self, type_hint: Any, name_hint: str, path: str) -> Any:  # noqa: C901
            origin = get_origin(type_hint)
            metadata: Tuple[Any, ...] = ()
            base_type = type_hint

            if origin is Annotated:
                args = get_args(type_hint)
                base_type = args[0]
                metadata = tuple(args[1:])
                origin = get_origin(base_type)

            if base_type is Any:
                self._unsupported(
                    path,
                    base_type,
                    "`Any` is too broad for OpenAI structured outputs.",
                )

            if base_type in (dict, Dict):
                self._unsupported(
                    path,
                    base_type,
                    "Use `Dict[K, V]` with explicit key and value types.",
                )

            if base_type in (list, List):
                self._unsupported(
                    path,
                    base_type,
                    "Use `List[T]` with an explicit item type.",
                )

            if base_type in (tuple, Tuple):
                self._unsupported(
                    path,
                    base_type,
                    "Use `Tuple[...]` with explicit item types.",
                )

            if base_type in (set, Set, frozenset):
                self._unsupported(
                    path,
                    base_type,
                    "Sets are not supported by OpenAI structured outputs.",
                )

            if _is_struct_type(base_type):
                lowered = self.lower_struct(
                    base_type, name_hint or base_type.__name__, path
                )
                return _wrap_annotated(lowered, metadata)

            if origin in (list, List):
                if len(get_args(base_type)) != 1:
                    self._unsupported(
                        path,
                        base_type,
                        "Lists require exactly one item type.",
                    )
                (item_type,) = get_args(base_type)
                lowered = List[
                    self.lower_type(item_type, f"{name_hint}_item", f"{path}[]")
                ]
                return _wrap_annotated(lowered, metadata)

            if origin in (dict, Dict):
                if len(get_args(base_type)) != 2:
                    self._unsupported(
                        path,
                        base_type,
                        "Dicts require explicit key and value types.",
                    )
                key_type, value_type = get_args(base_type)
                self._validate_dict_key_type(key_type, f"{path}.<key>")
                lowered_key_type = self.lower_type(
                    key_type, f"{name_hint}_key", f"{path}.<key>"
                )
                lowered_value_type = self.lower_type(
                    value_type, f"{name_hint}_value", f"{path}.<value>"
                )

                entry_struct = defstruct(
                    f"{_to_camel_case(name_hint)}Entry",
                    [
                        ("key", lowered_key_type),
                        ("value", lowered_value_type),
                    ],
                    kw_only=True,
                    module=__name__,
                )
                lowered = defstruct(
                    f"{_to_camel_case(name_hint)}Map",
                    [("entries", List[entry_struct])],
                    kw_only=True,
                    module=__name__,
                )
                return _wrap_annotated(lowered, metadata)

            if origin is Union:
                non_none_args = [
                    arg for arg in get_args(base_type) if arg is not type(None)
                ]
                if len(non_none_args) > 1:
                    self._unsupported(
                        path,
                        base_type,
                        "Only Optional[T] unions are supported.",
                    )
                lowered_args = tuple(
                    self.lower_type(arg, f"{name_hint}_option_{index}", path)
                    for index, arg in enumerate(get_args(base_type))
                )
                lowered = Union[lowered_args]
                return _wrap_annotated(lowered, metadata)

            if origin in (tuple, Tuple):
                tuple_args = get_args(base_type)
                if not tuple_args:
                    self._unsupported(
                        path,
                        base_type,
                        "Tuples require explicit item types.",
                    )
                if len(tuple_args) == 2 and tuple_args[1] is Ellipsis:
                    lowered = Tuple[
                        self.lower_type(
                            tuple_args[0], f"{name_hint}_item", f"{path}[]"
                        ),
                        ...,
                    ]
                else:
                    lowered = Tuple[
                        tuple(
                            self.lower_type(
                                arg,
                                f"{name_hint}_item_{index}",
                                f"{path}[{index}]",
                            )
                            for index, arg in enumerate(tuple_args)
                        )
                    ]
                return _wrap_annotated(lowered, metadata)

            if origin in (set, Set, frozenset):
                self._unsupported(
                    path,
                    base_type,
                    "Sets are not supported by OpenAI structured outputs.",
                )

            if origin in (Mapping, ABCMapping):
                self._unsupported(
                    path,
                    base_type,
                    "Use `Dict[K, V]` instead of mapping abstractions.",
                )

            if origin is not None and origin is not Literal:
                self._unsupported(
                    path,
                    base_type,
                    "This generic type is not supported by the OpenAI lowering layer.",
                )

            return _wrap_annotated(base_type, metadata)

    if not issubclass(struct_class, msgspec.Struct):
        raise TypeError(
            "`struct_class` must be a `msgspec.Struct` subclass "
            f"given `{type(struct_class)}`"
        )

    compiler = _Compiler()
    return compiler.lower_struct(struct_class, struct_class.__name__)


def restore_transport_value(  # noqa: C901
    value: Any,
    logical_type: Any,
    *,
    dict_factory: Type[dict] = dotdict,
    strict: bool = False,
    restore_structs: bool = False,
) -> Any:
    """Restore transport-lowered values using the original logical type hint.

    The helper is intentionally reusable:
    - ``strict=True`` is appropriate when restoring provider structured outputs.
    - ``strict=False`` is appropriate when preparing tool kwargs, where values may
      already be in their logical shape.
    """
    origin = get_origin(logical_type)

    if origin is Annotated:
        logical_type = get_args(logical_type)[0]
        origin = get_origin(logical_type)

    if value is None:
        return None

    if logical_type in (Any, object):
        return value

    if origin is Literal:
        return value

    if logical_type is bool:
        if isinstance(value, bool):
            return value
        if value in (0, 1):
            return bool(value)
        if strict:
            raise TypeError(f"Expected bool given `{type(value)}`")
        return value

    if logical_type in (str, int, float):
        if isinstance(value, logical_type):
            return value
        try:
            return logical_type(value)
        except (TypeError, ValueError):
            if strict:
                raise
            return value

    if isinstance(logical_type, type) and issubclass(logical_type, Enum):
        try:
            return logical_type(value)
        except ValueError:
            if strict:
                raise
            return value

    if isinstance(logical_type, type) and issubclass(logical_type, msgspec.Struct):
        if not isinstance(value, Mapping):
            if strict:
                raise TypeError(
                    "Expected a mapping transport value for "
                    f"`{logical_type.__name__}` given `{type(value)}`"
                )
            return value
        restored = {}
        for field in msgspec.structs.fields(logical_type):
            if field.name not in value:
                continue
            restored[field.name] = restore_transport_value(
                value[field.name],
                field.type,
                dict_factory=dict_factory,
                strict=strict,
                restore_structs=restore_structs,
            )
        if restore_structs:
            try:
                return logical_type(**restored)
            except (TypeError, ValueError):
                if strict:
                    raise
        return dict_factory(restored)

    if origin in (list, List):
        if not isinstance(value, list):
            if strict:
                raise TypeError(
                    "Expected a list transport value for "
                    f"`{str(logical_type).replace('typing.', '')}` "
                    f"given `{type(value)}`"
                )
            return value
        item_type = get_args(logical_type)[0]
        return [
            restore_transport_value(
                item,
                item_type,
                dict_factory=dict_factory,
                strict=strict,
                restore_structs=restore_structs,
            )
            for item in value
        ]

    if origin in (dict, Dict, Mapping, ABCMapping):
        args = get_args(logical_type)
        key_type, value_type = args if len(args) == 2 else (Any, Any)
        if not isinstance(value, Mapping):
            if strict:
                raise TypeError(
                    "Expected a mapping transport value for "
                    f"`{str(logical_type).replace('typing.', '')}` "
                    f"given `{type(value)}`"
                )
            return value

        if "entries" in value:
            items = value["entries"]
        elif strict:
            raise ValueError(
                "Expected transport mapping wrapper with required `entries` field "
                f"for `{str(logical_type).replace('typing.', '')}`"
            )
        else:
            items = [{"key": key, "value": item} for key, item in value.items()]

        restored = {}
        for item in items:
            key = restore_transport_value(
                item["key"],
                key_type,
                dict_factory=dict_factory,
                strict=strict,
                restore_structs=restore_structs,
            )
            restored[key] = restore_transport_value(
                item["value"],
                value_type,
                dict_factory=dict_factory,
                strict=strict,
                restore_structs=restore_structs,
            )
        return dict_factory(restored)

    if origin is Union:
        args = tuple(arg for arg in get_args(logical_type) if arg is not type(None))
        if len(args) == 1:
            return restore_transport_value(
                value,
                args[0],
                dict_factory=dict_factory,
                strict=strict,
                restore_structs=restore_structs,
            )
        if args and strict:
            raise TypeError(
                "Unsupported logical type during transport restoration: "
                f"`{str(logical_type).replace('typing.', '')}`. "
                "Only Optional[T] unions are supported."
            )
        return value

    if origin in (tuple, Tuple):
        if not isinstance(value, (list, tuple)):
            if strict:
                raise TypeError(
                    "Expected a tuple-compatible transport value for "
                    f"`{str(logical_type).replace('typing.', '')}` "
                    f"given `{type(value)}`"
                )
            return value
        tuple_args = get_args(logical_type)
        if len(tuple_args) == 2 and tuple_args[1] is Ellipsis:
            item_type = tuple_args[0]
            return tuple(
                restore_transport_value(
                    item,
                    item_type,
                    dict_factory=dict_factory,
                    strict=strict,
                    restore_structs=restore_structs,
                )
                for item in value
            )
        return tuple(
            restore_transport_value(
                item,
                tuple_args[index],
                dict_factory=dict_factory,
                strict=strict,
                restore_structs=restore_structs,
            )
            for index, item in enumerate(value)
        )

    return value


def restore_openai_structured_output(value: Any, logical_type: Any) -> Any:
    """Restore provider-specific transport shapes to the logical output schema."""
    return restore_transport_value(
        value,
        logical_type,
        dict_factory=dotdict,
        strict=True,
    )


def is_optional_field(struct_class: Type[Struct], field_name: str) -> bool:
    """Check if field is Optional."""
    field_type = struct_class.__annotations__.get(field_name)

    if field_type is None:
        return False

    origin = get_origin(field_type)
    if origin is Union or origin is Optional:
        args = get_args(field_type)
        return type(None) in args

    return False
