import ast
import os
import re
from collections import OrderedDict
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

from msgflux.dotdict import dotdict
from msgflux.dsl.signature import FieldInfo
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
        """Parse arguments inside Literal[...] robustly, respecting quotes and nested structures."""
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
        GENERIC_BASES = {
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
            if key in GENERIC_BASES:
                return GENERIC_BASES[key]
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

        if base_key not in GENERIC_BASES:
            # try to resolve via type_mapping
            if base_key in globals().get("type_mapping", {}):
                base = globals()["type_mapping"][base_key]
            else:
                raise ValueError(
                    f"Base type not supported: `{base_name}` in `{type_str}`"
                )
        else:
            base = GENERIC_BASES[base_key]

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
    def _parse_annotations(cls, signature: str) -> List[Any]:
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
