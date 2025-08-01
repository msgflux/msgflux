from typing import Any, Literal, Optional, Set, Tuple, Union

type_mapping = {
    "str": str,
    "string": str,
    "int": int,
    "integer": int,
    "float": float,
    "number": float,
    "bool": bool,
    "boolean": bool,
    "array": list,
    "list": list,
    "dict": dict,
    "object": dict,
    "none": type(None),
    "null": type(None),
    "any": Any,
    "literal": Literal,
    "optional": Optional,
    "union": Union,
    "tuple": Tuple,
    "set": Set,
}
