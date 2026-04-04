import re
from typing import Any, Literal, Optional, Set, Tuple, Union

def is_jinja_template(template: str) -> bool:
    """Return True if the template contains Jinja2 syntax (``{{ }}``, ``{% %}``, ``{# #}``)."""
    return bool(re.search(r"\{\{|\{%|\{#", template))


def has_format_placeholder(template: str) -> bool:
    """Return True if the template contains a Python positional format placeholder (``{}`` or ``{0}``)."""
    return bool(re.search(r"\{(?:\d+)?\}", template))


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
