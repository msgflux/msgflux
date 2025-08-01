import hashlib
import re
from typing import Any


def convert_camel_snake_to_title(name: str) -> str:
    """Convert a name to title format."""
    if "_" in name:
        name = name.replace("_", " ")
    else:
        name = re.sub(r"(?<!^)([A-Z])", r" \1", name)
    return name.title()


def convert_camel_to_snake_case(camel_str) -> str:
    snake_str = re.sub(r"(?<!^)([A-Z])", r"_\1", camel_str).lower()
    return snake_str


def convert_str_to_hash(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def convert_none_to_string(obj: Any) -> Any:
    """If a NoneType is detected in object, convert to 'None'."""
    if isinstance(obj, dict):
        return {k: convert_none_to_string(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_none_to_string(item) for item in obj]
    elif obj is None:
        return "None"
    else:
        return obj


def convert_string_to_none(obj: Any) -> Any:
    """If a 'None' is detected in object, convert to None type."""
    if isinstance(obj, dict):
        return {k: convert_string_to_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_string_to_none(item) for item in obj]
    elif obj == "None":
        return None
    else:
        return obj
