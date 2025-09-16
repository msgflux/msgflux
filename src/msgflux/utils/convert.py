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
