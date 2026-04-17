import base64
import inspect
import re
from typing import Any, Tuple, Type, Union


def is_subclass_of(obj: Any, cls: Union[Type[Any], Tuple[Type[Any], ...]]) -> bool:
    if not inspect.isclass(obj):
        return False
    return issubclass(obj, cls)


def is_builtin_type(obj: Any):
    builtin_types = (str, int, float, bool, list, dict, tuple, set, type(None))
    return isinstance(obj, builtin_types)


def is_base64(string) -> bool:
    if not isinstance(string, str):
        return False
    # Must contain only valid base64 characters (A-Z, a-z, 0-9, +, /, =)
    # and length must be a multiple of 4 (with padding)
    if not re.fullmatch(r"[A-Za-z0-9+/]*={0,2}", string):
        return False
    if len(string) % 4 != 0:
        return False
    try:
        base64.b64decode(string, validate=True)
        return True
    except Exception:
        return False
