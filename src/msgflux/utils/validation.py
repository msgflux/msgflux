import base64
import inspect
from typing import Any, Tuple, Type, Union


def is_subclass_of(obj: Any, cls: Union[Type[Any], Tuple[Type[Any], ...]]) -> bool:
    if not inspect.isclass(obj):
        return False
    return issubclass(obj, cls)


def is_builtin_type(obj: Any):
    builtin_types = (str, int, float, bool, list, dict, tuple, set, type(None))
    return isinstance(obj, builtin_types)


def is_base64(string: str):
    try:
        return base64.b64encode(base64.b64decode(string)) == string
    except Exception:
        return False
