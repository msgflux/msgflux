from typing import Any, Dict, Optional

import msgspec


class dotdict(dict): # noqa: N801
    """A dictionary with dot access and nested path support.

    dotdict allows you to access and modify values as attributes (e.g., `obj.key`)
    and also allows reading and writing nested paths using strings with dot separators
    (e.g., `obj.get("user.profile.name")`).

    Main features:
    - Dot access (`obj.key`)
    - Traditional square bracket access (`obj['key']`)
    - Nested reading via `.get("a.b.c")`
    - Nested writing via `.set("a.b.c", value)`
    - Conversion to standard dict with `.to_dict()`
    - Support for Msgspec serialization (`__json__`)
    - Support for lists with path indices (e.g., `"items.0.name"`)
    - Optional immutability (`frozen=True`)
    """

    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        *,
        frozen: Optional[bool] = False,
        **kwargs,
    ):
        """Initializes an instance of dotdict.

        Args:
            data:
                Base dictionary to initialize data.
            frozen:
                If True, prevents changes after creation.
            **kwargs:
                Additional key=value pairs.

        ::: example:
            d = dotdict({"user": {"name": "Maria"}}, frozen=False)
            print(d.user.name)
            >> Maria
        """
        data = data or {}
        self._frozen = frozen
        super().__init__()
        for key, value in {**data, **kwargs}.items():
            super().__setitem__(key, self._wrap(value))

    def __getattr__(self, attr: str):
        try:
            return self[attr]
        except KeyError as e:
            raise AttributeError(
                f"`dotdict` object has no attribute '{attr}'"
            ) from e

    def __setattr__(self, key: str, value: Any):
        if key.startswith("_"):
            super().__setattr__(key, value)
        elif hasattr(self, "_frozen") and self._frozen:
            raise AttributeError("Cannot modify frozen dotdict")
        else:
            self[key] = value

    def __setitem__(self, key: str, value: Any):
        if getattr(self, "_frozen", False):
            raise AttributeError("Cannot modify frozen dotdict")
        super().__setitem__(key, self._wrap(value))

    def __delattr__(self, key: str):
        if getattr(self, "_frozen", False):
            raise AttributeError("Cannot delete from frozen dotdict")
        try:
            del self[key]
        except KeyError as e:
            raise AttributeError(
                f"`dotdict` object has no attribute `{key}`"
            ) from e

    def _wrap(self, value: Any):
        if isinstance(value, dict):
            return dotdict(value, frozen=getattr(self, "_frozen", False))
        elif isinstance(value, list):
            return [self._wrap(item) for item in value]
        return value

    def get(self, path: str, default: Any = None) -> Any:
        """Access nested values via dot path.

        !!! example:
            get('user.profile.age').
        """
        keys = path.split(".")
        current = self
        try:
            for key in keys:
                if isinstance(current, list):
                    key = int(key) # noqa: PLW2901
                current = current[key]
            return current
        except (KeyError, IndexError, ValueError, TypeError):
            return default

    def set(self, path: str, value: Any):
        """Set nested value via dot path.

        !!! example

            set('user.profile.age', 31).
        """
        if self._frozen:
            raise AttributeError("Cannot modify frozen dotdict")

        keys = path.split(".")
        current = self
        for i, key in enumerate(keys):
            if isinstance(current, list):
                key = int(key) # noqa: PLW2901

            if i == len(keys) - 1:
                if isinstance(current, list):
                    key_i = int(key)
                    current[key_i] = self._wrap(value)
                else:
                    current[key] = self._wrap(value)
                return

            if isinstance(current, list):
                key_i = int(key)
                if (
                    key_i >= len(current)
                    or not isinstance(current[key_i], (dict, dotdict))
                ):
                    current[key_i] = dotdict()
                current = current[key_i]
            else:
                if key not in current or not isinstance(
                    current[key], (dict, dotdict, list)
                ):
                    current[key] = dotdict()
                current = current[key]

    def update(self, *args, **kwargs):
        """Extends dict.update to support nested keys while maintaining DotDict."""
        if self._frozen:
            raise AttributeError("Cannot modify frozen DotDict")

        # Collects all key-value pairs
        other = {}
        if args:
            if len(args) > 1:
                raise TypeError(
                    f"update expected at most 1 arguments, given `{len(args)}`"
                )
            other.update(args[0])
        other.update(kwargs)

        for key, value in other.items():
            # Nested key with dot?
            if isinstance(key, str) and "." in key:
                self.set(key, value)

            # Value is dict and there is already a dotdict on that key?
            # Merge recursively
            elif (
                isinstance(value, dict)
                and key in self
                and isinstance(self[key], dotdict)
            ):
                self[key].update(value)

            # General case: normal assignment (call __setitem__ e wrap)
            else:
                self[key] = value

    def to_dict(self):
        def unwrap(value):
            if isinstance(value, dotdict):
                return {k: unwrap(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [unwrap(item) for item in value]
            return value

        return unwrap(self)

    def to_json(self) -> bytes:
        """Returns a encoded-JSON."""
        return msgspec.json.encode(self.to_dict())

    def __repr__(self):
        attrs_str = "\n".join(f"   '{k}': {v!r}" for k, v in self.to_dict().items())
        return f"{self.__class__.__name__}({{\n{attrs_str}\n}})"

    def __str__(self):
        return str(self.to_dict())
