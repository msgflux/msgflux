from typing import Any, Dict, Optional, Union

from msgflux.core.dotdict import dotdict

_MISSING: Any = object()


class Message(dotdict):
    def __init__(
        self,
        *,
        content: Optional[Union[str, Dict[str, Any]]] = _MISSING,
        context: Optional[Dict[str, Any]] = _MISSING,
        texts: Optional[Dict[str, Any]] = _MISSING,
        audios: Optional[Dict[str, Any]] = _MISSING,
        images: Optional[Dict[str, Any]] = _MISSING,
        videos: Optional[Dict[str, Any]] = _MISSING,
        extra: Optional[Dict[str, Any]] = _MISSING,
        hidden_keys: Optional[list[str]] = None,
        store=None,
        store_prefix: Optional[str] = None,
    ):
        super().__init__(
            hidden_keys=hidden_keys,
            store=store,
            store_prefix=store_prefix,
        )

        def set_field(name: str, value: Any, default: Any):
            if value is _MISSING:
                if dict.__contains__(self, name):
                    return
                value = default() if callable(default) else default
            elif value is None and default is dict:
                value = {}
            wrapped = self._wrap(value)
            dict.__setitem__(self, name, wrapped)
            self._persist_store_key(name, wrapped)

        set_field("content", content, None)
        set_field("texts", texts, dict)
        set_field("context", context, dict)
        set_field("audios", audios, dict)
        set_field("images", images, dict)
        set_field("videos", videos, dict)
        set_field("extra", extra, dict)
        set_field("outputs", _MISSING, dict)
        set_field("response", _MISSING, dict)

    def get_response(self):
        if self.get("response"):
            return next(iter(self.get("response").values()))
        else:
            return self.get("response")
