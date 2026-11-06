"""Fast JSON codec shared by the checkpoint stores.

msgspec is used whenever it is provably equivalent to the stdlib ``json``
behaviour these stores historically relied on; anything else falls back to
``json`` so stored semantics never change.
"""

from __future__ import annotations

import json
import math
from typing import Any

import msgspec

_PAYLOAD_ENCODER = msgspec.json.Encoder()
_SORTED_PAYLOAD_ENCODER = msgspec.json.Encoder(order="sorted")


def encode_json_text(payload: Any, *, sort_keys: bool = False) -> str:
    """Encode a stored JSON document (item payload, state row, event).

    Plain JSON values (exact dict/list/str/int/finite float/bool/None with str
    keys) use msgspec, ~10x faster than ``json`` and decoding to the same
    value. Anything else keeps the historical ``json.dumps(default=str)``
    semantics (NaN, datetimes, sets, enums...).
    """
    if is_plain_json(payload):
        encoder = _SORTED_PAYLOAD_ENCODER if sort_keys else _PAYLOAD_ENCODER
        try:
            return encoder.encode(payload).decode("utf-8")
        except (UnicodeEncodeError, msgspec.EncodeError, OverflowError):
            pass
    return json.dumps(payload, ensure_ascii=False, sort_keys=sort_keys, default=str)


_DECODER = msgspec.json.Decoder()


def decode_json_text(text: str | bytes) -> Any:
    """Decode stored JSON; msgspec rejects exactly what it would decode differently.

    ``NaN``/``Infinity``, out-of-range floats and lone surrogates raise in
    msgspec and are then decoded by ``json`` with its historical semantics.
    """
    try:
        return _DECODER.decode(text)
    except msgspec.DecodeError:
        return json.loads(text)


def encode_canonical_json(value: Any) -> bytes:
    """Deterministic sorted, compact UTF-8 JSON for content addressing.

    Callers pass validated JSON-native values (finite floats, str keys).
    """
    return _SORTED_PAYLOAD_ENCODER.encode(value)


def is_plain_json(value: Any) -> bool:
    """Qualify optional optimizations without copying or invoking user hooks.

    Full APIs accept opaque objects and Mapping subclasses. Only exact native
    JSON containers/scalars are eligible for automatic fingerprints, so an
    optional optimization never calls an opaque object's iteration/conversion
    methods. The active set is bounded by nesting depth and rejects cycles.
    """
    try:
        return _plain_json_value(value, set())
    except RecursionError:
        return False


def _plain_json_value(value: Any, active: set[int]) -> bool:
    value_type = type(value)
    if value is None or value_type is str or value_type is bool or value_type is int:
        return True
    if value_type is float:
        return math.isfinite(value)
    if value_type is not list and value_type is not dict:
        return False
    identity = id(value)
    if identity in active:
        return False
    active.add(identity)
    try:
        if value_type is list:
            return all(_plain_json_value(item, active) for item in value)
        return all(
            type(key) is str and _plain_json_value(item, active)
            for key, item in value.items()
        )
    finally:
        active.remove(identity)
