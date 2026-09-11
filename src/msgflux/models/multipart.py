"""Helpers for replayable multipart model requests."""

from __future__ import annotations

import mimetypes
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import msgspec

from msgflux.utils.encode import aencode_data_to_bytes, encode_data_to_bytes

MultipartFile: TypeAlias = tuple[str, bytes, str]
MultipartData: TypeAlias = dict[str, str | list[str]]


def prepare_multipart_file(
    value: bytes | str,
    *,
    default_filename: str,
) -> MultipartFile:
    """Convert a supported input into an HTTPX-compatible replayable part."""
    buffer = encode_data_to_bytes(value, filename=default_filename)
    return _buffer_to_part(buffer, default_filename)


async def aprepare_multipart_file(
    value: bytes | str,
    *,
    default_filename: str,
) -> MultipartFile:
    """Asynchronously convert a supported input into a multipart part."""
    buffer = await aencode_data_to_bytes(value, filename=default_filename)
    return _buffer_to_part(buffer, default_filename)


def prepare_multipart_data(params: Mapping[str, Any]) -> MultipartData:
    """Encode scalar and repeated form fields using bracketed array names."""
    fields: MultipartData = {}
    for name, value in params.items():
        if value is None:
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            fields[f"{name}[]"] = [_form_value(item) for item in value]
        else:
            fields[name] = _form_value(value)
    return fields


def _buffer_to_part(buffer, default_filename: str) -> MultipartFile:
    filename = str(getattr(buffer, "name", default_filename))
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return filename, buffer.getvalue(), content_type


def _form_value(value: Any) -> str:
    if isinstance(value, Mapping):
        return msgspec.json.encode(value).decode()
    return str(value).lower() if isinstance(value, bool) else str(value)
