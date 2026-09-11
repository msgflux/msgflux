"""Shared Server-Sent Events decoding for model transports."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Iterator
from json import JSONDecodeError, loads
from typing import Any


def iter_sse_json(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from synchronous SSE lines."""
    for data in _iter_sse_data(lines):
        if data.strip() == "[DONE]":
            return
        yield _decode_sse_json(data)


async def aiter_sse_json(
    lines: AsyncIterator[str],
) -> AsyncIterator[dict[str, Any]]:
    """Yield JSON objects from asynchronous SSE lines."""
    async for data in _aiter_sse_data(lines):
        if data.strip() == "[DONE]":
            return
        yield _decode_sse_json(data)


def _iter_sse_data(lines: Iterable[str]) -> Iterator[str]:
    data_lines: list[str] = []
    for raw_line in lines:
        line = raw_line.rstrip("\r")
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines.clear()
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if field == "data":
            data_lines.append(
                value[1:] if separator and value.startswith(" ") else value
            )
    if data_lines:
        yield "\n".join(data_lines)


async def _aiter_sse_data(lines: AsyncIterator[str]) -> AsyncIterator[str]:
    data_lines: list[str] = []
    async for raw_line in lines:
        line = raw_line.rstrip("\r")
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines.clear()
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if field == "data":
            data_lines.append(
                value[1:] if separator and value.startswith(" ") else value
            )
    if data_lines:
        yield "\n".join(data_lines)


def _decode_sse_json(data: str) -> dict[str, Any]:
    try:
        payload = loads(data)
    except JSONDecodeError as exc:
        raise ValueError("Provider returned invalid SSE JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Provider SSE data must be a JSON object")
    return payload
