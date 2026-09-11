import asyncio
import base64
import io
import mimetypes
import os
from typing import Optional, Union
from urllib.parse import urlparse

import httpx2

try:
    import anyio
except ImportError:
    anyio = None


_HEADERS = {"User-Agent": "Mozilla/5.0"}


def encode_base64_from_url(url: str) -> str:
    try:
        response = httpx2.get(url, headers=_HEADERS, timeout=300)
        response.raise_for_status()
        return base64.b64encode(response.content).decode("utf-8")
    except (httpx2.HTTPError, UnicodeDecodeError):
        return url  # Fallback


def encode_local_file_in_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_data_to_base64(path: Union[str, bytes]) -> str:
    if isinstance(path, bytes):
        return base64.b64encode(path).decode("utf-8")
    if "http" in path:
        return encode_base64_from_url(path)
    elif os.path.exists(path) and not os.path.isdir(path):
        return encode_local_file_in_base64(path)
    else:
        return path  # Fallback


# Async versions
async def aencode_base64_from_url(url: str) -> str:
    """Async version of encode_base64_from_url using httpx2."""
    try:
        async with httpx2.AsyncClient(timeout=300.0) as client:
            response = await client.get(url, headers=_HEADERS)
            response.raise_for_status()
            return base64.b64encode(response.content).decode("utf-8")
    except (httpx2.HTTPError, UnicodeDecodeError):
        return url  # Fallback


async def aencode_local_file_in_base64(path: str) -> str:
    """Async version of encode_local_file_in_base64 using anyio.Path."""
    if anyio is None:
        # Fallback to sync version using run_in_executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, encode_local_file_in_base64, path)

    file = anyio.Path(path)
    async with await file.open("rb") as f:
        content = await f.read()
        return base64.b64encode(content).decode("utf-8")


async def aencode_data_to_base64(path: Union[str, bytes]) -> str:
    """Async version of encode_data_to_base64."""
    if isinstance(path, bytes):
        return base64.b64encode(path).decode("utf-8")
    if "http" in path:
        return await aencode_base64_from_url(path)
    elif os.path.exists(path) and not os.path.isdir(path):
        return await aencode_local_file_in_base64(path)
    else:
        return path  # Fallback


def encode_to_io_object(input_data: Union[bytes, str]) -> io.IOBase:
    """Converts an input to a file IO object (such as io.BytesIO or
    a file opened in binary mode).

    Supports:
        - URLs (downloads the content and returns an io.BytesIO).
        - Base64 strings (decodes to an io.BytesIO).
        - Local paths to files (opens the file in binary mode).
        - Bytes (returns an io.BytesIO directly).

    Args:
        input_data: The input to convert to an IO object.

    Returns:
        The IO object containing the data.
    """
    if isinstance(input_data, bytes):
        return io.BytesIO(input_data)

    if isinstance(input_data, str):
        if input_data.startswith("http://") or input_data.startswith("https://"):
            response = httpx2.get(input_data, timeout=300)
            response.raise_for_status()
            return io.BytesIO(response.content)

        try:
            decoded_data = base64.b64decode(input_data)
            return io.BytesIO(decoded_data)
        except (base64.binascii.Error, ValueError):
            pass

        if os.path.exists(input_data) and os.path.isfile(input_data):
            return open(input_data, "rb")

    raise ValueError(
        "Invalid input: must be a URL, Base64, file path, or bytes. "
        f"Given: {type(input_data)}"
    )


def encode_data_to_bytes(
    input_data: Union[bytes, str], *, filename: Optional[str] = "image.png"
) -> io.BytesIO:
    """Converts input to a BytesIO object and sets a name for MIME-type detection.

    Supports:
    - Bytes
    - File paths
    - URLs (http/https)
    - Base64-encoded strings

    Args:
        input_data:
            Raw bytes or string (URL, base64 or file path).
        filename:
            Optional filename used to set the .name attribute in fallback cases.

    Returns:
        A BytesIO object with `.name` attribute set.

    Raises:
        ValueError: If the input string cannot be resolved to a valid source.
    """
    if isinstance(input_data, bytes):
        data, resolved_filename = _resolve_bytes_input(input_data, filename)
    elif isinstance(input_data, str):
        data, resolved_filename = _resolve_string_input(input_data, filename)
    else:
        raise ValueError(f"Invalid input type: {type(input_data)}")

    buffer = io.BytesIO(data)
    if resolved_filename:
        buffer.name = resolved_filename
    return buffer


def _resolve_bytes_input(
    input_data: bytes,
    filename: Optional[str],
) -> tuple[bytes, Optional[str]]:
    if input_data[:2] == b"\xff\xd8":
        filename = "image.jpg"
    elif input_data[:4] == b"\x89PNG":
        filename = "image.png"
    elif input_data[:4] == b"RIFF" and input_data[8:12] == b"WEBP":
        filename = "image.webp"
    return input_data, filename


def _resolve_string_input(
    input_data: str,
    filename: Optional[str],
) -> tuple[bytes, Optional[str]]:
    if input_data.startswith("data:"):
        return _decode_data_url(input_data, filename)
    if os.path.isfile(input_data):
        with open(input_data, "rb") as file:
            return file.read(), os.path.basename(input_data)
    if input_data.startswith(("http://", "https://")):
        response = httpx2.get(input_data, timeout=300)
        response.raise_for_status()
        resolved = os.path.basename(urlparse(str(response.url)).path) or filename
        return response.content, resolved
    try:
        return base64.b64decode(input_data), filename
    except (base64.binascii.Error, ValueError) as exc:
        raise ValueError(
            f"Invalid string input (not a valid path, URL, or base64): {input_data}"
        ) from exc


def _decode_data_url(
    value: str,
    filename: Optional[str],
) -> tuple[bytes, Optional[str]]:
    try:
        header, encoded = value.split(",", 1)
        if ";base64" not in header:
            raise ValueError("data URL is not base64 encoded")
        data = base64.b64decode(encoded, validate=True)
    except (base64.binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 data URL") from exc
    media_type = header[5:].split(";", 1)[0]
    extension = mimetypes.guess_extension(media_type) if media_type else None
    return data, f"upload{extension}" if extension else filename


async def aencode_data_to_bytes(
    input_data: Union[bytes, str], *, filename: Optional[str] = "image.png"
) -> io.BytesIO:
    """Async version of :func:`encode_data_to_bytes`.

    Remote inputs use an async HTTPX2 client. Local file reads and base64
    decoding run in a worker thread so model ``acall`` does not block its event
    loop.
    """
    if not (
        isinstance(input_data, str) and input_data.startswith(("http://", "https://"))
    ):
        return await asyncio.to_thread(
            encode_data_to_bytes,
            input_data,
            filename=filename,
        )

    async with httpx2.AsyncClient(timeout=300.0) as client:
        response = await client.get(input_data)
        response.raise_for_status()
    resolved_filename = os.path.basename(urlparse(str(response.url)).path) or filename
    buffer = io.BytesIO(response.content)
    if resolved_filename:
        buffer.name = resolved_filename
    return buffer
