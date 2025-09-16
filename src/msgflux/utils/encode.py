import base64
import io
import os
from typing import Optional, Union

import requests


def encode_base64_from_url(url: str) -> str:
    try:
        with requests.get(url, timeout=300) as response:
            response.raise_for_status()
            return base64.b64encode(response.content).decode("utf-8")
    except (requests.RequestException, UnicodeDecodeError):
        return url  # Fallback


def encode_local_file_in_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_data_to_base64(path: str) -> str:
    if "http" in path:
        return encode_base64_from_url(path)
    elif os.path.exists(path) and not os.path.isdir(path):
        return encode_local_file_in_base64(path)
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
            response = requests.get(input_data, timeout=300)
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
    filename = None
    if isinstance(input_data, bytes):
        data = input_data

    elif isinstance(input_data, str):
        # 1. Check if it's a valid local file path
        if os.path.isfile(input_data):
            with open(input_data, "rb") as f:
                data = f.read()
            filename = os.path.basename(input_data)

        # 2. Check if it's a URL
        elif input_data.startswith(("http://", "https://")):
            response = requests.get(input_data, timeout=300)
            response.raise_for_status()
            data = response.content
            filename = os.path.basename(response.url) or filename

        # 3. Try to decode base64
        else:
            try:
                data = base64.b64decode(input_data)
            except (base64.binascii.Error, ValueError) as e:
                raise ValueError(
                    "Invalid string input (not a valid path, URL, or base64): "
                    f"{input_data}"
                ) from e

    else:
        raise ValueError(f"Invalid input type: {type(input_data)}")

    buffer = io.BytesIO(data)
    if filename:
        buffer.name = filename
    return buffer
