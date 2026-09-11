"""Serializable conversation content; publication performs no media I/O."""

from copy import deepcopy
from typing import Any
from urllib.parse import urlsplit


def normalize_content(content: Any) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list) or not content:
        raise TypeError("Inbox content must be text or a non-empty list of blocks")
    for block in content:
        if not isinstance(block, dict):
            raise TypeError("Inbox content blocks must be dictionaries")
        if block.get("type") == "text":
            if set(block) != {"type", "text"} or not isinstance(block["text"], str):
                raise ValueError("Expected a text block with string text")
        elif block.get("type") == "image_url":
            _validate_image(block)
        else:
            raise ValueError("Inbox supports only text and image_url blocks")
    return deepcopy(content)


def _validate_image(block):
    image = block.get("image_url")
    if set(block) != {"type", "image_url"} or not isinstance(image, dict):
        raise ValueError("Expected an image_url block")
    if set(image) - {"url", "detail"}:
        raise ValueError("Unsupported image fields")
    url = image.get("url")
    if not isinstance(url, str) or not url:
        raise ValueError("Image URL must be a non-empty string")
    parsed = urlsplit(url)
    if not (
        (parsed.scheme in {"https", "http"} and parsed.netloc)
        or (url.startswith("data:image/") and ";base64," in url)
    ):
        raise ValueError("Use an HTTP(S) image URL or an image data URI")
    if image.get("detail", "auto") not in {"auto", "low", "high"}:
        raise ValueError("Image detail must be auto, low or high")


def validate_description(metadata, ref):
    for field in ("origin", "description"):
        if not isinstance(metadata.get(field), str) or not metadata[field].strip():
            raise ValueError(f"Message {field} must be a non-empty string")
    if ref is not None and not isinstance(ref, str):
        raise TypeError("Message ref must be a string or None")
