import inspect
import mimetypes
import os
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse


def fn_has_parameters(fn: Callable) -> bool:
    sig = inspect.signature(fn)
    for param in sig.parameters.values():
        # Ignore *args and **kwargs
        if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            return True
    return False


def get_mime_type(source: str) -> str: # noqa: C901
    """Tries to guess the MIME type, with fallback."""
    mime_type, _ = mimetypes.guess_type(source)
    if mime_type:
        return mime_type
    # Extension-based fallbacks (simplistic)
    ext = Path(source).suffix.lower()
    if not ext:
        ext = f".{source.lower()}"
    if ext in {".jpeg", ".jpg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".gif":
        return "image/gif"
    if ext == ".mp3":
        return "audio/mpeg"
    if ext == ".wav":
        return "audio/wav"
    if ext == ".ogg":
        return "audio/ogg"
    if ext == ".flac":
        return "audio/flac"
    if ext == ".opus":
        return "audio/opus"
    if ext == ".m4a":
        return "audio/mp4"
    if ext == ".webm":
        return "audio/webm"
    if ext == ".pdf":
        return "application/pdf"
    # Generic fallback
    return "application/octet-stream"


def get_fn_name():
    return inspect.currentframe().f_back.f_code.co_name


def get_filename(data_path: str) -> str:
    if data_path.startswith(("http://", "https://", "ftp://")):
        parsed_url = urlparse(data_path)
        filename = os.path.basename(parsed_url.path)
    else:  # Local file
        filename = os.path.basename(data_path)
    return filename
