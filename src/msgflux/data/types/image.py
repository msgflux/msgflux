"""Image data type for multimodal processing."""

import base64
from functools import lru_cache
from typing import Any, Dict

from msgflux.data.types.base import MediaType
from msgflux.utils.chat import ChatBlock
from msgflux.utils.encode import aencode_data_to_base64, encode_data_to_base64
from msgflux.utils.inspect import get_mime_type
from msgflux.utils.validation import is_base64


@lru_cache(maxsize=128)
def _cached_encode(source: str) -> str:
    """Cache encoding to avoid re-encoding same files."""
    return encode_data_to_base64(source)


class Image(MediaType):
    """Image data type for multimodal content.

    Processes image files and URLs, returning ChatBlock format ready for use.

    Args:
        source: Path to local file or URL.
        force_encode: If True, forces base64 encoding even for URLs.
            Default is False (keeps URLs as-is, encodes local files).
        **kwargs: Additional arguments passed to ChatBlock.image
            (e.g., detail="high", etc.).

    Examples:
        Sync usage:
            >>> img = Image("/path/to/photo.jpg", detail="high")
            >>> block = img()
            >>> # {"type": "image_url", "image_url": {"url": "data:..."}}

        Async usage:
            >>> img = Image("https://example.com/image.png")
            >>> block = await img.acall()
            >>> # {"type": "image_url", "image_url": {"url": "https://..."}}

        Force encoding URL:
            >>> img = Image("https://example.com/image.png", force_encode=True)
            >>> block = img()
            >>> # URL content downloaded and encoded as data URI
    """

    def __init__(self, source: str, force_encode: bool = False, **kwargs: Any):  # noqa: FBT001, FBT002
        super().__init__(source, force_encode, **kwargs)

    @property
    def default_mime_type(self) -> str:
        return "image/jpeg"

    def _process(self) -> Dict[str, Any]:
        """Process image synchronously."""
        url = self._prepare_url(self._encode_sync())
        return ChatBlock.image(url, **self.kwargs)

    async def _aprocess(self) -> Dict[str, Any]:
        """Process image asynchronously."""
        url = self._prepare_url(await self._encode_async())
        return ChatBlock.image(url, **self.kwargs)

    def _encode_sync(self) -> str:
        """Encode image synchronously.

        Returns:
            Base64 encoded data or original URL.
        """
        source = self.source

        if isinstance(source, bytes):
            return base64.b64encode(source).decode("utf-8")

        if is_base64(source):
            return source

        if self._is_url(source) and not self.force_encode:
            return source

        return _cached_encode(source)

    async def _encode_async(self) -> str:
        """Encode image asynchronously.

        Returns:
            Base64 encoded data or original URL.
        """
        source = self.source

        if isinstance(source, bytes):
            return base64.b64encode(source).decode("utf-8")

        if is_base64(source):
            return source

        if self._is_url(source) and not self.force_encode:
            return source

        return await aencode_data_to_base64(source)

    def _prepare_url(self, data: str) -> str:
        """Convert data to URL or data URI.

        Args:
            data: Base64 encoded data or URL.

        Returns:
            URL or data URI string.
        """
        if data.startswith("http"):
            return data

        mime_type = self._detect_mime_type()
        return f"data:{mime_type};base64,{data}"

    def _detect_mime_type(self) -> str:
        """Detect MIME type from source."""
        if isinstance(self.source, bytes):
            if self.source[:2] == b"\xff\xd8":
                return "image/jpeg"
            if self.source[:4] == b"\x89PNG":
                return "image/png"
            if self.source[:4] in (b"GIF8", b"GIF9"):
                return "image/gif"
            if self.source[:4] == b"RIFF" and self.source[8:12] == b"WEBP":
                return "image/webp"
            return self.default_mime_type
        mime = get_mime_type(self.source)
        if not mime.startswith("image/"):
            return self.default_mime_type
        return mime
