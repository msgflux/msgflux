"""Video data type for multimodal processing."""

from functools import lru_cache
from typing import Any, Dict

from msgflux.data.types.base import MediaType
from msgflux.utils.chat import ChatBlock
from msgflux.utils.encode import aencode_data_to_base64, encode_data_to_base64
from msgflux.utils.inspect import get_mime_type
from msgflux.utils.validation import is_base64, is_http_url


@lru_cache(maxsize=16)
def _cached_encode(source: str) -> str:
    """Cache encoding to avoid re-encoding same files.

    Note: Cache size is limited to 16 entries to avoid high memory usage
    with large video files.
    """
    return encode_data_to_base64(source)


class Video(MediaType):
    """Video data type for multimodal content.

    Processes video files and URLs, returning ChatBlock format ready for use.
    Local files are always encoded; URLs can be kept as-is or encoded.

    Args:
        source: Path to local file or URL.
        force_encode: If True, forces base64 encoding even for URLs.
            Default is True for video (encodes local files, keeps URLs).
        **kwargs: Additional arguments passed to ChatBlock.video
            (e.g., format="mp4", etc.).

    Examples:
        Sync usage with local file:
            >>> vid = Video("/path/to/video.mp4", format="mp4")
            >>> block = vid()
            >>> # {"type": "video_url", "video_url": {"url": "data:..."}}

        Async usage with URL:
            >>> vid = Video("https://example.com/video.mp4")
            >>> block = await vid.acall()
            >>> # {"type": "video_url", "video_url": {"url": "https://..."}}
    """

    def __init__(self, source: str, force_encode: bool = True, **kwargs: Any):  # noqa: FBT001, FBT002
        super().__init__(source, force_encode, **kwargs)

    @property
    def default_mime_type(self) -> str:
        return "video/mp4"

    def _process(self) -> Dict[str, Any]:
        """Process video synchronously."""
        url = self._prepare_url(self._encode_sync())
        return ChatBlock.video(url, **self.kwargs)

    async def _aprocess(self) -> Dict[str, Any]:
        """Process video asynchronously."""
        url = self._prepare_url(await self._encode_async())
        return ChatBlock.video(url, **self.kwargs)

    def _encode_sync(self) -> str:
        """Encode video synchronously.

        Returns:
            Base64 encoded data or original URL.
        """
        source = self.source

        if is_base64(source):
            return source

        # URLs: keep as-is (don't force encode URLs for video by default)
        if self._is_url(source) and not self.force_encode:
            return source

        # Local files: always encode
        return _cached_encode(source)

    async def _encode_async(self) -> str:
        """Encode video asynchronously.

        Returns:
            Base64 encoded data or original URL.
        """
        source = self.source

        if is_base64(source):
            return source

        # URLs: keep as-is (don't force encode URLs for video by default)
        if self._is_url(source) and not self.force_encode:
            return source

        # Local files: always encode
        return await aencode_data_to_base64(source)

    def _prepare_url(self, data: str) -> str:
        """Convert data to URL or data URI.

        Args:
            data: Base64 encoded data or URL.

        Returns:
            URL or data URI string.
        """
        if is_http_url(data):
            return data

        mime_type = self._detect_mime_type()
        return f"data:{mime_type};base64,{data}"

    def _detect_mime_type(self) -> str:
        """Detect MIME type from source."""
        mime = get_mime_type(self.source)
        if not mime.startswith("video/"):
            return self.default_mime_type
        return mime
