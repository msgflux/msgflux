"""File data type for multimodal processing."""

from functools import lru_cache
from typing import Any, Dict

from msgflux.data.types.base import MediaType
from msgflux.utils.chat import ChatBlock
from msgflux.utils.encode import aencode_data_to_base64, encode_data_to_base64
from msgflux.utils.inspect import get_filename, get_mime_type
from msgflux.utils.validation import is_base64


@lru_cache(maxsize=32)
def _cached_encode(source: str) -> str:
    """Cache encoding to avoid re-encoding same files.

    Note: Cache size is limited to 32 entries to avoid high memory usage
    with large document files.
    """
    return encode_data_to_base64(source)


class File(MediaType):
    """File data type for multimodal content.

    Processes files (PDFs, documents, etc.), returning ChatBlock format ready for use.
    Files are always encoded to base64.

    Args:
        source: Path to local file or URL.
        force_encode: If True, forces base64 encoding. Default is True
            (files are always encoded).
        **kwargs: Additional arguments (currently unused for files).

    Examples:
        Sync usage:
            >>> doc = File("/path/to/document.pdf")
            >>> block = doc()
            >>> # {"type": "file", "file": {"filename": "...", "file_data": "..."}}

        Async usage:
            >>> doc = File("https://example.com/report.pdf")
            >>> block = await doc.acall()
    """

    def __init__(self, source: str, force_encode: bool = True, **kwargs: Any):  # noqa: FBT001, FBT002
        super().__init__(source, force_encode, **kwargs)

    @property
    def default_mime_type(self) -> str:
        return "application/octet-stream"

    def _process(self) -> Dict[str, Any]:
        """Process file synchronously."""
        base64_data = self._encode_sync()
        filename = get_filename(self.source)
        file_data_uri = self._build_data_uri(base64_data)
        return ChatBlock.file(filename, file_data_uri)

    async def _aprocess(self) -> Dict[str, Any]:
        """Process file asynchronously."""
        base64_data = await self._encode_async()
        filename = get_filename(self.source)
        file_data_uri = self._build_data_uri(base64_data)
        return ChatBlock.file(filename, file_data_uri)

    def _encode_sync(self) -> str:
        """Encode file synchronously.

        Returns:
            Base64 encoded data.
        """
        source = self.source

        if not self.force_encode and is_base64(source):
            return source

        return _cached_encode(source)

    async def _encode_async(self) -> str:
        """Encode file asynchronously.

        Returns:
            Base64 encoded data.
        """
        source = self.source

        if not self.force_encode and is_base64(source):
            return source

        return await aencode_data_to_base64(source)

    def _build_data_uri(self, base64_data: str) -> str:
        """Build data URI from base64 data.

        Args:
            base64_data: Base64 encoded data.

        Returns:
            Data URI string.
        """
        mime_type = self._detect_mime_type()
        return f"data:{mime_type};base64,{base64_data}"

    def _detect_mime_type(self) -> str:
        """Detect MIME type from source."""
        filename = get_filename(self.source)
        mime = get_mime_type(self.source)

        # Special handling for PDFs
        if mime == "application/octet-stream" and filename.lower().endswith(".pdf"):
            return "application/pdf"

        return mime
