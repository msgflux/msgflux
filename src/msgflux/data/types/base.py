"""Base class for multimodal data types."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class MediaType(ABC):
    """Base class for multimodal types.

    Returns a ChatBlock ready for use when called.
    Supports both sync (__call__) and async (acall) processing.

    Args:
        source: Path or URL to the media file.
        force_encode: If True, forces base64 encoding even for URLs.
        **kwargs: Additional arguments passed to ChatBlock method.

    Attributes:
        source: The original source path or URL.
        force_encode: Whether to force encoding.
        kwargs: Additional arguments for ChatBlock.
    """

    def __init__(self, source: str, force_encode: bool = False, **kwargs: Any):  # noqa: FBT001, FBT002
        self.source = source
        self.force_encode = force_encode
        self.kwargs = kwargs
        self._result: Optional[Dict[str, Any]] = None

    def __call__(self) -> Dict[str, Any]:
        """Process and return ChatBlock (sync).

        Returns:
            ChatBlock dict ready for use in messages.
        """
        if self._result is None:
            self._result = self._process()
        return self._result

    async def acall(self) -> Dict[str, Any]:
        """Process and return ChatBlock (async).

        Returns:
            ChatBlock dict ready for use in messages.
        """
        if self._result is None:
            self._result = await self._aprocess()
        return self._result

    @abstractmethod
    def _process(self) -> Dict[str, Any]:
        """Synchronous processing - returns ChatBlock."""
        pass

    @abstractmethod
    async def _aprocess(self) -> Dict[str, Any]:
        """Asynchronous processing - returns ChatBlock."""
        pass

    @property
    @abstractmethod
    def default_mime_type(self) -> str:
        """Default MIME type for fallback."""
        pass

    def _is_url(self, source) -> bool:
        """Check if source is a URL."""
        if isinstance(source, bytes):
            return source.startswith(b"http")
        return source.startswith("http")
