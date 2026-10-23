"""Audio data type for multimodal processing."""

from pathlib import Path
from typing import Any, Dict

from msgflux.data.types.base import MediaType
from msgflux.utils.chat import ChatBlock
from msgflux.utils.encode import aencode_data_to_base64, encode_data_to_base64
from msgflux.utils.inspect import get_mime_type
from msgflux.utils.validation import is_base64


class Audio(MediaType):
    """Audio data type for multimodal content.

    Processes audio files and URLs, returning ChatBlock format ready for use.
    Audio is always encoded to base64 by default.

    Args:
        source: Path to local file or URL.
        force_encode: If True, forces base64 encoding even for URLs.
            Default is True for audio (always encodes).
        **kwargs: Additional arguments (currently unused for audio).

    Examples:
        Sync usage:
            >>> aud = Audio("/path/to/audio.mp3")
            >>> block = aud()
            >>> # {"type": "input_audio", "input_audio": {"data": "..."}}

        Async usage:
            >>> aud = Audio("https://example.com/audio.wav")
            >>> block = await aud.acall()
    """

    def __init__(self, source: str, force_encode: bool = True, **kwargs: Any):  # noqa: FBT001, FBT002
        super().__init__(source, force_encode, **kwargs)

    @property
    def default_mime_type(self) -> str:
        return "audio/mpeg"

    def _process(self) -> Dict[str, Any]:
        """Process audio synchronously."""
        base64_data = self._encode_sync()
        audio_format = self._get_audio_format()
        return ChatBlock.audio(base64_data, audio_format)

    async def _aprocess(self) -> Dict[str, Any]:
        """Process audio asynchronously."""
        base64_data = await self._encode_async()
        audio_format = self._get_audio_format()
        return ChatBlock.audio(base64_data, audio_format)

    def _encode_sync(self) -> str:
        """Encode audio synchronously.

        Returns:
            Base64 encoded data.
        """
        source = self.source

        if is_base64(source):
            return source

        return encode_data_to_base64(source)

    async def _encode_async(self) -> str:
        """Encode audio asynchronously.

        Returns:
            Base64 encoded data.
        """
        source = self.source

        if is_base64(source):
            return source

        return await aencode_data_to_base64(source)

    def _get_audio_format(self) -> str:
        """Get audio format from source extension.

        Returns:
            Audio format string (e.g., "mp3", "wav").
        """
        suffix = Path(self.source).suffix.lstrip(".")
        if suffix:
            return suffix
        # Fallback based on MIME type
        mime = get_mime_type(self.source)
        if mime.startswith("audio/"):
            return mime.split("/")[1]
        return "mpeg"

    def _detect_mime_type(self) -> str:
        """Detect MIME type from source."""
        mime = get_mime_type(self.source)
        if not mime.startswith("audio/"):
            return self.default_mime_type
        return mime
