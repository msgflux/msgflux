from os import getenv
from typing import Optional

from msgflux.models.providers.openai import OpenAITextToSpeech
from msgflux.models.registry import register_model


class _BaseKokoro:
    """Configuration for OpenAI-compatible Kokoro TTS servers."""

    provider: str = "kokoro"

    def _get_base_url(self):
        return getenv("KOKORO_BASE_URL", "http://localhost:8880/v1")

    def _get_api_key(self):
        return getenv("KOKORO_API_KEY", "not-needed")


@register_model
class KokoroTextToSpeech(_BaseKokoro, OpenAITextToSpeech):
    """Kokoro Text to Speech via OpenAI-compatible speech endpoints."""

    def __init__(
        self,
        model_id: str = "kokoro",
        voice: Optional[str] = "af_heart",
        speed: Optional[float] = 1.0,
        stream_chunk_size: int = 1024,
        base_url: Optional[str] = None,
        retry: Optional[object] = None,
    ):
        super().__init__(
            model_id=model_id,
            voice=voice,
            speed=speed,
            stream_chunk_size=stream_chunk_size,
            base_url=base_url,
            retry=retry,
        )
