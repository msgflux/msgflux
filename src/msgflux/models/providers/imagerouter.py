from os import getenv

from msgflux.models.providers.openai import OpenAITextToImage
from msgflux.models.registry import register_model
from msgflux.models.types import TextToImageModel


class _BaseImageRouter:
    """Configurations to use ImageRouter models."""

    provider: str = "imagerouter"
    api_key_env: str = "IMAGEROUTER_API_KEY"

    def _get_base_url(self):
        default_url = "https://api.imagerouter.io/v1/openai"
        base_url = getenv("IMAGEROUTER_BASE_URL", default_url)
        if base_url is None:
            raise ValueError("Please set `IMAGEROUTER_BASE_URL`")
        return base_url

    def _get_api_key(self):
        """Load API keys from environment variable."""
        key = getenv("IMAGEROUTER_API_KEY")
        if not key:
            raise ValueError(
                "The ImageRouter API key is not available."
                "Please set `IMAGEROUTER_API_KEY`"
            )
        return key


@register_model
class ImageRouterTextToImage(_BaseImageRouter, OpenAITextToImage, TextToImageModel):
    """ImageRouter Text to Image."""
