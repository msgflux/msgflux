import base64
from os import getenv
from typing import (
    Any, Dict, List, Literal, Mapping, Optional, Union
)

try:
    import replicate
except ImportError:
    replicate = None

from msgflux.dotdict import dotdict
from msgflux.models.base import BaseModel
from msgflux.models.registry import register_model
from msgflux.models.response import ModelResponse
from msgflux.models.types import (
    ImageTextToImageModel,
    ImageTextToVideoModel,    
    TextToImageModel,
    TextToVideoModel
)
from msgflux.utils.encode import encode_data_to_bytes
from msgflux.utils.tenacity import model_retry


class _BaseReplicate(BaseModel):
    provider: str = "replicate"

    def _initialize(self):
        """Initialize the OpenAI client with empty API key."""
        if replicate is None:
            raise ImportError(
                "`replicate` client is not available. "
                "Install with `pip install replicate`."
            )
        self.client = replicate.run
        self.aclient = replicate.async_run
        return

    def _get_api_key(self):
        """Load API keys from environment variable."""
        key = getenv("REPLICATE_API_KEY")
        if not key:
            raise ValueError(
                "The Replicate API key is not available. Please set `REPLICATE_API_KEY`"
            )

    def _execute_model(self, **kwargs):
        """Main method to execute the model."""
        model_id = kwargs.pop("model_id")
        model_output = self.client(model_id, input=kwargs)
        return model_output

    async def _aexecute_model(self, **kwargs):
        """Async version of _execute_model."""
        model_id = kwargs.pop("model_id")
        model_output = await self.aclient(model_id, input=kwargs)
        return model_output

class ReplicateImageTextToImage(_BaseReplicate, ImageTextToImageModel):

    def __init__(
        self,
        *,
        model_id: str,
        go_fast: Optional[bool] = False,        
        moderation: Optional[Literal["auto", "low"]] = None,
        base_url: Optional[str] = None,
    ):
        """Args:
        model_id:
            Model ID in provider.
        go_fast:
            Run faster predictions with model optimized for speed (currently
            fp8 quantized); disable to run in original bf16. Note that outputs
            will not be deterministic when this is enabled, even if you set a seed.
        moderation:
            Control the content-moderation level for images generated.
        base_url:
            URL to model provider.
        """
        super().__init__()
        self.model_id = model_id
        self.sampling_params = {"base_url": base_url or self._get_base_url()}
        sampling_run_params = {}
        if moderation:
            sampling_run_params["moderation"] = moderation
        self.sampling_run_params = sampling_run_params
        self._initialize()
        self._get_api_key()


    def _generate(self, **kwargs):
        response = ModelResponse()
        response_format = kwargs.pop("response_format")
        model_output = self._execute_model(**kwargs)
        response.set_response_type("image_generation")
        #base64.b64encode(f.read()).decode("utf-8")
        model_output.url

    async def _agenerate(self, **kwargs):
        response = ModelResponse()
        response_format = kwargs.pop("response_format")
        model_output = await self._aexecute_model(**kwargs)
        response.set_response_type("image_generation")
        #base64.b64encode(f.read()).decode("utf-8")
        model_output.url

    @model_retry
    def __call__(
        self,
        prompt: str,
        *,
        image: Optional[Union[str, List[str]]] = None,
        aspect_ratio: Optional[str] = None,
        mask: Optional[str] = None,
        response_format: Optional[Literal["url", "base64"]] = None,
        n: Optional[int] = 1,
        output_format: Optional[Literal["png", "webp"]] = "png",
        num_inference_steps: Optional[int] = None,
        disable_safety_checker: Optional[bool] = None,
        **kwargs
    ) -> ModelResponse:
        """Args:
        prompt:
            A text description of the desired image(s).
        image:
            The image(s) to edit. Can be a path, an url or base64 string.
        mask:
            An additional image whose fully transparent areas
            (e.g. where alpha is zero) indicate where image
            should be edited. If there are multiple images provided,
            the mask will be applied on the first image.
        response_format:
            Format in which images are returned.
        n:
            The number of images to generate.
        num_inference_steps:
            Number of denoising steps. 4 is recommended, and lower number
            of steps produce lower quality outputs, faster.            
        """
        generation_params = dotdict(
            prompt=prompt, num_outputs=n, output_format=output_format, 
            **kwargs, model=self.model_id,
        )

        if aspect_ratio:
            generation_params.aspect_ratio = aspect_ratio

        if num_inference_steps:
            generation_params.num_inference_steps = num_inference_steps

        if disable_safety_checker is not None:
            generation_params.disable_safety_checker = disable_safety_checker

        if response_format is not None:
            if response_format == "base64":
                response_format = "b64_json"
            generation_params.response_format = response_format

        inputs = self._prepare_inputs(image, mask)
        response = self._generate(**generation_params, **inputs)
        return response

    @model_retry
    async def acall(
        self,
        prompt: str,
        *,
        image: Optional[Union[str, List[str]]] = None,
        aspect_ratio: Optional[str] = None,
        mask: Optional[str] = None,
        response_format: Optional[Literal["url", "base64"]] = None,
        n: Optional[int] = 1,
        output_format: Optional[Literal["png", "webp"]] = "png",
        num_inference_steps: Optional[int] = None,
        disable_safety_checker: Optional[bool] = None,
        **kwargs
    ) -> ModelResponse:
        """Async version of __call__. Args:
        prompt:
            A text description of the desired image(s).
        image:
            The image(s) to edit. Can be a path, an url or base64 string.
        mask:
            An additional image whose fully transparent areas
            (e.g. where alpha is zero) indicate where image
            should be edited. If there are multiple images provided,
            the mask will be applied on the first image.
        response_format:
            Format in which images are returned.
        n:
            The number of images to generate.
        num_inference_steps:
            Number of denoising steps. 4 is recommended, and lower number
            of steps produce lower quality outputs, faster.
        """
        generation_params = dotdict(
            prompt=prompt, num_outputs=n, output_format=output_format,
            **kwargs, model=self.model_id,
        )

        if aspect_ratio:
            generation_params.aspect_ratio = aspect_ratio

        if num_inference_steps:
            generation_params.num_inference_steps = num_inference_steps

        if disable_safety_checker is not None:
            generation_params.disable_safety_checker = disable_safety_checker

        if response_format is not None:
            if response_format == "base64":
                response_format = "b64_json"
            generation_params.response_format = response_format

        inputs = self._prepare_inputs(image, mask)
        response = await self._agenerate(**generation_params, **inputs)
        return response
