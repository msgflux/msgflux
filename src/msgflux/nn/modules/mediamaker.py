from typing import Any, Callable, Dict, Literal, Mapping, Optional, Union

from msgflux.dotdict import dotdict
from msgflux.message import Message
from msgflux.models.base import BaseModel
from msgflux.models.gateway import ModelGateway
from msgflux.models.response import ModelResponse
from msgflux.models.types import (    
    ImageTextTo3DModel,
    ImageTextToImageModel,
    TextTo3DModel,
    TextToImageModel,
    TextToVideoModel,
    VideoTextToAudioModel,
    VideoTextToVideoModel,
)
from msgflux.nn.modules.module import Module


MEDIA_MODEL_TYPES = Union[
    ModelGateway,
    ImageTextTo3DModel,
    ImageTextToImageModel,
    TextTo3DModel,
    TextToImageModel,
    TextToVideoModel,
    VideoTextToAudioModel,
    VideoTextToVideoModel,
]


class MediaMaker(Module):
    """MediaMaker is a Module type that uses generative
    models to create content.
    """

    def __init__(
        self,
        name: str,
        model: MEDIA_MODEL_TYPES,
        *,
        input_guardrail: Optional[Callable] = None,
        output_guardrail: Optional[Callable] = None,
        task_inputs: Optional[str] = None,
        task_multimodal_inputs: Optional[Dict[str, str]] = None,
        response_format: Optional[Literal["base64", "url"]] = None,
        response_mode: Optional[str] = "plain_response",
        negative_prompt: Optional[str] = None,
        fps: Optional[int] = None,
        duration_seconds: Optional[int] = None,
        aspect_ratio: Optional[str] = None,
        n: Optional[int] = None,
        execution_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Args:
        name:
            Designer name in snake case format.
        model:
            Designer Model client.
        input_guardrail:
            Guardrail to input.
        output_guardrail:
            Guardrail to output.
        task_inputs:
            Fields of the Message object that will be the input to the task.
        task_multimodal_inputs:
            Fields of the Message object that will be the multimodal input
            to the task.
        response_format:
            Data output format.
        response_mode:
            What the response should be.
            * `plain_response` (default): Returns the final agent response directly.
            * other: Write on field in Message object.
        negative_prompt:
            Instructions on what not to have.
        fps:
            Number of frames-per-secound in videos.
        duration_seconds:
            Video duration in secounds.
        n:
            Number of content to generate.
        aspect_ratio:
            Aspect ratio to vision content.
        execution_kwargs:
            Extra kwargs to model execution.
        """
        super().__init__()
        self.set_name(name)
        self._set_aspect_ratio(aspect_ratio)
        self._set_duration_seconds(duration_seconds)
        self._set_execution_kwargs(execution_kwargs)
        self._set_fps(fps)
        self._set_input_guardrail(input_guardrail)
        self._set_output_guardrail(output_guardrail)
        self._set_model(model)
        self._set_n(n)
        self._set_negative_prompt(negative_prompt)
        self._set_response_mode(response_mode)
        self._set_response_format(response_format)
        self._set_task_inputs(task_inputs)
        self._set_task_multimodal_inputs(task_multimodal_inputs)

    def forward(self, message: Union[str, Message], **kwargs) -> Union[str, Message]:
        inputs = self._prepare_task(message, **kwargs)
        model_response = self._execute_model(**inputs)
        response = self._process_model_response(model_response, message)
        return response

    def _execute_model(
        self,
        prompt: str,
        image: Optional[str] = None,
        mask: Optional[str] = None,
        model_preference: Optional[str] = None,
    ) -> ModelResponse:
        model_execution_params = self._prepare_model_execution(
            prompt, image, mask, model_preference
        )
        if self.guardrail is not None:
            self._execute_guardrail(model_execution_params)
        model_response = self.model(**model_execution_params)
        return model_response

    def _prepare_model_execution(
        self,
        prompt: str,
        image: Optional[str] = None,
        mask: Optional[str] = None,
        model_preference: Optional[str] = None,
    ) -> Dict[str, Any]:
        model_execution_params = self.execution_kwargs or dotdict()
        model_execution_params.prompt = prompt
        if image:
            model_execution_params.image = image
        if mask:
            model_execution_params.mask = mask
        if model_preference:
            model_execution_params.model_preference = model_preference
        if self.aspect_ratio:
            model_execution_params.aspect_ratio = self.aspect_ratio
        if self.duration_seconds:
            model_execution_params.duration_seconds = self.duration_seconds
        if self.fps:
            model_execution_params.fps = self.fps
        if self.n:
            model_execution_params.n = self.n
        if self.negative_prompt:
            model_execution_params.negative_prompt = self.negative_prompt
        return model_execution_params

    def _prepare_guardrail_execution(
        self, model_execution_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        prompt = model_execution_params.prompt
        image = model_execution_params.image
        if image is not None:
            messages = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                },
            ]
            data = {"data": messages}
        else:
            data = {"data": prompt}
        guardrail_params = data
        return guardrail_params

    def _process_model_response(
        self, model_response: ModelResponse, message: Union[str, Message]
    ) -> str:
        if model_response.response_type == "audio_generation":
            raw_response = self._extract_raw_response(model_response)
            response = self._prepare_response(raw_response, message)
            return response
        else:
            raise ValueError(
                f"Unsupported model response type `{model_response.response_type}`"
            )

    def _prepare_task(self, message: Union[str, Message], **kwargs) -> Dict[str, Any]:
        inputs = dotdict()

        if isinstance(message, Message):
            prompt = self._extract_message_values(self.task_inputs, message)
        else:
            prompt = message

        if prompt is None:
            raise ValueError(
                "`prompt` cannot be None, pass `message` as str or"
                "set `task_inputs` and pass a Message"
            )
        else:
            inputs.prompt = prompt

        model_preference = kwargs.pop("model_preference", None)
        if model_preference is None and isinstance(message, Message):
            model_preference = self.get_model_preference_from_message(message)

        if model_preference:
            inputs.model_preference = model_preference

        multimodal_content = self._process_task_multimodal_inputs(message, **kwargs)
        if multimodal_content:
            inputs.update(multimodal_content)

        return inputs

    def _process_task_multimodal_inputs(
        self, message: Union[str, Message, Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        """Processes multimodal image inputs."""
        task_multimodal_inputs = kwargs.pop("task_multimodal_inputs", None)
        if task_multimodal_inputs is None and isinstance(message, Message):
            task_multimodal_inputs = self._extract_message_values(
                self.task_multimodal_inputs, message
            )

        content = {}

        for media_source in ["image", "mask"]:
            data = task_multimodal_inputs.get(media_source, None)
            if data:
                content[media_source] = data

        return content

    def inspect_model_execution_params(self, *args, **kwargs) -> Mapping[str, Any]:
        """Debug model input parameters."""
        inputs = self._prepare_task(*args, **kwargs)
        model_execution_params = self._prepare_model_execution(**inputs)
        return model_execution_params

    def _set_model(self, model: Union[BaseModel, ModelGateway]):
        if isinstance(model, tuple(MEDIA_MODEL_TYPES)):
            self.register_buffer("model", model)
        else:
            raise TypeError(
                f"`model` need be a `{MEDIA_MODEL_TYPES!s}` "
                f"model, given `{type(model)}`"
            )

    def _set_response_format(self, response_format: Optional[str] = None):
        if isinstance(response_format, str) or response_format is None:
            self.register_buffer("response_format", response_format)
        else:
            raise TypeError(
                f"`response_format` need be a str or given `{type(response_format)}"
            )

    def _set_negative_prompt(self, negative_prompt: Optional[str] = None):
        if isinstance(negative_prompt, str) or negative_prompt is None:
            self.register_buffer("negative_prompt", negative_prompt)
        else:
            raise TypeError(
                "`negative_prompt` need be a str or None given "
                f"`{type(negative_prompt)}`"
            )

    def _set_fps(self, fps: Optional[int] = None):
        if isinstance(fps, int) or fps is None:
            self.register_buffer("fps", fps)
        else:
            raise TypeError(f"`fps` need be an int or None given `{type(fps)}`")

    def _set_duration_seconds(self, duration_seconds: Optional[int] = None):
        if isinstance(duration_seconds, int) or duration_seconds is None:
            self.register_buffer("duration_seconds", duration_seconds)
        else:
            raise TypeError(
                "`duration_seconds` need be an int or None given "
                f"`{type(duration_seconds)}`"
            )

    def _set_aspect_ratio(self, aspect_ratio: Optional[str] = None):
        if isinstance(aspect_ratio, str) or aspect_ratio is None:
            self.register_buffer("aspect_ratio", aspect_ratio)
        else:
            raise TypeError(
                "`aspect_ratio` need be an str or None given "
                f"`{type(aspect_ratio)}`"
            )

    def _set_n(self, n: Optional[int] = None):
        if isinstance(n, int) or n is None:
            self.register_buffer("n", n)
        else:
            raise TypeError(f"`n` need be an int or None given `{type(n)}`")
