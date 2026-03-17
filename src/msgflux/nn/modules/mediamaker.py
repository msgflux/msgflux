from typing import Any, Dict, Literal, Mapping, Optional, Union

from msgflux.auto import AutoParams
from msgflux.core.dotdict import dotdict
from msgflux.core.message import Message
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
from msgflux.nn.modules.generator import Generator
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


class MediaMaker(Module, metaclass=AutoParams):
    """MediaMaker is a Module type that uses generative
    models to create content.
    """

    def __init__(
        self,
        model: MEDIA_MODEL_TYPES,
        *,
        hooks: Optional[list] = None,
        message_fields: Optional[Dict[str, Any]] = None,
        response_format: Optional[Literal["base64", "url"]] = None,
        response_mode: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ):
        """Initialize the MediaMaker module.

        Args:
        model:
            MediaMaker Model client.
        hooks:
            List of Hook instances to register on the model.
        message_fields:
            Dictionary mapping Message field names to their paths in the Message object.
            Valid keys: "task_inputs", "task_multimodal_inputs"
            !!! example
                message_fields={
                    "task_inputs": "prompt.text",
                    "task_multimodal_inputs": {"image": "image.input"}
                }

            Field descriptions:
            - task_inputs: Field path for task input (str)
            - task_multimodal_inputs: Map datatypes to field paths (dict)
        response_format:
            Data output format.
        response_mode:
            Controls how the response is returned.
            * ``None`` (default): Returns the response directly.
            * ``"<path>"``: Writes to ``obj.<path>`` and returns ``None``
              (``dotdict`` or ``Message`` is mutated in place).
        negative_prompt:
            Instructions on what not to have.
        config:
            Dictionary with configuration options. Accepts any keys without validation.
            Common options: "fps", "duration_seconds", "aspect_ratio", "n"
            Any additional parameters will be passed directly to model execution.
            !!! example
                config={
                    "fps": 24,
                    "duration_seconds": 5,
                    "aspect_ratio": "16:9",
                    "n": 1
                }
        name:
            MediaMaker name in snake case format.
        """
        super().__init__()
        self._set_model(model)
        self._set_hooks(hooks)
        self._set_negative_prompt(negative_prompt)
        self._set_response_mode(response_mode)
        self._set_response_format(response_format)
        self._set_message_fields(message_fields)
        self._set_config(config)
        if name:
            self.set_name(name)

    def forward(self, message: Union[str, Message], **kwargs) -> Union[str, Message]:
        """Execute the media maker with the given message.

        Args:
            message: The input message, which can be:
                - str: Direct prompt for media generation
                - Message: Message object with fields mapped via message_fields
            **kwargs: Runtime overrides for message_fields. Can include:
                - task_inputs: Override field path or direct value
                - task_multimodal_inputs: Override multimodal inputs
                  (e.g., {"image": "path"})

        Returns:
            Generated media content (str or Message depending on response_mode).
        """
        inputs = self._prepare_task(message, **kwargs)
        model_response = self._execute_model(**inputs)
        response = self._process_model_response(model_response, message)
        return response

    async def aforward(
        self, message: Union[str, Message], **kwargs
    ) -> Union[str, Message]:
        """Async version of forward. Execute the media maker asynchronously."""
        inputs = self._prepare_task(message, **kwargs)
        model_response = await self._aexecute_model(**inputs)
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
        model_response = self.generator(**model_execution_params)
        return model_response

    async def _aexecute_model(
        self,
        prompt: str,
        image: Optional[str] = None,
        mask: Optional[str] = None,
        model_preference: Optional[str] = None,
    ) -> ModelResponse:
        model_execution_params = self._prepare_model_execution(
            prompt, image, mask, model_preference
        )
        model_response = await self.generator.acall(**model_execution_params)
        return model_response

    def _prepare_model_execution(
        self,
        prompt: str,
        image: Optional[str] = None,
        mask: Optional[str] = None,
        model_preference: Optional[str] = None,
    ) -> Dict[str, Any]:
        model_execution_params = dotdict(self.config) if self.config else dotdict()
        model_execution_params.prompt = prompt
        if image:
            model_execution_params.image = image
        if mask:
            model_execution_params.mask = mask
        if model_preference:
            model_execution_params.model_preference = model_preference
        if self.negative_prompt:
            model_execution_params.negative_prompt = self.negative_prompt
        return model_execution_params

    def _process_model_response(
        self, model_response: ModelResponse, message: Union[str, Message]
    ) -> str:
        raw_response = self._extract_raw_response(model_response)
        response = self._prepare_response(raw_response, message)
        return response

    def _prepare_task(self, message: Union[str, Message], **kwargs) -> Dict[str, Any]:
        inputs = dotdict()

        if isinstance(message, dotdict):
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
        if model_preference is None and isinstance(message, dotdict):
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
        if task_multimodal_inputs is None and isinstance(message, dotdict):
            task_multimodal_inputs = self._extract_message_values(
                self.task_multimodal_inputs, message
            )

        content = {}

        if task_multimodal_inputs:
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
        if isinstance(model, (BaseModel, ModelGateway)):
            self.generator = Generator(model)
        else:
            raise TypeError(
                f"`model` need be a `BaseModel` or `ModelGateway` "
                f"model, given `{type(model)}`"
            )

    @property
    def model(self):
        """Access underlying model."""
        return self.generator.model

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

    def _set_config(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            self.register_buffer("config", {})
            return

        if not isinstance(config, dict):
            raise TypeError(f"`config` must be a dict or None, given `{type(config)}`")

        self.register_buffer("config", config.copy())
