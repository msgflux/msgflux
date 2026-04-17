import asyncio
from functools import partial
from typing import Any, Dict, Mapping, Optional, Union

from msgflux.auto import AutoParams
from msgflux.core.dotdict import dotdict
from msgflux.core.message import Message
from msgflux.models import Model
from msgflux.models.gateway import ModelGateway
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.models.types import SpeechToTextModel
from msgflux.nn.modules.generator import Generator
from msgflux.nn.modules.module import Module


class Transcriber(Module, metaclass=AutoParams):
    """Transcriber is a Module type that uses language models to transcribe audios."""

    # Configure AutoParams to use class name as 'name' parameter
    _autoparams_use_classname_for = "name"

    def __init__(
        self,
        model: Union[SpeechToTextModel, ModelGateway, str],
        *,
        message_fields: Optional[Dict[str, Any]] = None,
        response_mode: Optional[str] = None,
        response_template: Optional[str] = None,
        response_format: Optional[str] = "text",
        prompt: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ):
        """Initialize the Transcriber module.

        Args:
        model:
            Transcriber model client. Accepts a `SpeechToTextModel`,
            `ModelGateway`, or a shorthand string in the form
            ``"provider/model-id"`` (e.g. ``"openai/whisper-1"``).
            When a string is provided, `Model.speech_to_text` is called
            internally with no extra configuration.
        message_fields:
            Dictionary mapping Message field names to their paths in the Message object.
            Valid keys: "task_multimodal", "model_preference"
            !!! example
                message_fields={
                    "task_multimodal": "audio.user",
                    # or dict-based: "task_multimodal": {"audio": "audio.user"}
                    "model_preference": "model.preference"
                }

            Field descriptions:
            - task_multimodal: Field path for audio input (str or dict)
            - model_preference: Field path for model preference (str, only valid
              with ModelGateway)
        response_mode:
            Controls how the response is returned.
            * ``None`` (default): Returns the response directly.
            * ``"<path>"``: Writes to ``obj.<path>`` and returns ``None``
              (``dotdict`` or ``Message`` is mutated in place).
        response_format: How the model should format the output. Options:
            * text (default)
            * json
            * srt
            * verbose_json
            * vtt
        prompt:
            Useful for instructing the model to follow some transcript
            generation pattern.
        config:
            Dictionary with configuration options. Accepts any keys without validation.
            Common options: "language", "stream", "timestamp_granularities"
            !!! example
                config={
                    "language": "en",
                    "stream": False,
                    "timestamp_granularities": "word"
                }

            Configuration options:
            - language: Spoken language acronym (str)
            - stream: Transmit response on-the-fly (bool)
            - timestamp_granularities: Enable timestamp granularities - "word",
              "segment", or None
              (requires response_format=verbose_json)
        name:
            Transcriber name in snake case format.
        """
        super().__init__()
        self._set_model(model)
        self._set_prompt(prompt)
        self._set_message_fields(message_fields)
        self._set_response_format(response_format)
        self._set_response_mode(response_mode)
        if response_template:
            self.register_buffer("response_template", response_template)
        self._set_config(config)
        if name:
            self.set_name(name)

    def forward(
        self, message: Union[bytes, str, Dict[str, str], Message], **kwargs: Any
    ) -> Union[str, Dict[str, str], Message, ModelStreamResponse]:
        """Execute the transcriber with the given message.

        Args:
            message: The input message, which can be:
                - bytes: Direct audio bytes to transcribe
                - str: Audio file path or URL
                - dict: Audio input as dictionary
                - Message: Message object with fields mapped via message_fields
            **kwargs: Runtime overrides for message_fields. Can include:
                - task_multimodal: Override multimodal inputs
                  (e.g., "audio.path" or {"audio": "audio.path"})
                - model_preference: Override model preference

        Returns:
            Transcribed text (str, dict, Message, or ModelStreamResponse depending
            on configuration).
        """
        inputs = self._prepare_inputs(message, **kwargs)
        model_response = self._execute_model(**inputs)
        response = self._process_model_response(model_response, message)
        return response

    async def aforward(
        self, message: Union[bytes, str, Dict[str, str], Message], **kwargs: Any
    ) -> Union[str, Dict[str, str], Message, ModelStreamResponse]:
        """Async version of forward. Execute the transcriber asynchronously."""
        inputs = await self._aprepare_inputs(message, **kwargs)
        model_response = await self._aexecute_model(**inputs)
        response = await self._aprocess_model_response(model_response, message)
        return response

    def _execute_model(
        self, data: Union[str, bytes], model_preference: Optional[str] = None
    ) -> Union[ModelResponse, ModelStreamResponse]:
        model_execution_params = self._prepare_model_execution(data, model_preference)
        model_response = self.generator(**model_execution_params)
        return model_response

    async def _aexecute_model(
        self, data: Union[str, bytes], model_preference: Optional[str] = None
    ) -> Union[ModelResponse, ModelStreamResponse]:
        model_execution_params = self._prepare_model_execution(data, model_preference)
        model_response = await self.generator.acall(**model_execution_params)
        return model_response

    def _prepare_model_execution(
        self, data: Union[str, bytes], model_preference: Optional[str] = None
    ) -> Dict[str, Any]:
        model_execution_params = dotdict(
            data=data,
            language=self.config.get("language"),
            response_format=self.response_format,
            timestamp_granularities=self.config.get("timestamp_granularities"),
            prompt=self.prompt,
            stream=self.config.get("stream", False),
        )
        if isinstance(self.model, ModelGateway) and model_preference is not None:
            model_execution_params.model_preference = model_preference
        return model_execution_params

    def _process_model_response(
        self,
        model_response: Union[ModelResponse, ModelStreamResponse],
        message: Union[str, Message],
    ) -> Union[str, Dict[str, str], Message, ModelStreamResponse]:
        if model_response.response_type == "transcript":
            raw_response = self._extract_raw_response(model_response)
            response = self._prepare_response(raw_response, message)
            return response
        else:
            raise ValueError(
                f"Unsupported model response type `{model_response.response_type}`"
            )

    async def _aprocess_model_response(
        self,
        model_response: Union[ModelResponse, ModelStreamResponse],
        message: Union[str, Message],
    ) -> Union[str, Dict[str, str], Message, ModelStreamResponse]:
        """Async version of _process_model_response."""
        if model_response.response_type == "transcript":
            raw_response = self._extract_raw_response(model_response)
            response = self._prepare_response(raw_response, message)
            return response
        else:
            raise ValueError(
                f"Unsupported model response type `{model_response.response_type}`"
            )

    def _prepare_inputs(
        self, message: Union[bytes, str, Dict[str, str], Message], **kwargs
    ) -> Dict[str, Union[bytes, str]]:
        data = self._process_task_multimodal(message)

        model_preference = kwargs.pop("model_preference", None)
        if model_preference is None and isinstance(message, dotdict):
            model_preference = self.get_model_preference_from_message(message)

        return {"data": data, "model_preference": model_preference}

    async def _aprepare_inputs(
        self, message: Union[bytes, str, Dict[str, str], Message], **kwargs
    ) -> Dict[str, Union[bytes, str]]:
        """Async version of _prepare_inputs.

        Executes _prepare_inputs in an executor to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        func = partial(self._prepare_inputs, message, **kwargs)
        return await loop.run_in_executor(None, func)

    def _process_task_multimodal(
        self, message: Union[bytes, str, Dict[str, str], Message]
    ) -> bytes:
        if isinstance(message, dotdict):
            audio_content = self._extract_message_values(self.task_multimodal, message)
        else:
            audio_content = message

        if isinstance(audio_content, dict):
            audio = audio_content.get("audio", None)
            if audio:
                audio_content = audio
            else:
                raise ValueError(
                    "`task_multimodal` path based-on dict requires "
                    f"an `audio` key given {audio_content}"
                )

        return audio_content

    def inspect_model_execution_params(self, *args, **kwargs) -> Mapping[str, Any]:
        """Debug model input parameters."""
        inputs = self._prepare_inputs(*args, **kwargs)
        model_execution_params = self._prepare_model_execution(**inputs)
        return model_execution_params

    def _set_model(self, model: Union[SpeechToTextModel, ModelGateway, str]):
        if isinstance(model, str):
            model = Model.speech_to_text(model)
        if model.model_type == "speech_to_text":
            self.generator = Generator(model)
        else:
            raise TypeError(
                f"`model` need be a `speech_to_text` model, given `{type(model)}`"
            )

    @property
    def model(self):
        """Access underlying model."""
        return self.generator.model

    @model.setter
    def model(self, value: Union[SpeechToTextModel, ModelGateway, str]):
        self._set_model(value)

    def _set_config(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            self.register_buffer("config", {})
            return

        if not isinstance(config, dict):
            raise TypeError(f"`config` must be a dict or None, given `{type(config)}`")

        self.register_buffer("config", config.copy())

    def _set_response_format(self, response_format: str):
        supported_formats = ["json", "text", "srt", "verbose_json", "vtt"]
        if isinstance(response_format, str):
            if response_format in supported_formats:
                self.register_buffer("response_format", response_format)
            else:
                raise ValueError(
                    f"`response_format` can be `{supported_formats}` "
                    f"given `{response_format}"
                )
        else:
            raise TypeError(
                f"`response_format` need be a str or given `{type(response_format)}"
            )
