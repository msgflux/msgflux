from typing import Any, Callable, Dict, Literal, Mapping, Optional, Union

from msgflux.dotdict import dotdict
from msgflux.message import Message
from msgflux.models.gateway import ModelGateway
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.models.types import TextToSpeechModel
from msgflux.nn.modules.module import Module


class Speaker(Module):
    """Speaker is a Module type that uses language models to transform text in speak."""

    def __init__(
        self,
        name: str,
        model: Union[TextToSpeechModel, ModelGateway],
        *,
        input_guardrail: Optional[Callable] = None,
        stream: Optional[bool] = False,
        task_inputs: Optional[str] = None,
        response_mode: Optional[str] = "plain_response",
        response_format: Optional[
            Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]
        ] = "opus",
        prompt: Optional[str] = None,
    ):
        """Args:
        name:
            Transcriber name in snake case format.
        model:
            Transcriber Model client.
        task_multimodal_inputs:
            Fields of the Message object that will be the multimodal input
            to the task.
        response_mode:
            What the response should be.
            * `plain_response` (default): Returns the final agent response directly.
            * other: Write on field in Message object.
        response_format:
            The format to audio in.
        prompt:
            Useful for instructing the model to follow some speak generation pattern.
        """
        super().__init__()
        self.set_name(name)
        self._set_input_guardrail(input_guardrail)
        self._set_model(model)
        self._set_prompt(prompt)
        self._set_response_format(response_format)
        self._set_response_mode(response_mode)
        self._set_stream(stream)
        self._set_task_inputs(task_inputs)

    def forward(
        self, message: Union[str, Message], **kwargs
    ) -> Union[bytes, ModelStreamResponse]:
        inputs = self._prepare_task(message, **kwargs)
        model_response = self._execute_model(**inputs)
        response = self._process_model_response(model_response, message)
        return response

    def _execute_model(
        self, data: str, model_preference: Optional[str] = None
    ) -> Union[ModelResponse, ModelStreamResponse]:
        model_execution_params = self._prepare_model_execution(data, model_preference)
        if self.input_guardrail is not None:
            self._execute_input_guardrail(model_execution_params)
        model_response = self.model(**model_execution_params)
        return model_response

    def _prepare_model_execution(
        self, data: str, model_preference: Optional[str] = None
    ) -> Dict[str, Union[str, bool]]:
        model_execution_params = dotdict(
            {
                "data": data,
                "response_format": self.response_format,
                "prompt": self.prompt,
            }
        )
        if self.stream:
            model_execution_params.stream = self.stream
        if isinstance(self.model, ModelGateway) and model_preference is not None:
            model_execution_params.model_preference = model_preference
        return model_execution_params

    def _prepare_guardrail_execution(
        self, model_execution_params: Dict[str, Union[str, bool]]
    ) -> Dict[str, str]:
        guardrail_params = {"data": model_execution_params.data}
        return guardrail_params

    def _process_model_response(
        self,
        model_response: Union[ModelResponse, ModelStreamResponse],
        message: Union[str, Message],
    ) -> Union[bytes, Message, ModelStreamResponse]:
        if model_response.response_type == "audio_generation":
            raw_response = self._extract_raw_response(model_response)
            response = self._prepare_response(raw_response, message)
            return response
        else:
            raise ValueError(
                f"Unsupported model response type `{model_response.response_type}`"
            )

    def _prepare_task(self, message: Union[str, Message], **kwargs) -> Dict[str, str]:
        if isinstance(message, Message):
            data = self._extract_message_values(self.task_inputs, message)
            if data is None:
                raise ValueError(f"No text found in paths: `{self.task_inputs}`")
        elif isinstance(message, str):
            data = message
        else:
            raise ValueError(f"Unsupported message type: `{type(message)}`")

        model_preference = kwargs.pop("model_preference", None)
        if model_preference is None and isinstance(message, Message):
            model_preference = self.get_model_preference_from_message(message)

        return {"data": data, "model_preference": model_preference}

    def inspect_model_execution_params(self, *args, **kwargs) -> Mapping[str, Any]:
        """Debug model input parameters."""
        inputs = self._prepare_task(*args, **kwargs)
        model_execution_params = self._prepare_model_execution(**inputs)
        return model_execution_params

    def _set_model(self, model: Union[TextToSpeechModel, ModelGateway]):
        if model.model_type == "text_to_speech":
            self.register_buffer("model", model)
        else:
            raise TypeError(
                f"`model` need be a `text_to_speech` model, given `{type(model)}`"
            )

    def _set_response_format(self, response_format: str):
        supported_formats = ["mp3", "opus", "aac", "flac", "wav", "pcm"]
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
