from typing import Any, Dict, Mapping, Optional, Union

from msgflux.dotdict import dotdict
from msgflux.message import Message
from msgflux.models.gateway import ModelGateway
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.models.types import SpeechToTextModel
from msgflux.nn.modules.module import Module


class Transcriber(Module):
    """Transcriber is a Module type that uses language models to transcribe audios."""

    def __init__(
        self,
        name: str,
        model: Union[SpeechToTextModel, ModelGateway],
        *,
        stream: Optional[bool] = False,
        task_multimodal_inputs: Optional[Union[str, Dict[str, str]]] = None,
        response_mode: Optional[str] = "plain_response",
        response_template: Optional[str] = None,
        language: Optional[str] = None,
        response_format: Optional[str] = "text",
        timestamp_granularities: Optional[str] = None,
        prompt: Optional[str] = None,
        model_preference: Optional[str] = None,
    ):
        """Args:
        name:
            Transcriber name in snake case format.
        model:
            Transcriber Model client.
        task_multimodal_inputs:
            Fields of the Message object that will be the multimodal input
            to the task.
            !!! example.

                task_multimodal_inputs="audio.user" # direct path
                task_multimodal_inputs={"audio": "audio.user"} # dict-based path
        response_mode:
            What the response should be.
            * `plain_response` (default): Returns the final agent response directly.
            * other: Write on field in Message object.
        language:
            Spoken language acronym.
        response_format: How the model should format the output. Options:
            * text (default)
            * json
            * srt
            * verbose_json
            * vtt
        timestamp_granularities:
            Enable timestamp granularities.
            Requires `response_format=verbose_json`. Options:
            * word
            * segment
            * None (default)
        prompt:
            Useful for instructing the model to follow some transcript
            generation pattern.
        model_preference:
            Field of the Message object that will be the model preference.
            This is only valid if the model is of type ModelGateway.
        """
        super().__init__()
        self.set_name(name)
        self._set_language(language)
        self._set_model(model)
        self._set_prompt(prompt)
        self._set_model_preference(model_preference)
        self._set_response_format(response_format)
        self._set_response_mode(response_mode)
        self._set_response_template(response_template)
        self._set_stream(stream)
        self._set_task_multimodal_inputs(task_multimodal_inputs)
        self._set_timestamp_granularities(timestamp_granularities)

    def forward(
        self, message: Union[bytes, str, Dict[str, str], Message], **kwargs
    ) -> Union[str, Dict[str, str], Message, ModelStreamResponse]:
        inputs = self._prepare_task(message, **kwargs)
        model_response = self._execute_model(**inputs)
        response = self._process_model_response(model_response, message)
        return response

    def _execute_model(
        self, data: Union[str, bytes], model_preference: Optional[str] = None
    ) -> Union[ModelResponse, ModelStreamResponse]:
        model_execution_params = self._prepare_model_execution(data, model_preference)
        model_response = self.model(**model_execution_params)
        return model_response

    def _prepare_model_execution(
        self, data: Union[str, bytes], model_preference: Optional[str] = None
    ) -> Dict[str, Any]:
        model_execution_params = dotdict(
            {
                "data": data,
                "language": self.language,
                "response_format": self.response_format,
                "timestamp_granularities": self.timestamp_granularities,
                "prompt": self.prompt,
                "stream": self.stream,
            }
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

    def _prepare_task(
        self, message: Union[bytes, str, Dict[str, str], Message], **kwargs
    ) -> Dict[str, Union[bytes, str]]:
        data = self._process_task_multimodal_inputs(message)

        model_preference = kwargs.pop("model_preference", None)
        if model_preference is None and isinstance(message, Message):
            model_preference = self.get_model_preference_from_message(message)

        return {"data": data, "model_preference": model_preference}

    def _process_task_multimodal_inputs(
        self, message: Union[bytes, str, Dict[str, str], Message]
    ) -> bytes:
        if isinstance(message, Message):
            audio_content = self._extract_message_values(
                self.task_multimodal_inputs, message
            )
        else:
            audio_content = message

        if isinstance(audio_content, dict):
            audio = audio_content.get("audio", None)
            if audio:
                audio_content = audio
            else:
                raise ValueError(
                    "`task_multimodal_inputs` path based-on dict requires "
                    f"an `audio` key given {audio_content}"
                )

        return audio_content

    def inspect_model_execution_params(self, *args, **kwargs) -> Mapping[str, Any]:
        """Debug model input parameters."""
        inputs = self._prepare_task(*args, **kwargs)
        model_execution_params = self._prepare_model_execution(**inputs)
        return model_execution_params

    def _set_model(self, model: Union[SpeechToTextModel, ModelGateway]):
        if model.model_type == "speech_to_text":
            self.register_buffer("model", model)
        else:
            raise TypeError(
                f"`model` need be a `speech_to_text` model, given `{type(model)}`"
            )

    def _set_language(self, language: Optional[str] = None):
        if isinstance(language, str) or language is None:
            self.register_buffer("language", language)
        else:
            raise TypeError(
                f"`language` need be a `str` or `None` given `{type(language)}"
            )

    def _set_timestamp_granularities(self, timestamp_granularities: str):
        if isinstance(timestamp_granularities, str):
            supported_granularities = ["word", "segment"]
            if timestamp_granularities in supported_granularities:
                timestamp_granularities = [timestamp_granularities]
            else:
                raise ValueError(
                    f"`timestamp_granularities` can be {supported_granularities} "
                    f"given {timestamp_granularities}"
                )
        elif timestamp_granularities is not None:
            raise TypeError(
                "`timestamp_granularities` need be a `str` or `None` "
                f"given `{type(timestamp_granularities)}"
            )
        self.register_buffer("timestamp_granularities", timestamp_granularities)

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
