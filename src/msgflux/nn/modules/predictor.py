from typing import Any, Dict, Mapping, Optional, Union

from msgflux.dotdict import dotdict
from msgflux.message import Message
from msgflux.models.base import BaseModel
from msgflux.models.gateway import ModelGateway
from msgflux.models.response import ModelResponse
from msgflux.nn.modules.module import Module


class Predictor(Module):
    """Predictor is a generic Module type that uses Classifier, Regressors,
    Detectors and Segmenters to generate insights above data.
    """

    def __init__(
        self,
        name: str,
        model: Union[BaseModel, ModelGateway],
        *,
        task_inputs: Optional[str] = None,
        response_mode: Optional[str] = "plain_response",
        response_template: Optional[str] = None,
        model_preference: Optional[str] = None,
        execution_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Args:
        name:
            Predictor name in snake case format.
        model:
            Predictor Model client.
        task_inputs:
            Fields of the Message object that will be the input to the task.
        response_mode:
            What the response should be.
            * `plain_response` (default): Returns the final agent response directly.
            * other: Write on field in Message object.
        response_template:
            A Jinja template to format response.
        model_preference:
            Fields of the Message object that will be the model preference.
            This is only valid if the model is of type ModelGateway.
        execution_kwargs:
            Extra kwargs to model execution.
        """
        super().__init__()
        self.set_name(name)
        self._set_model(model)
        self._set_execution_kwargs(execution_kwargs)
        self._set_model_preference(model_preference)
        self._set_response_mode(response_mode)
        self._set_response_template(response_template)
        self._set_task_inputs(task_inputs)

    def forward(self, message: Union[Any, Message], **kwargs) -> Any:
        inputs = self._prepare_task(message, **kwargs)
        model_response = self._execute_model(**inputs)
        response = self._process_model_response(model_response, message)
        return response

    def _execute_model(
        self, data: Any, model_preference: Optional[str] = None
    ) -> ModelResponse:
        model_execution_params = self._prepare_model_execution(data, model_preference)
        model_response = self.model(**model_execution_params)
        return model_response

    def _prepare_model_execution(
        self, data: Any, model_preference: Optional[str] = None
    ) -> Dict[str, Any]:
        model_execution_params = dotdict(self.execution_kwargs or {})
        model_execution_params.data = data
        if model_preference:
            model_execution_params.model_preference = model_preference
        return model_execution_params

    def _process_model_response(
        self, model_response: ModelResponse, message: Union[Any, Message]
    ) -> Any:
        if model_response.response_type == "audio_generation":
            raw_response = self._extract_raw_response(model_response)
            response = self._prepare_response(raw_response, message)
            return response
        else:
            raise ValueError(
                f"Unsupported model response type `{model_response.response_type}`"
            )

    def _prepare_task(self, message: Union[Any, Message], **kwargs) -> Dict[str, Any]:
        inputs = dotdict()

        if isinstance(message, Message):
            data = self._extract_message_values(self.task_inputs, message)
        else:
            data = message

        inputs.data = data

        model_preference = kwargs.pop("model_preference", None)
        if model_preference is None and isinstance(message, Message):
            model_preference = self.get_model_preference_from_message(message)

        if model_preference:
            inputs.model_preference = model_preference

        return inputs

    def inspect_model_execution_params(self, *args, **kwargs) -> Mapping[str, Any]:
        """Debug model input parameters."""
        inputs = self._prepare_task(*args, **kwargs)
        model_execution_params = self._prepare_model_execution(**inputs)
        return model_execution_params

    def _set_model(self, model: Union[BaseModel, ModelGateway]):
        if isinstance(model, (BaseModel, ModelGateway)):
            self.register_buffer("model", model)
        else:
            raise TypeError(
                f"`model` need be a `BaseModel` model, given `{type(model)}`"
            )
