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
        model: Union[BaseModel, ModelGateway],
        *,
        message_fields: Optional[Dict[str, Any]] = None,
        response_mode: Optional[str] = "plain_response",
        response_template: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ):
        """Args:
        model:
            Predictor Model client.
        message_fields:
            Dictionary mapping Message field names to their paths in the Message object.
            Valid keys: "task_inputs", "model_preference"
            !!! example
                message_fields={
                    "task_inputs": "data.input",
                    "model_preference": "model.preference"
                }

            Field descriptions:
            - task_inputs: Field path for task input (str)
            - model_preference: Field path for model preference (str, only valid with ModelGateway)
        response_mode:
            What the response should be.
            * `plain_response` (default): Returns the final agent response directly.
            * other: Write on field in Message object.
        response_template:
            A Jinja template to format response.
        config:
            Dictionary with configuration options. Accepts any keys without validation.
            All parameters will be passed directly to model execution.
            !!! example
                config={"temperature": 0.7, "top_k": 50}
        name:
            Predictor name in snake case format.
        """
        super().__init__()
        self._set_model(model)
        self._set_message_fields(message_fields)
        self._set_response_mode(response_mode)
        self._set_response_template(response_template)
        self._set_config(config)
        if name:
            self.set_name(name)

    def forward(self, message: Union[Any, Message], **kwargs) -> Any:
        """Execute the predictor with the given message.

        Args:
            message: The input message, which can be:
                - Any: Direct data input for prediction (text, image, audio, etc.)
                - Message: Message object with fields mapped via message_fields
            **kwargs: Runtime overrides for message_fields. Can include:
                - task_inputs: Override field path or direct value
                - model_preference: Override model preference

        Returns:
            Prediction results (type depends on model and response_mode)
        """
        inputs = self._prepare_task(message, **kwargs)
        model_response = self._execute_model(**inputs)
        response = self._process_model_response(model_response, message)
        return response

    async def aforward(self, message: Union[Any, Message], **kwargs) -> Any:
        """Async version of forward. Execute the predictor asynchronously."""
        inputs = self._prepare_task(message, **kwargs)
        model_response = await self._aexecute_model(**inputs)
        response = self._process_model_response(model_response, message)
        return response

    def _execute_model(
        self, data: Any, model_preference: Optional[str] = None
    ) -> ModelResponse:
        model_execution_params = self._prepare_model_execution(data, model_preference)
        model_response = self.model(**model_execution_params)
        return model_response

    async def _aexecute_model(
        self, data: Any, model_preference: Optional[str] = None
    ) -> ModelResponse:
        model_execution_params = self._prepare_model_execution(data, model_preference)
        model_response = await self.model.acall(**model_execution_params)
        return model_response

    def _prepare_model_execution(
        self, data: Any, model_preference: Optional[str] = None
    ) -> Dict[str, Any]:
        model_execution_params = dotdict(self.config) if self.config else dotdict()
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

    def _set_config(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            self.register_buffer("config", {})
            return

        if not isinstance(config, dict):
            raise TypeError(f"`config` must be a dict or None, given `{type(config)}`")

        self.register_buffer("config", config.copy())
