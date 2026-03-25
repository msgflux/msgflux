from typing import Any, Dict, Mapping, Optional, Union

from msgflux.auto import AutoParams
from msgflux.core.dotdict import dotdict
from msgflux.core.message import Message
from msgflux.models.base import BaseModel
from msgflux.models.gateway import ModelGateway
from msgflux.nn.modules.generator import Generator
from msgflux.nn.modules.module import Module


class Predictor(Module, metaclass=AutoParams):
    """Predictor is the most generic Module type — it feeds data to a model
    and returns predictions.

    Works with any msgflux model (classifiers, regressors, detectors, etc.)
    or custom models that inherit from ``BaseModel``.
    """

    # Configure AutoParams to use class name as 'name' parameter
    _autoparams_use_classname_for = "name"

    def __init__(
        self,
        model: Union[BaseModel, ModelGateway],
        *,
        hooks: Optional[list] = None,
        message_fields: Optional[Dict[str, Any]] = None,
        response_mode: Optional[str] = None,
        templates: Optional[Dict[str, str]] = None,
        config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ):
        """Initialize the Predictor module.

        Args:
        model:
            Model client. Any msgflux ``BaseModel`` or ``ModelGateway``.
        hooks:
            List of Hook instances to register on the module.
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
            - model_preference: Field path for model preference (str, only valid
              with ModelGateway)
        response_mode:
            Controls how the response is returned.
            * ``None`` (default): Returns the response directly.
            * ``"<path>"``: Writes to ``obj.<path>`` and returns ``None``
              (``dotdict`` or ``Message`` is mutated in place).
        templates:
            Dictionary mapping template types to Jinja template strings.
            Valid keys: "response"
            !!! example
                templates={"response": "Label: {{ prediction }}"}
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
        self._set_hooks(hooks)
        self._set_message_fields(message_fields)
        self._set_response_mode(response_mode)
        self._set_templates(templates)
        self._set_config(config)
        if name:
            self.set_name(name)

    def forward(self, message: Union[Any, Message], **kwargs) -> Any:
        """Execute the predictor with the given message.

        Args:
            message: The input message, which can be:
                - Any: Direct data input for prediction (text, image, array, etc.)
                - Message: Message object with fields mapped via message_fields
            **kwargs: Runtime overrides for message_fields. Can include:
                - task_inputs: Override field path or direct value
                - model_preference: Override model preference

        Returns:
            Prediction results (type depends on model and response_mode).
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

    def _execute_model(self, data: Any, model_preference: Optional[str] = None) -> Any:
        model_execution_params = self._prepare_model_execution(data, model_preference)
        model_response = self.generator(**model_execution_params)
        return model_response

    async def _aexecute_model(
        self, data: Any, model_preference: Optional[str] = None
    ) -> Any:
        model_execution_params = self._prepare_model_execution(data, model_preference)
        model_response = await self.generator.acall(**model_execution_params)
        return model_response

    def _prepare_model_execution(
        self, data: Any, model_preference: Optional[str] = None
    ) -> Dict[str, Any]:
        model_execution_params = dotdict(self.config) if self.config else dotdict()
        model_execution_params.data = data
        if isinstance(self.model, ModelGateway) and model_preference is not None:
            model_execution_params.model_preference = model_preference
        return model_execution_params

    def _process_model_response(
        self, model_response: Any, message: Union[Any, Message]
    ) -> Any:
        raw_response = self._extract_raw_response(model_response)
        response = self._prepare_response(raw_response, message)
        return response

    def _prepare_task(self, message: Union[Any, Message], **kwargs) -> Dict[str, Any]:
        inputs = dotdict()

        if isinstance(message, dotdict):
            data = self._extract_message_values(self.task_inputs, message)
        else:
            data = message

        inputs.data = data

        model_preference = kwargs.pop("model_preference", None)
        if model_preference is None and isinstance(message, dotdict):
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
            self.generator = Generator(model)
        else:
            raise TypeError(
                f"`model` must be a `BaseModel` or `ModelGateway`, "
                f"given `{type(model)}`"
            )

    @property
    def model(self):
        """Access underlying model."""
        return self.generator.model

    def _set_config(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            self.register_buffer("config", {})
            return

        if not isinstance(config, dict):
            raise TypeError(f"`config` must be a dict or None, given `{type(config)}`")

        self.register_buffer("config", config.copy())

    def _set_message_fields(self, message_fields: Optional[Dict[str, Any]] = None):
        """Set message field mappings.

        Args:
            message_fields: Dictionary mapping field names to their values.
                Valid keys: "task_inputs", "model_preference"

        Raises:
            TypeError: If message_fields is not a dict or None
            ValueError: If invalid keys are provided
        """
        valid_keys = {"task_inputs", "model_preference"}

        if message_fields is None:
            self._set_task_inputs(None)
            self._set_model_preference(None)
            return

        if not isinstance(message_fields, dict):
            raise TypeError(
                f"`message_fields` must be a dict or None, got `{type(message_fields)}`"
            )

        invalid_keys = set(message_fields.keys()) - valid_keys
        if invalid_keys:
            raise ValueError(
                f"Invalid keys in message_fields: {invalid_keys}. "
                f"Valid keys are: {valid_keys}"
            )

        self._set_task_inputs(message_fields.get("task_inputs"))
        self._set_model_preference(message_fields.get("model_preference"))

    def _set_task_inputs(self, task_inputs: Optional[str] = None):
        """Set task_inputs field mapping."""
        if isinstance(task_inputs, str) or task_inputs is None:
            self.register_buffer("task_inputs", task_inputs)
        else:
            raise TypeError(
                f"`task_inputs` requires a string or None, given `{type(task_inputs)}`"
            )

    def _set_model_preference(self, model_preference: Optional[str] = None):
        """Set model_preference field mapping."""
        if isinstance(model_preference, str) or model_preference is None:
            self.register_buffer("model_preference", model_preference)
        else:
            raise TypeError(
                f"`model_preference` requires a string or None, "
                f"got `{type(model_preference)}`"
            )
