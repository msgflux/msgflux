from functools import partial
from typing import Any, Dict, List, Mapping, Optional, Union

from msgflux.dotdict import dotdict
from msgflux.message import Message
from msgflux.models.gateway import ModelGateway
from msgflux.models.types import (
    AudioEmbedderModel,
    ImageEmbedderModel,
    TextEmbedderModel,
)
from msgflux.nn import functional as F
from msgflux.nn.modules.module import Module

EMBEDDER_MODELS = Union[
    AudioEmbedderModel, ImageEmbedderModel, TextEmbedderModel, ModelGateway
]


class Embedder(Module):
    """Embedder is a Module that converts data into vector embeddings.

    Supports both batch and non-batch models transparently:
        - Batch models: Passes all data at once for efficient processing
        - Non-batch models: Uses parallel execution via F.map_gather
    """

    def __init__(
        self,
        model: EMBEDDER_MODELS,
        *,
        message_fields: Optional[Dict[str, Any]] = None,
        response_mode: Optional[str] = "plain_response",
        config: Optional[Dict[str, Any]] = None,
    ):
        """Args:
        model:
            Embedding model client (supports batch or single).
        message_fields:
            Dictionary mapping Message field names to paths.
            Valid keys: "task_inputs", "model_preference"
            !!! example
                message_fields={
                    "task_inputs": "texts",
                    "model_preference": "model.preference"
                }

            Field descriptions:
            - task_inputs: Field path for input data (str, list of str, or other data types)
            - model_preference: Field path for model preference (str, only valid with ModelGateway)
        response_mode:
            What the response should be.
            * `plain_response` (default): Returns embeddings directly.
            * other: Write on field in Message object.
        config:
            Dictionary with configuration options. Accepts any keys without validation.
            Additional parameters will be passed directly to model execution.
            !!! example
                config={"normalize": True, "truncate": True}
        """
        super().__init__()
        self._set_model(model)
        self._set_message_fields(message_fields)
        self._set_response_mode(response_mode)
        self._set_config(config)

    def forward(
        self, message: Union[str, List[str], Message], **kwargs
    ) -> Union[List[float], List[List[float]], Message]:
        """Execute the embedder with the given message.

        Args:
            message: The input message, which can be:
                - str: Single text to embed
                - List[str]: Multiple texts to embed
                - Message: Message object with fields mapped via message_fields
            **kwargs: Runtime overrides for message_fields. Can include:
                - task_inputs: Override field path or direct value
                - model_preference: Override model preference

        Returns:
            Embeddings as list(s) of floats, or Message depending on response_mode.
        """
        inputs = self._prepare_task(message, **kwargs)
        embeddings = self._execute_model(**inputs)
        response = self._prepare_response(embeddings, message)
        return response

    async def aforward(
        self, message: Union[str, List[str], Message], **kwargs
    ) -> Union[List[float], List[List[float]], Message]:
        """Async version of forward. Execute the embedder asynchronously."""
        inputs = self._prepare_task(message, **kwargs)
        embeddings = await self._aexecute_model(**inputs)
        response = self._prepare_response(embeddings, message)
        return response

    def _execute_model(
        self, data: Union[str, List[str]], model_preference: Optional[str] = None
    ) -> Union[List[float], List[List[float]]]:
        """Execute embedding with batch support.

        Uses model.batch_support to determine execution strategy:
        - batch_support=True: Pass all data at once
        - batch_support=False: Use F.map_gather for parallel execution
        """
        # Normalize input to list
        is_single = isinstance(data, str)
        data_list = [data] if is_single else data

        # Check if model supports batch processing
        if self.model.batch_support or len(data_list) == 1:
            # Batch mode: pass all data at once
            model_execution_params = self._prepare_model_execution(
                data_list, model_preference
            )
            model_response = self.model(**model_execution_params)
            embeddings = self._extract_raw_response(model_response)

            # Ensure list format
            if not isinstance(embeddings, list):
                embeddings = [embeddings]
        else:
            # Non-batch mode: use parallel execution via F.map_gather
            prepare_execution = partial(
                self._prepare_model_execution, model_preference=model_preference
            )
            distributed_params = list(map(prepare_execution, data_list))
            # map_gather requires args_list (list of tuples) - use empty tuples since we only have kwargs
            args_list = [()] * len(data_list)
            responses = F.map_gather(
                self.model, args_list=args_list, kwargs_list=distributed_params
            )
            embeddings = [
                self._extract_raw_response(response) for response in responses
            ]

        # Return single embedding if input was single
        if is_single:
            # Check if embeddings is already a flat list (single embedding vector)
            # or if it's a list containing a single embedding vector
            if len(embeddings) > 0 and isinstance(embeddings[0], list):
                return embeddings[0]
            return embeddings
        return embeddings

    async def _aexecute_model(
        self, data: Union[str, List[str]], model_preference: Optional[str] = None
    ) -> Union[List[float], List[List[float]]]:
        """Async execute embedding with batch support."""
        # Normalize input to list
        is_single = isinstance(data, str)
        data_list = [data] if is_single else data

        # Check if model supports batch processing
        if self.model.batch_support or len(data_list) == 1:
            # Batch mode: pass all data at once
            model_execution_params = self._prepare_model_execution(
                data_list, model_preference
            )
            model_response = await self.model.acall(**model_execution_params)
            embeddings = self._extract_raw_response(model_response)

            # Ensure list format
            if not isinstance(embeddings, list):
                embeddings = [embeddings]
        else:
            # Non-batch mode: use parallel execution via F.amap_gather
            prepare_execution = partial(
                self._prepare_model_execution, model_preference=model_preference
            )
            distributed_params = list(map(prepare_execution, data_list))
            # amap_gather requires args_list (list of tuples) - use empty tuples since we only have kwargs
            args_list = [()] * len(data_list)
            responses = await F.amap_gather(
                self.model.acall, args_list=args_list, kwargs_list=distributed_params
            )
            embeddings = [
                self._extract_raw_response(response) for response in responses
            ]

        # Return single embedding if input was single
        if is_single:
            # Check if embeddings is already a flat list (single embedding vector)
            # or if it's a list containing a single embedding vector
            if len(embeddings) > 0 and isinstance(embeddings[0], list):
                return embeddings[0]
            return embeddings
        return embeddings

    def _prepare_model_execution(
        self, data: Union[str, List[str]], model_preference: Optional[str] = None
    ) -> Dict[str, Any]:
        """Prepare model execution parameters."""
        model_execution_params = dotdict(self.config) if self.config else dotdict()
        model_execution_params.data = data
        if model_preference:
            model_execution_params.model_preference = model_preference
        return model_execution_params

    def _prepare_task(
        self, message: Union[str, List[str], Message], **kwargs
    ) -> Dict[str, Any]:
        """Prepare task inputs."""
        if isinstance(message, Message):
            data = self._extract_message_values(self.task_inputs, message)
        else:
            data = message

        model_preference = kwargs.pop("model_preference", None)
        if model_preference is None and isinstance(message, Message):
            model_preference = self.get_model_preference_from_message(message)

        return {"data": data, "model_preference": model_preference}

    def inspect_model_execution_params(self, *args, **kwargs) -> Mapping[str, Any]:
        """Debug model input parameters."""
        inputs = self._prepare_task(*args, **kwargs)
        model_execution_params = self._prepare_model_execution(**inputs)
        return model_execution_params

    def _set_model(self, model: EMBEDDER_MODELS):
        """Set and validate embedding model."""
        if "embedder" in model.model_type:
            self.register_buffer("model", model)
        else:
            raise TypeError(
                f"`model` requires be `embedder` model, given `{type(model)}`"
            )

    def _set_config(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            self.register_buffer("config", {})
            return

        if not isinstance(config, dict):
            raise TypeError(f"`config` must be a dict or None, given `{type(config)}`")

        self.register_buffer("config", config.copy())
