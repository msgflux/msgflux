from typing import Any, Dict, List, Mapping, Optional, Union

from msgflux.data.dbs.types import VectorDB
from msgflux.data.retrievers.types import (
    LexicalRetriever,
    SemanticRetriever,
    WebRetriever,
)
from msgflux.dotdict import dotdict
from msgflux.message import Message
from msgflux.models.gateway import ModelGateway
from msgflux.models.types import (
    AudioEmbedderModel,
    ImageEmbedderModel,
    TextEmbedderModel,
)
from msgflux.nn.modules.embedder import Embedder
from msgflux.nn.modules.module import Module

RETRIVERS = Union[WebRetriever, LexicalRetriever, SemanticRetriever, VectorDB]
EMBEDDER_MODELS = Union[
    AudioEmbedderModel, ImageEmbedderModel, TextEmbedderModel, ModelGateway
]


class Retriever(Module):
    """Retriever is a Module type that uses information retrivers."""

    def __init__(
        self,
        retriever: RETRIVERS,
        *,
        model: Optional[Union[EMBEDDER_MODELS, Embedder]] = None,
        message_fields: Optional[Dict[str, Any]] = None,
        response_mode: Optional[str] = "plain_response",
        templates: Optional[Dict[str, str]] = None,
        config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ):
        """Args:
        retriever:
            Retriever client.
        model:
            Embedding model for converting queries to embeddings. Can be either:
            - Embedder: Custom Embedder instance (for advanced usage with hooks)
            - EmbedderModel/ModelGateway: Will be auto-wrapped in Embedder
            Optional - only needed for semantic retrieval.
        message_fields:
            Dictionary mapping Message field names to their paths in the Message object.
            Valid keys: "task_inputs"
            !!! example
                message_fields={"task_inputs": "query.user"}

            Field description:
            - task_inputs: Field path for query input (str or dict)
        response_mode:
            What the response should be.
            * `plain_response` (default): Returns the final agent response directly.
            * other: Write on field in Message object.
        templates:
            Dictionary mapping template types to Jinja template strings.
            Valid keys: "response"
            !!! example
                templates={"response": "Results: {{ content }}"}
        config:
            Dictionary with configuration options. Accepts any keys without validation.
            Common options: "top_k", "threshold", "return_score", "dict_key"
            !!! example
                config={
                    "top_k": 4,
                    "threshold": 0.0,
                    "return_score": False,
                    "dict_key": "name"
                }

            Configuration options:
            - top_k: Maximum return of similar points (int)
            - threshold: Retriever threshold (float)
            - return_score: If True, return similarity score (bool)
            - dict_key: Help to extract a value from task_inputs if dict (str)
        name:
            Retriever name in snake case format.
        """
        super().__init__()
        self._set_retriever(retriever)
        self._set_model(model)
        self._set_message_fields(message_fields)
        self._set_response_mode(response_mode)
        self._set_templates(templates)
        self._set_config(config)
        if name:
            self.set_name(name)

    def forward(
        self, message: Union[str, List[str], List[Dict[str, Any]], Message], **kwargs
    ) -> Union[str, Dict[str, str], Message]:
        """Execute the retriever with the given message.

        Args:
            message: The input message, which can be:
                - str: Direct query string for retrieval
                - List[str]: List of query strings
                - List[Dict[str, Any]]: List of query dictionaries
                - Message: Message object with fields mapped via message_fields
            **kwargs: Runtime overrides for message_fields. Can include:
                - task_inputs: Override field path or direct value

        Returns:
            Retrieved results (str, dict, or Message depending on response_mode)
        """
        inputs = self._prepare_task(message, **kwargs)
        retriever_response = self._execute_retriever(**inputs)
        response = self._prepare_response(retriever_response, message)
        return response

    async def aforward(
        self, message: Union[str, List[str], List[Dict[str, Any]], Message], **kwargs
    ) -> Union[str, Dict[str, str], Message]:
        """Async version of forward. Execute the retriever asynchronously."""
        inputs = self._prepare_task(message, **kwargs)
        retriever_response = await self._aexecute_retriever(**inputs)
        response = self._prepare_response(retriever_response, message)
        return response

    def _execute_retriever(
        self, queries: List[str], model_preference: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        queries_embed = None
        if self.embedder:
            queries_embed = self.embedder(queries, model_preference=model_preference)
            # Ensure list format
            if not isinstance(queries_embed[0], list):
                queries_embed = [queries_embed]

        retriever_execution_params = self._prepare_retriever_execution(
            queries_embed or queries
        )
        retriever_response = self.retriever(**retriever_execution_params)

        results = []

        for query, query_results in zip(queries, retriever_response):
            formatted_result = {
                "results": [
                    {"data": item.get("data", None), "score": item.get("score", None)}
                    for item in query_results
                ],
            }
            if isinstance(query, str):
                formatted_result["query"] = query
            results.append(formatted_result)

        return results

    async def _aexecute_retriever(
        self, queries: List[str], model_preference: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        queries_embed = None
        if self.embedder:
            queries_embed = await self.embedder.aforward(
                queries, model_preference=model_preference
            )
            # Ensure list format
            if not isinstance(queries_embed[0], list):
                queries_embed = [queries_embed]

        retriever_execution_params = self._prepare_retriever_execution(
            queries_embed or queries
        )
        retriever_response = self.retriever(**retriever_execution_params)

        results = []

        for query, query_results in zip(queries, retriever_response):
            formatted_result = {
                "results": [
                    {"data": item.get("data", None), "score": item.get("score", None)}
                    for item in query_results
                ],
            }
            if isinstance(query, str):
                formatted_result["query"] = query
            results.append(formatted_result)

        return results

    def _prepare_retriever_execution(
        self, queries: List[Union[str, List[float]]]
    ) -> Dict[str, Any]:
        retriever_execution_params = dotdict(
            queries=queries,
            top_k=self.config.get("top_k", 4),
            return_score=self.config.get("return_score", False),
        )
        threshold = self.config.get("threshold")
        if threshold:
            retriever_execution_params.threshold = threshold
        return retriever_execution_params

    def _prepare_task(
        self, message: Union[str, List[str], List[Dict[str, Any]], Message], **kwargs
    ) -> List[str]:
        if isinstance(message, Message):
            queries = self._extract_message_values(self.task_inputs, message)
        else:
            queries = message

        if isinstance(queries, str):
            queries = [queries]
        elif isinstance(queries, list):
            if isinstance(queries[0], dict):
                queries = self._process_list_of_dict_inputs(queries)

        model_preference = kwargs.pop("model_preference", None)
        if model_preference is None and isinstance(message, Message):
            model_preference = self.get_model_preference_from_message(message)

        return {"queries": queries, "model_preference": model_preference}

    def _process_list_of_dict_inputs(self, queries: List[Dict[str, Any]]) -> List[str]:
        """Extract the query value from a dict."""
        dict_key = self.config.get("dict_key")
        if dict_key:
            queries_list = [data[dict_key] for data in queries]
            return queries_list
        else:
            raise AttributeError(
                "message that contain `List[Dict[str, Any]]` "
                "require a `dict_key` to select the key for retrieval"
            )

    def inspect_embedder_params(self, *args, **kwargs) -> Mapping[str, Any]:
        """Debug embedder input parameters.

        Returns the parameters that would be passed to the embedder module.
        """
        if self.embedder:
            inputs = self._prepare_task(*args, **kwargs)
            return {
                "queries": inputs["queries"],
                "model_preference": inputs.get("model_preference"),
            }
        return {}

    def _set_retriever(self, retriever: RETRIVERS):
        if isinstance(
            retriever, (WebRetriever, LexicalRetriever, SemanticRetriever, VectorDB)
        ):
            self.register_buffer("retriever", retriever)
        else:
            raise TypeError(
                "`retriever` requires `HybridRetriever`, `LexicalRetriever`, "
                f"`SemanticRetriever` or `VectorDB` instance given `{type(retriever)}`"
            )

    def _set_model(self, model: Optional[Union[EMBEDDER_MODELS, Embedder]] = None):
        if model is None:
            self.embedder = None
            return

        if isinstance(model, Embedder):  # If already Embedder, use directly
            self.embedder = model
        else:  # Auto-wrap in Embedder
            self.embedder = Embedder(model=model)

    @property
    def model(self):
        """Access underlying model for convenience.

        Returns:
            The wrapped model instance, or None if no embedder
        """
        if self.embedder is None:
            return None
        return self.embedder.model

    @model.setter
    def model(self, value: Optional[Union[EMBEDDER_MODELS, Embedder]]):
        """Update the retriever's model."""
        self._set_model(value)

    def _set_config(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            self.register_buffer("config", {})
            return

        if not isinstance(config, dict):
            raise TypeError(f"`config` must be a dict or None, given `{type(config)}`")

        self.register_buffer("config", config.copy())
