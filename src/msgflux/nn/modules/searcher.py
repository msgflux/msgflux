from typing import Any, Dict, List, Mapping, Optional, Union

from msgflux.auto import AutoParams
from msgflux.core.dotdict import dotdict
from msgflux.core.message import Message
from msgflux.data.dbs.types import VectorDB
from msgflux.data.retrievers.types import (
    FuzzyRetriever,
    LexicalRetriever,
    SemanticRetriever,
    WebRetriever,
)
from msgflux.models.gateway import ModelGateway
from msgflux.models.types import (
    AudioEmbedderModel,
    ImageEmbedderModel,
    TextEmbedderModel,
)
from msgflux.nn.modules.embedder import Embedder
from msgflux.nn.modules.module import Module

RETRIVERS = Union[
    WebRetriever, LexicalRetriever, FuzzyRetriever, SemanticRetriever, VectorDB
]
EMBEDDER_MODELS = Union[
    AudioEmbedderModel, ImageEmbedderModel, TextEmbedderModel, ModelGateway
]


class Searcher(Module, metaclass=AutoParams):
    """Searcher is a Module type that uses information retrivers."""

    _autoparams_use_docstring_for = "description"
    _autoparams_use_classname_for = "name"

    def __init__(
        self,
        retriever: RETRIVERS,
        *,
        model: Optional[Union[EMBEDDER_MODELS, Embedder]] = None,
        message_fields: Optional[Dict[str, Any]] = None,
        response_mode: Optional[str] = None,
        templates: Optional[Dict[str, str]] = None,
        config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        annotations: Optional[Dict[str, type]] = None,
    ):
        """Initialize the Searcher module.

        Args:
        retriever:
            Retriever backend client.
        model:
            Embedding model for converting queries to embeddings. Can be either:
            - Embedder: Custom Embedder instance (for advanced usage with hooks)
            - EmbedderModel/ModelGateway: Will be auto-wrapped in Embedder
            Optional - only needed for semantic retrieval.
        message_fields:
            Dictionary mapping Message field names to their paths in the Message object.
            Valid keys: "query", "model_preference"
            !!! example
                message_fields={"query": "query.user"}

            Field description:
            - query: Field path for search input (str or dict)
        response_mode:
            Controls how the response is returned.
            * ``None`` (default): Returns the response directly.
            * ``"<path>"``: Writes to ``obj.<path>`` and returns ``None``
              (``dotdict`` or ``Message`` is mutated in place).
            * ``"<path>:"``: Returns a new ``dotdict`` with the response stored
              under ``<path>`` (no ``Message`` required).
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
            - threshold: Search threshold (float)
            - return_score: If True, return similarity score (bool)
            - dict_key: Help to extract a value from query if dict (str)
        name:
            Searcher name in snake case format.
        description:
            Searcher description. Used as tool description when plugging
            into an Agent. With AutoParams, the class docstring is used
            automatically.
        annotations:
            Type annotations for tool parameters. Defaults to
            ``{"query": str, "return": str}`` so the Searcher can be
            used directly as an Agent tool.
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
        self.set_description(description)
        self.set_annotations(annotations or {"query": str, "return": str})

    def _set_message_fields(self, message_fields: Optional[Dict[str, Any]] = None):
        """Searcher-specific message field mappings.

        Searcher uses `query` as its canonical input name and maps it to the
        base module task field internally.
        """
        valid_keys = {"query", "model_preference"}

        if message_fields is None:
            self._set_task(None)
            self._set_model_preference(None)
            return

        if not isinstance(message_fields, dict):
            raise TypeError(
                f"`message_fields` must be a dict or None, given "
                f"`{type(message_fields)}`"
            )

        invalid_keys = set(message_fields.keys()) - valid_keys
        if invalid_keys:
            raise ValueError(
                f"Invalid message_fields keys: {invalid_keys}. "
                f"Valid keys are: {valid_keys}"
            )

        self._set_task(message_fields.get("query"))
        self._set_model_preference(message_fields.get("model_preference"))

    def forward(
        self,
        message: Optional[Union[str, List[str], List[Dict[str, Any]], Message]] = None,
        **kwargs: Any,
    ) -> Union[str, Dict[str, str], Message]:
        """Execute the retriever with the given message.

        Args:
            message: The input message, which can be:
                - str: Direct query string for retrieval
                - List[str]: List of query strings
                - List[Dict[str, Any]]: List of query dictionaries
                - Message: Message object with fields mapped via message_fields
            **kwargs: Runtime overrides for message_fields. Can include:
                - task: Override field path or direct value

        Returns:
            Retrieved results (str, dict, or Message depending on response_mode)
        """
        if message is None and "query" in kwargs:
            message = kwargs.pop("query")
        if message is None:
            raise TypeError("Searcher.forward() missing required argument: 'message'")
        inputs = self._prepare_inputs(message, **kwargs)
        retriever_response = self._execute_retriever(**inputs)
        response = self._prepare_response(retriever_response, message)
        return response

    async def aforward(
        self,
        message: Optional[Union[str, List[str], List[Dict[str, Any]], Message]] = None,
        **kwargs: Any,
    ) -> Union[str, Dict[str, str], Message]:
        """Async version of forward. Execute the retriever asynchronously."""
        if message is None and "query" in kwargs:
            message = kwargs.pop("query")
        if message is None:
            raise TypeError("Searcher.aforward() missing required argument: 'message'")
        inputs = self._prepare_inputs(message, **kwargs)
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
        query_results_list = self._normalize_retriever_response(retriever_response)
        return self._format_results(queries, query_results_list)

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
        query_results_list = self._normalize_retriever_response(retriever_response)
        return self._format_results(queries, query_results_list)

    def _normalize_retriever_response(
        self, response: Any
    ) -> List[List[Dict[str, Any]]]:
        """Normalize retriever response to a flat list of result lists.

        Retrievers return dotdict(response_type=..., data=[dotdict(results=[...])]).
        This method extracts the inner results into a simple list of lists.
        """
        if isinstance(response, dotdict) and "data" in response:
            return [
                item.results
                if isinstance(item, dotdict) and "results" in item
                else item
                for item in response.data
            ]
        return response

    def _format_results(
        self, queries: List[str], query_results_list: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Format raw retriever results into a consistent output structure."""
        return_score = self.config.get("return_score", False)
        results = []
        for query, query_results in zip(queries, query_results_list):
            formatted_results = []
            for item in query_results:
                entry = {"data": item.get("data", None)}
                if return_score:
                    entry["score"] = item.get("score", None)
                formatted_results.append(entry)
            formatted_result = {"results": formatted_results}
            if isinstance(query, str):
                formatted_result["query"] = query
            results.append(formatted_result)
        return results

    def _format_response_template(self, content: Any) -> str:
        """Apply response template to retriever results.

        Handles list results by applying the template to each result dict
        individually and joining them.
        """
        template = self.templates.get("response")
        if isinstance(content, list):
            formatted = [self._format_template(item, template) for item in content]
            return "\n\n".join(formatted)
        return self._format_template(content, template)

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

    def _prepare_inputs(
        self, message: Union[str, List[str], List[Dict[str, Any]], Message], **kwargs
    ) -> List[str]:
        if isinstance(message, dotdict):
            queries = self._extract_message_values(self.task, message)
        else:
            queries = message

        if isinstance(queries, str):
            queries = [queries]
        elif isinstance(queries, list):
            if isinstance(queries[0], dict):
                queries = self._process_list_of_dict_inputs(queries)

        model_preference = kwargs.pop("model_preference", None)
        if model_preference is None and isinstance(message, dotdict):
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
            inputs = self._prepare_inputs(*args, **kwargs)
            return {
                "queries": inputs["queries"],
                "model_preference": inputs.get("model_preference"),
            }
        return {}

    def _set_retriever(self, retriever: RETRIVERS):
        if isinstance(
            retriever,
            (
                WebRetriever,
                LexicalRetriever,
                FuzzyRetriever,
                SemanticRetriever,
                VectorDB,
            ),
        ):
            self.register_buffer("retriever", retriever)
        else:
            raise TypeError(
                "`retriever` requires `WebRetriever`, `LexicalRetriever`, "
                "`FuzzyRetriever`, `SemanticRetriever` or `VectorDB` instance "
                f"given `{type(retriever)}`"
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
    def model(self) -> Any:
        """Access underlying model for convenience.

        Returns:
            The wrapped model instance, or None if no embedder.
        """
        if self.embedder is None:
            return None
        return self.embedder.model

    @model.setter
    def model(self, value: Optional[Union[EMBEDDER_MODELS, Embedder]]):
        """Update the searcher's model."""
        self._set_model(value)

    def _set_config(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            self.register_buffer("config", {})
            return

        if not isinstance(config, dict):
            raise TypeError(f"`config` must be a dict or None, given `{type(config)}`")

        self.register_buffer("config", config.copy())
