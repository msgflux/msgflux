from functools import partial
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
from msgflux.nn import functional as F
from msgflux.nn.modules.module import Module


RETRIVERS = Union[WebRetriever, LexicalRetriever, SemanticRetriever, VectorDB]
EMBEDDER_MODELS = Union[
    AudioEmbedderModel, ImageEmbedderModel, TextEmbedderModel, ModelGateway
]


class Retriever(Module):
    """Retriever is a Module type that uses information retrivers."""

    def __init__(
        self,
        name: str,
        retriever: RETRIVERS,
        *,
        model: Optional[EMBEDDER_MODELS] = None,
        task_inputs: Optional[Union[str, Dict[str, str]]] = None,
        response_mode: Optional[str] = "plain_response",
        response_template: Optional[str] = None,
        top_k: Optional[int] = 4,
        threshold: Optional[float] = 0.0,
        return_score: Optional[bool] = False,
        dict_key: Optional[str] = None,
    ):
        """Args:
        name:
            Designer name in snake case format.
        retriever:
            Retriever client.
        model:
            An embedding model.
        task_inputs:
            Fields of the Message object that will be the input to the task.
        response_mode:
            What the response should be.
            * `plain_response` (default): Returns the final agent response directly.
            * other: Write on field in Message object.
        response_template:
            A Jinja template to format response.
        top_k:
            Maximum return of similar points.
        threshold:
            Retriever threshold.
        return_score:
            If True, return similarity score.
        dict_key:
            Help to extract a value from task_inputs if dict.
            e.g.:
                self.dict_key='name'
                [{'name': 'clark', 'age': 27}].
        """
        super().__init__()
        self.set_name(name)
        self._set_retriever(retriever)
        self._set_model(model)
        self._set_task_inputs(task_inputs)
        self._set_response_mode(response_mode)
        self._set_response_template(response_template)
        self._set_top_k(top_k)
        self._set_threshold(threshold)
        self._set_return_score(return_score)
        self._set_dict_key(dict_key)

    def forward(
        self, message: Union[str, List[str], List[Dict[str, Any]], Message], **kwargs
    ) -> Union[str, Dict[str, str], Message]:
        inputs = self._prepare_task(message, **kwargs)
        retriever_response = self._execute_retriever(**inputs)
        response = self._prepare_response(retriever_response, message)
        return response

    def _execute_retriever(
        self, queries: List[str], model_preference: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        queries_embed = None
        if self.model:
            queries_embed = self._execute_model(queries, model_preference)

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
            {
                "queries": queries,
                "top_k": self.top_k,
                "return_score": self.return_score,
            }
        )
        if self.threshold:
            retriever_execution_params.threshold = self.threshold
        return retriever_execution_params

    def _execute_model(
        self, queries: List[str], model_preference: Optional[str] = None
    ) -> List[List[float]]:
        if "bached" in self.model.model_type or len(queries) == 1:
            model_execution_params = self._prepare_model_execution(
                queries, model_preference
            )
            model_response = self.model(**model_execution_params)
            queries_embed = self._extract_raw_response(model_response)
            if not isinstance(queries_embed, list):
                queries_embed = [queries_embed]
        else:
            prepare_execution = partial(
                self._prepare_model_execution, model_preference=model_preference
            )
            distributed_params = list(map(prepare_execution, queries))
            responses = F.map_gather(self.model, kwargs_list=distributed_params)
            raw_resposes = [
                self._extract_raw_response(model_response)
                for model_response in responses
            ]
            return raw_resposes

        return queries_embed

    def _prepare_model_execution(
        self, queries: List[str], model_preference: Optional[str] = None
    ) -> Dict[str, Union[str, List[str]]]:
        if len(queries) == 1:
            queries = queries[0]
        model_execution_params = dotdict({"data": queries})
        if isinstance(self.model, ModelGateway) and model_preference is not None:
            model_execution_params.model_preference = model_preference
        return model_execution_params

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
        if self.dict_key:
            queries_list = [data[self.dict_key] for data in queries]
            return queries_list
        else:
            raise AttributeError(
                "message that contain `List[Dict[str, Any]]` "
                "require a `dict_key` to select the key for retrieval"
            )

    def inspect_model_execution_params(self, *args, **kwargs) -> Mapping[str, Any]:
        """Debug model input parameters."""
        if self.model:
            inputs = self._prepare_task(*args, **kwargs)
            model_execution_params = self._prepare_model_execution(**inputs)
            return model_execution_params
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

    def _set_model(self, model: Optional[EMBEDDER_MODELS] = None):
        if "embedder" in model.model_type or model is None:
            self.register_buffer("model", model)
        else:
            raise TypeError(
                f"`model` requires be `embedder` model, given `{type(model)}`"
            )

    def _set_threshold(self, threshold: Optional[float] = None):
        if isinstance(threshold, float):
            if threshold < 0.0:
                raise ValueError(f"`threshold` requires be >= 0.0 given `{threshold}`")
            self.register_buffer("threshold", threshold)
        elif threshold is None:
            self.register_buffer("threshold", threshold)
        else:
            raise TypeError(
                f"`threshold` requires a float or None given `{type(threshold)}`"
            )

    def _set_return_score(self, return_score: Optional[bool] = False): # noqa: FBT001, FBT002
        if isinstance(return_score, bool):
            self.register_buffer("return_score", return_score)
        else:
            raise TypeError(
                f"`threshold` requires a `bool` or None given `{type(return_score)}`"
            )

    def _set_top_k(self, top_k: Optional[int] = 4):
        if isinstance(top_k, int):
            if top_k <= 0:
                raise ValueError(f"`top_k` requires be >= 1 given `{top_k}`")
            self.register_buffer("top_k", top_k)
        else:
            raise TypeError(f"`top_k` requires a int given `{type(top_k)}`")

    def _set_dict_key(self, dict_key: Optional[str] = None):
        if isinstance(dict_key, str) or dict_key is None:
            self.register_buffer("dict_key", dict_key)
        else:
            raise TypeError(
                f"`dict_key` need be a `str` or None given `{type(dict_key)}`"
            )
