import inspect
from os import getenv
from typing import Any, Dict, Optional, Tuple

from msgflux.data.retrievers.retriever import Retriever
from msgflux.models.model import Model


class WebSearch:
    """Search the web through a retriever backend or a web-search model.

    The engine string selects the execution mode:

    - ``retriever/<provider>`` uses ``Retriever.web_search(<provider>)``
    - ``model/<provider>/<model-id>`` uses ``Model.chat_completion(...)``
    - if ``engine`` is omitted, the tool reads ``MSGFLUX_WEB_SEARCH_ENGINE``
      from the environment

    The public tool schema is configured dynamically from the selected engine.
    Retriever-backed tools expose ``query`` only. Model-backed tools expose an
    optional ``prompt`` parameter so the caller can steer the underlying model.

    Returns:
        A dict with:
        - ``data``: the search result payload
        - ``annotations``: citation metadata when the backend provides it
    """

    name = "web_search"
    engine_env_var = "MSGFLUX_WEB_SEARCH_ENGINE"

    def __init__(
        self,
        engine: Optional[str] = None,
        *,
        top_k: int = 5,
        web_search_options: Optional[Dict[str, Any]] = None,
    ):
        self.engine = engine or getenv(self.engine_env_var)
        if not self.engine:
            raise ValueError(
                "Missing web search engine. Pass `engine=` or set "
                f"`{self.engine_env_var}`."
            )
        self.top_k = top_k
        self.web_search_options = web_search_options or {}

        self.engine_kind, self.engine_target = self._parse_engine(self.engine)

        if self.engine_kind == "retriever":
            if self.web_search_options:
                raise ValueError(
                    "`web_search_options` is only supported for model engines."
                )
            self.retriever = Retriever.web_search(self.engine_target)
            self.annotations = {"query": str, "return": str}
        else:
            self.model = Model.chat_completion(
                self.engine_target, web_search_options=self.web_search_options
            )
            self.annotations = {"query": str, "prompt": Optional[str], "return": str}

        self.description = self._build_description()

    @staticmethod
    def _parse_engine(engine: str) -> Tuple[str, str]:
        if not isinstance(engine, str) or not engine.strip():
            raise TypeError("`engine` must be a non-empty string")

        parts = engine.split("/", 2)
        if len(parts) < 2:
            raise ValueError(
                "Invalid engine format. Use `retriever/<provider>` or "
                "`model/<provider>/<model-id>`."
            )

        engine_kind = parts[0].strip()
        if engine_kind == "retriever":
            if len(parts) != 2 or not parts[1].strip():
                raise ValueError(
                    "Retriever engines must use `retriever/<provider>`."
                )
            return engine_kind, parts[1].strip()

        if engine_kind == "model":
            if len(parts) != 3 or not parts[1].strip() or not parts[2].strip():
                raise ValueError(
                    "Model engines must use `model/<provider>/<model-id>`."
                )
            return engine_kind, f"{parts[1].strip()}/{parts[2].strip()}"

        raise ValueError(
            "Invalid engine prefix. Use `retriever/...` or `model/...`."
        )

    def _build_description(self) -> str:
        if self.engine_kind == "retriever":
            return inspect.cleandoc(
                f"""
                Search the web using the `{self.engine_target}` retriever backend.

                This mode is deterministic and delegates retrieval to the selected
                provider implementation.
                """
            )

        return inspect.cleandoc(
            f"""
            Search the web using the `{self.engine_target}` model backend.

            This mode relies on the model's built-in web search capability and
            uses `web_search_options` at model initialization time. The call-time
            `prompt` argument can steer the model before it answers.
            """
        )

    @staticmethod
    def _default_model_prompt() -> str:
        return inspect.cleandoc(
            """
            Use the built-in web search capability to answer the query with
            current, relevant information. Prefer concise, grounded responses.
            """
        )

    @staticmethod
    def _extract_annotations(response: Any) -> list:
        metadata = getattr(response, "metadata", None)
        if metadata is None:
            return []
        if isinstance(metadata, dict):
            return metadata.get("annotations", []) or []
        return getattr(metadata, "annotations", []) or []

    def _run_retriever(self, query: str) -> Dict[str, Any]:
        result = self.retriever(query, top_k=self.top_k)
        return {
            "data": getattr(result, "data", result),
            "annotations": self._extract_annotations(result),
        }

    def _run_model(self, query: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        effective_prompt = (
            prompt if prompt is not None else self._default_model_prompt()
        )
        response = self.model(messages=query, system_prompt=effective_prompt)
        consumed = response.consume() if hasattr(response, "consume") else response
        return {
            "data": consumed,
            "annotations": self._extract_annotations(response),
        }

    def __call__(self, query: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """Search the web with the configured backend."""
        if self.engine_kind == "retriever":
            if prompt is not None:
                raise ValueError(
                    "The `prompt` argument is only supported for model engines."
                )
            return self._run_retriever(query)
        return self._run_model(query, prompt=prompt)

    async def acall(self, query: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """Async version of ``__call__``."""
        if self.engine_kind == "retriever":
            if prompt is not None:
                raise ValueError(
                    "The `prompt` argument is only supported for model engines."
                )
            result = await self.retriever.acall(query, top_k=self.top_k)
            return {
                "data": getattr(result, "data", result),
                "annotations": self._extract_annotations(result),
            }

        effective_prompt = (
            prompt if prompt is not None else self._default_model_prompt()
        )
        response = await self.model.acall(
            messages=query,
            system_prompt=effective_prompt,
        )
        consumed = response.consume() if hasattr(response, "consume") else response
        return {
            "data": consumed,
            "annotations": self._extract_annotations(response),
        }
