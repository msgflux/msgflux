import inspect
import json
from os import getenv
from typing import Any, Dict, Optional, Tuple

from msgflux.data.retrievers.retriever import Retriever
from msgflux.models.model import Model


class WebSearchTool:
    """Search current web sources with a configured retriever or model."""

    name = "web_search"
    default_usage_guidance = (
        "Use for current or external facts; prefer a concise search query."
    )
    display_name = "Web Search"
    default_engine = "retriever/firecrawl"
    engine_env_var = "MSGFLUX_TOOL_WEB_SEARCH_ENGINE"
    init_params_env_var = "MSGFLUX_TOOL_WEB_SEARCH_INIT_PARAMS"
    call_params_env_var = "MSGFLUX_TOOL_WEB_SEARCH_CALL_PARAMS"

    def __init__(
        self,
        engine: Optional[str] = None,
        *,
        init_params: Optional[Dict[str, Any]] = None,
        call_params: Optional[Dict[str, Any]] = None,
    ):
        self.engine = engine or getenv(self.engine_env_var) or self.default_engine
        self.init_params = self._resolve_params(
            init_params,
            env_var=self.init_params_env_var,
            name="init_params",
        )
        self.call_params = self._resolve_params(
            call_params,
            env_var=self.call_params_env_var,
            name="call_params",
        )

        self.engine_kind, self.engine_target = self._parse_engine(self.engine)

        if self.engine_kind == "retriever":
            self.retriever = Retriever.web_search(
                self.engine_target,
                **self.init_params,
            )
            self.annotations = {"query": str, "return": str}
        else:
            if self.call_params:
                raise ValueError(
                    "`call_params` is only supported for retriever engines."
                )
            self.model = Model.chat_completion(self.engine_target, **self.init_params)
            self.annotations = {"query": str, "goal": Optional[str], "return": str}

        self.description = self._build_description()

    @staticmethod
    def _resolve_params(
        params: Optional[Dict[str, Any]],
        *,
        env_var: str,
        name: str,
    ) -> Dict[str, Any]:
        if params is not None:
            if not isinstance(params, dict):
                raise TypeError(f"`{name}` must be a dict")
            return dict(params)

        raw_params = getenv(env_var)
        if raw_params is None or not raw_params.strip():
            return {}

        try:
            decoded = json.loads(raw_params)
        except json.JSONDecodeError as exc:
            raise ValueError(f"`{env_var}` must be a valid JSON object") from exc

        if not isinstance(decoded, dict):
            raise ValueError(f"`{env_var}` must be a JSON object")

        return decoded

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
                raise ValueError("Retriever engines must use `retriever/<provider>`.")
            return engine_kind, parts[1].strip()

        if engine_kind == "model":
            if len(parts) != 3 or not parts[1].strip() or not parts[2].strip():
                raise ValueError(
                    "Model engines must use `model/<provider>/<model-id>`."
                )
            return engine_kind, f"{parts[1].strip()}/{parts[2].strip()}"

        raise ValueError("Invalid engine prefix. Use `retriever/...` or `model/...`.")

    def _build_description(self) -> str:
        if self.engine_kind == "retriever":
            return inspect.cleandoc(
                """
                Search current web sources and return structured results.
                Use for broad questions, not a known URL.
                """
            )

        return inspect.cleandoc(
            """
            Search current web sources and synthesize an answer. Use `goal` to
            focus the result when needed.
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

    @staticmethod
    def _consume_response(response: Any) -> Any:
        return response.consume() if hasattr(response, "consume") else response

    def _model_prompt(self, goal: Optional[str]) -> str:
        return goal if goal is not None else self._default_model_prompt()

    def _run_retriever(
        self,
        query: str,
    ) -> Dict[str, Any]:
        result = self.retriever(query, **self.call_params)
        return {
            "data": getattr(result, "data", result),
            "annotations": self._extract_annotations(result),
        }

    def _run_model(
        self,
        query: str,
        goal: Optional[str] = None,
    ) -> Dict[str, Any]:
        response = self.model(
            messages=query,
            system_prompt=self._model_prompt(goal),
        )
        return {
            "data": self._consume_response(response),
            "annotations": self._extract_annotations(response),
        }

    async def _arun_retriever(
        self,
        query: str,
    ) -> Dict[str, Any]:
        result = await self.retriever.acall(query, **self.call_params)
        return {
            "data": getattr(result, "data", result),
            "annotations": self._extract_annotations(result),
        }

    async def _arun_model(
        self,
        query: str,
        goal: Optional[str] = None,
    ) -> Dict[str, Any]:
        response = await self.model.acall(
            messages=query,
            system_prompt=self._model_prompt(goal),
        )
        return {
            "data": self._consume_response(response),
            "annotations": self._extract_annotations(response),
        }

    def __call__(
        self,
        query: str,
        goal: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search the web with the configured backend."""
        if self.engine_kind == "retriever":
            if goal is not None:
                raise ValueError(
                    "The `goal` argument is only supported for model engines."
                )
            return self._run_retriever(query)
        return self._run_model(query, goal=goal)

    async def acall(
        self,
        query: str,
        goal: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async version of ``__call__``."""
        if self.engine_kind == "retriever":
            if goal is not None:
                raise ValueError(
                    "The `goal` argument is only supported for model engines."
                )
            return await self._arun_retriever(query)
        return await self._arun_model(query, goal=goal)

    def close(self) -> None:
        """Close the search backend after synchronous use."""
        backend = self.retriever if self.engine_kind == "retriever" else self.model
        close = getattr(backend, "close", None)
        if callable(close):
            close()

    async def aclose(self) -> None:
        """Close the search backend when its async host releases the tool."""
        backend = self.retriever if self.engine_kind == "retriever" else self.model
        aclose = getattr(backend, "aclose", None)
        if callable(aclose):
            result = aclose()
            if inspect.isawaitable(result):
                await result
        else:
            self.close()
