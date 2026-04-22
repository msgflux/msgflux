import inspect
import json
from os import getenv
from typing import Any, Dict, Optional, Tuple

from msgflux.data.retrievers.retriever import Retriever
from msgflux.models.model import Model


class WebSearch:
    """Search the web through a retriever backend or a web-search model.

    The engine string selects the execution mode:

    - ``retriever/<provider>`` uses ``Retriever.web_search(<provider>)``
    - ``model/<provider>/<model-id>`` uses ``Model.chat_completion(...)``
    - if ``engine`` is omitted, the tool reads
      ``MSGFLUX_TOOL_WEB_SEARCH_ENGINE`` from the environment

    ``init_params`` are forwarded when the backend client is initialized.
    ``call_params`` are supported for retriever engines and forwarded on every
    retriever call. When they are omitted, the tool reads
    ``MSGFLUX_TOOL_WEB_SEARCH_INIT_PARAMS`` and
    ``MSGFLUX_TOOL_WEB_SEARCH_CALL_PARAMS`` as JSON objects.

    The public tool schema is configured dynamically from the selected engine.
    Retriever-backed tools expose ``query`` only. Model-backed tools expose an
    optional ``prompt`` parameter so the caller can steer the underlying model.

    Returns:
        A dict with:
        - ``data``: the search result payload
        - ``annotations``: citation metadata when the backend provides it
    """

    name = "web_search"
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
        self.engine = engine or getenv(self.engine_env_var)
        if not self.engine:
            raise ValueError(
                "Missing web search engine. Pass `engine=` or set "
                f"`{self.engine_env_var}`."
            )
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
            self.annotations = {"query": str, "prompt": Optional[str], "return": str}

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
                f"""
                Search the web using the `{self.engine_target}` retriever backend.

                This mode is deterministic and delegates retrieval to the selected
                provider implementation. Backend initialization and execution
                options can be configured through `init_params` and `call_params`.
                """
            )

        return inspect.cleandoc(
            f"""
            Search the web using the `{self.engine_target}` model backend.

            This mode relies on the model's built-in web search capability and
            accepts model initialization options through `init_params`. The
            call-time `prompt` argument can steer the model before it answers.
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

    def _model_prompt(self, prompt: Optional[str]) -> str:
        return prompt if prompt is not None else self._default_model_prompt()

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
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        response = self.model(
            messages=query,
            system_prompt=self._model_prompt(prompt),
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
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        response = await self.model.acall(
            messages=query,
            system_prompt=self._model_prompt(prompt),
        )
        return {
            "data": self._consume_response(response),
            "annotations": self._extract_annotations(response),
        }

    def __call__(
        self,
        query: str,
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search the web with the configured backend."""
        if self.engine_kind == "retriever":
            if prompt is not None:
                raise ValueError(
                    "The `prompt` argument is only supported for model engines."
                )
            return self._run_retriever(query)
        return self._run_model(query, prompt=prompt)

    async def acall(
        self,
        query: str,
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async version of ``__call__``."""
        if self.engine_kind == "retriever":
            if prompt is not None:
                raise ValueError(
                    "The `prompt` argument is only supported for model engines."
                )
            return await self._arun_retriever(query)
        return await self._arun_model(query, prompt=prompt)
