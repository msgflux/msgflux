import warnings
from datetime import datetime, time, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from msgflux.exceptions import ModelRouterError
from msgflux.logger import logger
from msgflux.models.base import BaseModel
from msgflux.models.chat_extensions import validate_chat_speed
from msgflux.models.model import Model
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.models.types import validate_reasoning_effort
from msgflux.utils.time import utc_now

if TYPE_CHECKING:
    from msgflux.tools.catalog import ToolCatalogView


class ModelGateway:
    """Routes calls to a list of model deployments, with fallback, retries,
    initial model selection, and timing constraints (configured via HH:MM strings).

    Each deployment is a dict with:
        - ``model_name`` (str): Alias for the model (e.g. "weak", "strong").
        - ``model`` (BaseModel | str): The model instance or a chat-completion
          shorthand in ``"provider/model-id"`` form.
        - ``description`` (optional str): A concise description of the model's
          capabilities for model-facing selection interfaces.
        - ``time_constraints`` (optional): List of ``(start, end)`` HH:MM tuples.
    """

    msgflux_type = "model_gateway"
    model_types = None

    def __init__(
        self,
        models: List[Dict[str, Any]],
        *,
        fallback: bool = True,
    ):
        """Initialize the ModelGateway.

        Args:
            models:
                A list of model deployment dicts. Each dict must contain:

                - ``model_name`` (str): A unique alias for the model.
                - ``model`` (BaseModel | str): The model instance, or a
                  ``"provider/model-id"`` shorthand initialized through
                  ``Model.chat_completion``.
                - ``description`` (optional str): A concise model capability
                  description. Interfaces that expose model selection may
                  require it.
                - ``time_constraints`` (optional): A list of string tuples
                  ``(start_time, end_time)`` in ``"HH:MM"`` format. The model
                  will NOT be used if the current time falls within any range.
            fallback:
                Whether a failed or unavailable selected model may transition
                to another deployment. Defaults to True.

                !!! example

                    [
                        {
                            "model_name": "weak",
                            "model": "openai/gpt-5.6-luna",
                            "description": "Fast for clear, repeatable tasks.",
                            "time_constraints": [("22:00", "06:00")],
                        },
                        {
                            "model_name": "strong",
                            "model": "openai/gpt-5.6-sol",
                            "description": "Deep reasoning for complex tasks.",
                        },
                    ]

        Raises:
            ModelRouterError:
                Raised when all models fail or are restricted.
            ValueError:
                Raised for misconfiguration in time formats or duplicate model names.
            TypeError:
                Raised for invalid argument types.
        """
        if not isinstance(fallback, bool):
            raise TypeError("`fallback` must be a bool")
        self.fallback = fallback
        self._model_name_to_index: Dict[str, int] = {}
        self._set_models(models)

        # Build time_constraints dict from deployments
        raw_constraints: Dict[str, List[Tuple[str, str]]] = {}
        for deployment in self._deployments:
            tc = deployment.get("time_constraints")
            if tc is not None:
                raw_constraints[deployment["model_name"]] = tc

        self.raw_time_constraints = raw_constraints if raw_constraints else None

        try:
            self.parsed_time_constraints = (
                self._parse_time_constraints(raw_constraints) if raw_constraints else {}
            )
        except ValueError as e:
            logger.error(f"Error to parse time_constraints: {e}")
            raise ValueError(f"Invalid format in time_constraints: {e}") from e

        self.current_model_index = 0
        logger.debug(
            f"ModelGateway initialized with {len(self.models)} models. Type: "
            f"`{self.model_type}`"
        )
        if self.parsed_time_constraints:
            logger.debug(
                "Time constraints applied to models: "
                f"{list(self.parsed_time_constraints.keys())}"
            )

    def _parse_time_constraints(
        self, constraints: Optional[Dict[str, List[Tuple[str, str]]]] = None
    ) -> Dict[str, List[Tuple[time, time]]]:
        """Validates and converts "HH:MM" time strings into datetime.time objects.

        Raises:
            ValueError: If a time string is in an invalid format.
            TypeError: If the constraint data structure is incorrect.
        """
        if constraints is None:
            return {}

        parsed_constraints: Dict[str, List[Tuple[time, time]]] = {}
        time_format = "%H:%M"

        for model_name, intervals in constraints.items():
            if not isinstance(intervals, list):
                raise TypeError(
                    f"Constraints for `{model_name}` must be a list of tuples "
                    f"(start, end). Given: `{type(intervals)}`"
                )
            parsed_intervals = []
            for i, interval in enumerate(intervals):
                if (
                    not isinstance(interval, (tuple, list)) or len(interval) != 2
                ):  # Tuples or lists
                    raise TypeError(
                        f"Interval #{i + 1} for `{model_name}` must be a "
                        "tuple/list of two strings (start_time_str, end_time_str)"
                        f". Given: `{interval}`"
                    )

                start, end = interval
                if not isinstance(start, str) or not isinstance(end, str):
                    raise TypeError(
                        f"Start and end times in range #{i + 1} for "
                        f"`{model_name}` must be strings. Given: "
                        f"`({type(start)}, {type(end)})`"
                    )

                try:
                    start_dt = datetime.strptime(start, time_format).replace(
                        tzinfo=timezone.utc
                    )
                    end_dt = datetime.strptime(end, time_format).replace(
                        tzinfo=timezone.utc
                    )
                    start_t = start_dt.time()
                    end_t = end_dt.time()
                    parsed_intervals.append((start_t, end_t))
                except ValueError as e:
                    raise ValueError(
                        f"Invalid time format in range #{i + 1} for "
                        f"`{model_name}`. Use 'HH:MM'. Error parsing "
                        f"`{start}` or `{end}`: {e}"
                    ) from e

            parsed_constraints[model_name] = parsed_intervals
        return parsed_constraints

    def _is_time_restricted(self, model_name: str) -> bool:
        """Checks if the model is constrained at the current time
        using the parsed constraints.
        """
        if model_name not in self.parsed_time_constraints:
            return False

        now = utc_now().time()

        for start_time, end_time in self.parsed_time_constraints[model_name]:
            if start_time <= end_time:
                if start_time <= now <= end_time:
                    logger.debug(
                        f"Model `{model_name}` restricted. Current time `{now}` "
                        f"is between `{start_time}` and `{end_time}`"
                    )
                    return True
            elif now >= start_time or now <= end_time:
                logger.debug(
                    f"Restricted model `{model_name}`. Current time `{now}` is "
                    f"in the range crosses midnight: `{start_time} - {end_time}`"
                )
                return True
        return False

    def _validate_deployment(
        self, deployment: Dict[str, Any], index: int
    ) -> Tuple[str, BaseModel, Optional[str]]:
        """Validate and normalize one deployment."""
        if "model_name" not in deployment:
            raise ValueError(
                f"Deployment at position {index} is missing required key `model_name`"
            )
        if "model" not in deployment:
            raise ValueError(
                f"Deployment at position {index} is missing required key `model`"
            )

        model_name = deployment["model_name"]
        model = deployment["model"]
        description = deployment.get("description")

        if not isinstance(model_name, str) or not model_name:
            raise TypeError(
                f"`model_name` at position {index} must be a non-empty string"
            )

        model = self._normalize_deployment_model(model_name, model)
        description = self._normalize_description(model_name, description)

        if not hasattr(model, "model_type") or not model.model_type:
            raise AttributeError(
                f"Model `{model_name}` does not have a valid `model_type` attribute"
            )
        if not hasattr(model, "model_id") or not model.model_id:
            raise AttributeError(
                f"Model `{model_name}` does not have a valid `model_id` attribute"
            )
        if not hasattr(model, "provider"):
            raise AttributeError(
                f"Model `{model_name}` does not have a valid `provider` attribute"
            )

        return model_name, model, description

    @staticmethod
    def _normalize_deployment_model(model_name: str, model: Any) -> BaseModel:
        if isinstance(model, str) and (
            "/" not in model or not all(model.split("/", 1))
        ):
            raise TypeError(
                f"Model `{model_name}` does not inherit from `BaseModel` or use "
                "a valid `provider/model-id` string"
            )
        if isinstance(model, str):
            model = Model.chat_completion(model)
        if not isinstance(model, BaseModel):
            raise TypeError(
                f"Model `{model_name}` must inherit from `BaseModel` or be a "
                "`provider/model-id` string"
            )
        return model

    @staticmethod
    def _normalize_description(
        model_name: str,
        description: Any,
    ) -> Optional[str]:
        if description is None:
            return None
        if not isinstance(description, str) or not description.strip():
            raise TypeError(
                f"`description` for model `{model_name}` must be a non-empty "
                "string or None"
            )
        return description.strip()

    def _set_models(self, models: List[Dict[str, Any]]):
        if not models or not isinstance(models, list):
            raise TypeError(
                "`models` must be a non-empty list of model deployment dicts"
            )

        if not all(isinstance(d, dict) for d in models):
            raise TypeError(
                "`models` requires a list of dicts with `model_name` and `model` keys"
            )

        model_types = set()
        model_names = set()
        extracted_models = []
        model_name_list = []
        model_descriptions: Dict[str, Optional[str]] = {}
        normalized_deployments = []

        for i, deployment in enumerate(models):
            model_name, model, description = self._validate_deployment(deployment, i)

            model_types.add(model.model_type)

            if model_name in model_names:
                raise ValueError(
                    f"Duplicate model name found: `{model_name}`. Names must be unique"
                )
            model_names.add(model_name)
            self._model_name_to_index[model_name] = i
            extracted_models.append(model)
            model_name_list.append(model_name)
            model_descriptions[model_name] = description
            normalized_deployment = dict(deployment)
            normalized_deployment["model"] = model
            if description is not None:
                normalized_deployment["description"] = description
            normalized_deployments.append(normalized_deployment)

        if len(models) < 2:
            logger.warning(
                f"`models` has only {len(models)} deployments. "
                "Fallback will not be effective"
            )

        if len(model_types) > 1:
            raise TypeError(
                "All models in `models` must be of the same `model_type`. "
                f"Given: `{model_types}`"
            )

        self.models = extracted_models
        self.model_names = model_name_list
        self._model_descriptions = model_descriptions
        self._deployments = normalized_deployments
        self.model_type = next(iter(model_types))

        # Determine if gateway supports batch processing
        # Only True if ALL models support batch
        self.batch_support = (
            all(getattr(model, "batch_support", False) for model in extracted_models)
            if extracted_models
            else False
        )

    def _execute_model(
        self, model_preference: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Attempts to execute the call on the configured models, respecting
        time constraints and failure limits.
        """
        available = self._available_models(model_preference)

        if not available:
            raise ModelRouterError(
                [], [], message="No model available due to time constraints"
            )

        failures = []

        for name, model in available:
            try:
                response = model(**kwargs)
                return response
            except Exception as e:
                logger.debug(
                    f"""Model `{name}` ({model.provider})
                    failed to execute: {e}""",
                    exc_info=False,
                )
                failures.append((name, model.provider, e))

        error_message = f"All {len(available)} available models failed"
        logger.error(error_message)
        raise ModelRouterError(
            [failure[2] for failure in failures],
            failures,
            message=error_message,
        )

    async def _aexecute_model(
        self, model_preference: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Async version of _execute_model. Attempts to execute the call on the
        configured models, respecting time constraints and failure limits.
        """
        available = self._available_models(model_preference)

        if not available:
            raise ModelRouterError(
                [], [], message="No model available due to time constraints"
            )

        failures = []

        for name, model in available:
            try:
                response = await model.acall(**kwargs)
                return response
            except Exception as e:
                logger.debug(
                    f"""Model `{name}` ({model.provider})
                    failed to execute: {e}""",
                    exc_info=False,
                )
                failures.append((name, model.provider, e))

        error_message = f"All {len(available)} available models failed"
        logger.error(error_message)
        raise ModelRouterError(
            [failure[2] for failure in failures],
            failures,
            message=error_message,
        )

    def _available_models(
        self,
        model_preference: Optional[str] = None,
    ) -> List[Tuple[str, BaseModel]]:
        if model_preference is not None:
            self.validate_model_name(model_preference)

        if not self.fallback:
            selected_name = model_preference or self.model_names[0]
            if self._is_time_restricted(selected_name):
                raise ModelRouterError(
                    [],
                    [],
                    message=(
                        f"Selected model `{selected_name}` is unavailable due to "
                        "time constraints"
                    ),
                )
            selected_index = self._model_name_to_index[selected_name]
            return [(selected_name, self.models[selected_index])]

        available = [
            (name, model)
            for name, model in zip(self.model_names, self.models)
            if not self._is_time_restricted(name)
        ]
        if model_preference:
            preferred = next(
                ((n, m) for n, m in available if n == model_preference), None
            )
            if preferred:
                available = [preferred] + [
                    (n, m) for n, m in available if n != model_preference
                ]
        return available

    def validate_model_name(self, model_name: str) -> None:
        """Validate a public deployment alias."""
        if not isinstance(model_name, str) or not model_name:
            raise TypeError("`model_name` must be a non-empty string")
        if model_name not in self._model_name_to_index:
            available = ", ".join(self.model_names)
            raise ValueError(
                f"Unknown model `{model_name}`. Available models: {available}."
            )

    def get_model_description(self, model_name: str) -> Optional[str]:
        """Return the configured capability description for one deployment."""
        self.validate_model_name(model_name)
        return self._model_descriptions[model_name]

    def set_reasoning_effort(
        self,
        reasoning_effort: str | None,
        *,
        model_name: str | None = None,
    ) -> "ModelGateway":
        """Update request-level reasoning effort on one or every deployment.

        Args:
            reasoning_effort:
                Provider-specific effort level, or None to remove the setting.
            model_name:
                Optional deployment alias. When omitted, updates every model so
                fallback preserves the same requested effort.
        """
        validate_reasoning_effort(reasoning_effort)
        if model_name is None:
            targets = self.models
        else:
            self.validate_model_name(model_name)
            targets = [self.models[self._model_name_to_index[model_name]]]

        for model in targets:
            setter = getattr(model, "set_reasoning_effort", None)
            if callable(setter):
                setter(reasoning_effort)
                continue
            if reasoning_effort is not None:
                warnings.warn(
                    f"Model `{model.model_id}` from provider `{model.provider}` "
                    "does not support request-level `reasoning_effort`; the "
                    "setting was ignored.",
                    UserWarning,
                    stacklevel=2,
                )
        return self

    def set_speed(
        self,
        speed: str | None,
        *,
        model_name: str | None = None,
    ) -> "ModelGateway":
        """Update provider-neutral request speed on one or every deployment."""
        validate_chat_speed(speed)
        if model_name is None:
            targets = self.models
        else:
            self.validate_model_name(model_name)
            targets = [self.models[self._model_name_to_index[model_name]]]

        for model in targets:
            setter = getattr(model, "set_speed", None)
            if callable(setter):
                setter(speed)
                continue
            if speed is not None:
                warnings.warn(
                    f"Model `{model.model_id}` from provider `{model.provider}` "
                    f"does not support `speed={speed!r}`; the setting was ignored.",
                    UserWarning,
                    stacklevel=2,
                )
        return self

    @property
    def model_descriptions(self) -> Dict[str, Optional[str]]:
        """Return capability descriptions keyed by deployment alias."""
        return dict(self._model_descriptions)

    @property
    def context_capacity(self) -> int | None:
        """Return the smallest known context window across all deployments."""
        capacities = [getattr(model, "context_capacity", None) for model in self.models]
        if not capacities or any(capacity is None for capacity in capacities):
            return None
        return min(capacities)

    def count_context_tokens(
        self,
        messages,
        *,
        system_prompt: str | None = None,
        tool_catalog: "ToolCatalogView | None" = None,
        model_preference: str | None = None,
    ):
        return self._execute_model_method(
            "count_context_tokens",
            model_preference=model_preference,
            messages=messages,
            system_prompt=system_prompt,
            tool_catalog=tool_catalog,
        )

    async def acount_context_tokens(
        self,
        messages,
        *,
        system_prompt: str | None = None,
        tool_catalog: "ToolCatalogView | None" = None,
        model_preference: str | None = None,
    ):
        return await self._aexecute_model_method(
            "acount_context_tokens",
            model_preference=model_preference,
            messages=messages,
            system_prompt=system_prompt,
            tool_catalog=tool_catalog,
        )

    def compact_context(
        self,
        messages,
        *,
        system_prompt: str | None = None,
        model_preference: str | None = None,
        native: bool = True,
    ):
        # A gateway can fall back across providers after compaction. Force a
        # portable view so opaque provider state never pins a future request.
        _ = native
        return self._execute_model_method(
            "compact_context",
            model_preference=model_preference,
            messages=messages,
            system_prompt=system_prompt,
            native=False,
        )

    async def acompact_context(
        self,
        messages,
        *,
        system_prompt: str | None = None,
        model_preference: str | None = None,
        native: bool = True,
    ):
        _ = native
        return await self._aexecute_model_method(
            "acompact_context",
            model_preference=model_preference,
            messages=messages,
            system_prompt=system_prompt,
            native=False,
        )

    def _execute_model_method(
        self,
        method_name: str,
        *,
        model_preference: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        available = self._available_models(model_preference)
        if not available:
            raise ModelRouterError(
                [], [], message="No model available due to time constraints"
            )

        failures = []
        for name, model in available:
            try:
                return getattr(model, method_name)(**kwargs)
            except Exception as e:
                logger.debug(
                    f"""Model `{name}` ({model.provider})
                    failed to execute `{method_name}`: {e}""",
                    exc_info=False,
                )
                failures.append((name, model.provider, e))

        error_message = f"All {len(available)} available models failed"
        logger.error(error_message)
        raise ModelRouterError(
            [failure[2] for failure in failures],
            failures,
            message=error_message,
        )

    async def _aexecute_model_method(
        self,
        method_name: str,
        *,
        model_preference: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        available = self._available_models(model_preference)
        if not available:
            raise ModelRouterError(
                [], [], message="No model available due to time constraints"
            )

        failures = []
        for name, model in available:
            try:
                return await getattr(model, method_name)(**kwargs)
            except Exception as e:
                logger.debug(
                    f"""Model `{name}` ({model.provider})
                    failed to execute `{method_name}`: {e}""",
                    exc_info=False,
                )
                failures.append((name, model.provider, e))

        error_message = f"All {len(available)} available models failed"
        logger.error(error_message)
        raise ModelRouterError(
            [failure[2] for failure in failures],
            failures,
            message=error_message,
        )

    def __call__(
        self, *, model_preference: Optional[str] = None, **kwargs: Any
    ) -> Union[ModelResponse, ModelStreamResponse]:
        """Executes the call on the gateway.

        Args:
            model_preference:
                The ``model_name`` of the deployment that should be tried first.
                If None, starts from the first model in the list.
            kwargs:
                Arguments to pass to the __call__ method of the selected model.

        Returns:
            The response of the first model that executes successfully.

        Raises:
            ModelRouterError:
                If all models fail consecutively, or if no models are
                available/functional.
        """
        return self._execute_model(model_preference=model_preference, **kwargs)

    async def acall(
        self, *, model_preference: Optional[str] = None, **kwargs: Any
    ) -> Union[ModelResponse, ModelStreamResponse]:
        """Async version of __call__. Executes the call on the gateway.

        Args:
            model_preference:
                The ``model_name`` of the deployment that should be tried first.
                If None, starts from the first model in the list.
            kwargs:
                Arguments to pass to the acall method of the selected model.

        Returns:
            The response of the first model that executes successfully.

        Raises:
            ModelRouterError:
                If all models fail consecutively, or if no models are
                available/functional.
        """
        return await self._aexecute_model(model_preference=model_preference, **kwargs)

    def warmup_system_prompt(
        self,
        *,
        system_prompt: Optional[str],
        tool_catalog: Optional["ToolCatalogView"] = None,
        model_preference: Optional[str] = None,
    ) -> Any:
        """Warm the first available deployment that supports prompt warmup."""
        return self._execute_model_method(
            "warmup_system_prompt",
            model_preference=model_preference,
            system_prompt=system_prompt,
            tool_catalog=tool_catalog,
        )

    async def awarmup_system_prompt(
        self,
        *,
        system_prompt: Optional[str],
        tool_catalog: Optional["ToolCatalogView"] = None,
        model_preference: Optional[str] = None,
    ) -> Any:
        """Async prompt warmup for the first available deployment."""
        return await self._aexecute_model_method(
            "awarmup_system_prompt",
            model_preference=model_preference,
            system_prompt=system_prompt,
            tool_catalog=tool_catalog,
        )

    def serialize(self) -> Dict[str, Any]:
        """Serializes the gateway state including deployments."""
        serialized_deployments = []
        for name, model, deployment in zip(
            self.model_names, self.models, self._deployments
        ):
            entry: Dict[str, Any] = {
                "model_name": name,
                "model": model.serialize(),
            }
            tc = deployment.get("time_constraints")
            if tc is not None:
                entry["time_constraints"] = tc
            description = deployment.get("description")
            if description is not None:
                entry["description"] = description
            serialized_deployments.append(entry)

        state = {
            "models": serialized_deployments,
            "fallback": self.fallback,
        }
        data = {"msgflux_type": self.msgflux_type, "state": state}
        return data

    @classmethod
    def from_serialized(cls, data: Dict[str, Any]) -> "ModelGateway":
        """Creates a ModelGateway instance from serialized data.

        Args:
            data: The dictionary of serialized gateway data.
        """
        if data.get("msgflux_type") != cls.msgflux_type:
            raise ValueError(
                f"Incorrect msgflux type. Expected `{cls.msgflux_type}`, "
                f"given `{data.get('msgflux_type')}`"
            )

        state = data.get("state", {})
        serialized_deployments = state.get("models", [])
        if not serialized_deployments:
            raise ValueError("Serialized data does not contain models")

        deployments = []
        for entry in serialized_deployments:
            model = Model.from_serialized(**entry["model"])
            deployment: Dict[str, Any] = {
                "model_name": entry["model_name"],
                "model": model,
            }
            if "time_constraints" in entry:
                deployment["time_constraints"] = entry["time_constraints"]
            if "description" in entry:
                deployment["description"] = entry["description"]
            deployments.append(deployment)

        return cls(
            models=deployments,
            fallback=state.get("fallback", True),
        )
