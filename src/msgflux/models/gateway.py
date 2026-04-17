from datetime import datetime, time, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from msgflux.exceptions import ModelRouterError
from msgflux.logger import logger
from msgflux.models.base import BaseModel
from msgflux.models.model import Model
from msgflux.models.response import ModelResponse, ModelStreamResponse


class ModelGateway:
    """Routes calls to a list of model deployments, with fallback, retries,
    initial model selection, and timing constraints (configured via HH:MM strings).

    Each deployment is a dict with:
        - ``model_name`` (str): Alias for the model (e.g. "weak", "strong").
        - ``model`` (BaseModel): The model instance.
        - ``time_constraints`` (optional): List of ``(start, end)`` HH:MM tuples.
    """

    msgflux_type = "model_gateway"
    model_types = None

    def __init__(
        self,
        models: List[Dict[str, Any]],
    ):
        """Initialize the ModelGateway.

        Args:
            models:
                A list of model deployment dicts. Each dict must contain:

                - ``model_name`` (str): A unique alias for the model.
                - ``model`` (BaseModel): The model instance.
                - ``time_constraints`` (optional): A list of string tuples
                  ``(start_time, end_time)`` in ``"HH:MM"`` format. The model
                  will NOT be used if the current time falls within any range.

                !!! example

                    [
                        {
                            "model_name": "weak",
                            "model": Model.chat_completion("openai/gpt-4.1-mini"),
                            "time_constraints": [("22:00", "06:00")],
                        },
                        {
                            "model_name": "strong",
                            "model": Model.chat_completion("openai/gpt-4.1"),
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

        now = datetime.now(tz=timezone.utc).time()

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
    ) -> Tuple[str, BaseModel]:
        """Validates a single deployment dict and returns (model_name, model)."""
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

        if not isinstance(model_name, str) or not model_name:
            raise TypeError(
                f"`model_name` at position {index} must be a non-empty string"
            )

        if not isinstance(model, BaseModel):
            raise TypeError(f"Model `{model_name}` does not inherit from `BaseModel`")

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

        return model_name, model

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

        for i, deployment in enumerate(models):
            model_name, model = self._validate_deployment(deployment, i)

            model_types.add(model.model_type)

            if model_name in model_names:
                raise ValueError(
                    f"Duplicate model name found: `{model_name}`. Names must be unique"
                )
            model_names.add(model_name)
            self._model_name_to_index[model_name] = i
            extracted_models.append(model)
            model_name_list.append(model_name)

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
        self._deployments = models
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
        if not self.models:
            raise ModelRouterError([], [], message="No model configured on gateway")

        available = [
            (name, model)
            for name, model in zip(self.model_names, self.models)
            if not self._is_time_restricted(name)
        ]

        if not available:
            raise ModelRouterError(
                [], [], message="No model available due to time constraints"
            )

        if model_preference:
            preferred = next(
                ((n, m) for n, m in available if n == model_preference), None
            )
            if preferred:
                available = [preferred] + [
                    (n, m) for n, m in available if n != model_preference
                ]

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
        if not self.models:
            raise ModelRouterError([], [], message="No model configured on gateway")

        available = [
            (name, model)
            for name, model in zip(self.model_names, self.models)
            if not self._is_time_restricted(name)
        ]

        if not available:
            raise ModelRouterError(
                [], [], message="No model available due to time constraints"
            )

        if model_preference:
            preferred = next(
                ((n, m) for n, m in available if n == model_preference), None
            )
            if preferred:
                available = [preferred] + [
                    (n, m) for n, m in available if n != model_preference
                ]

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
            serialized_deployments.append(entry)

        state = {"models": serialized_deployments}
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
            deployments.append(deployment)

        return cls(models=deployments)
