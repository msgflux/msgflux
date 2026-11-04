import copy
from abc import abstractmethod
from typing import Mapping

from msgflux._private.client import BaseClient


class BaseModel(BaseClient):
    msgflux_type = "model"
    to_ignore = [
        "_api_key",
        "model",
        "processor",
        "client",
        "aclient",
        "_native_client",
        "_native_aclient",
        "chat_transport",
        "http_transport",
        "credential_resolver",
        "api_adapter",
        "api_mode_capabilities",
        "reasoning_codec",
        "_response_cache",
        "__call__",
        "acall",
        # Live catalog references are never serialized; catalog-created
        # models carry only the plain ``_provider_source`` metadata dict.
        "provider_catalog",
        "_provider_catalog",
    ]
    batch_support: bool = False  # Whether model supports batch processing

    def instance_type(self) -> Mapping[str, str]:
        return {"model_type": self.model_type}

    def get_model_info(self) -> Mapping[str, str]:
        return {
            "model_id": self.model_id,
            "provider": self.provider,
        }

    def serialize(self):
        data = super().serialize()
        source = self.__dict__.get("_provider_source")
        if source is not None:
            # State is a shallow copy; hand out an independent plain-data
            # copy so callers cannot mutate the instance's metadata and no
            # live catalog/class object can leak into serialized output.
            state = data.get("state")
            if isinstance(state, dict):
                state["_provider_source"] = copy.deepcopy(source)
        return data

    @classmethod
    def from_serialized(cls, state: Mapping):
        if isinstance(state, Mapping) and state.get("_provider_source") is not None:
            from msgflux.models.catalog import (  # noqa: PLC0415
                get_current_provider_catalog,
                validate_serialized_source,
            )

            validate_serialized_source(
                getattr(cls, "provider", None) or "",
                state,
                get_current_provider_catalog(),
            )
        return super().from_serialized(state)

    @abstractmethod
    def _initialize(self):
        """Initialize the class. This method must be implemented by subclasses.

        This method is called during the deserialization process to ensure that
        the client is properly initialized after its state has been restored.

        Raises:
            NotImplementedError:
                If the method is not implemented by the subclass.
        """
        raise NotImplementedError
