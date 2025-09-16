from abc import abstractmethod
from typing import Mapping

from msgflux._private.client import BaseClient


class BaseModel(BaseClient):
    msgflux_type = "model"
    to_ignore = ["_api_key", "model", "processor", "client"]

    def instance_type(self) -> Mapping[str, str]:
        return {"model_type": self.model_type}

    def get_model_info(self) -> Mapping[str, str]:
        return {
            "model_id": self.model_id,
            "provider": self.provider,
        }

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
