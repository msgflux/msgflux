from abc import abstractmethod
from typing import Dict

from msgflux._private.client import BaseClient


class BaseDB(BaseClient):
    msgflux_type = "db"
    to_ignore = ["client", "documents"]

    def instance_type(self) -> Dict[str, str]:
        return {"db_type": self.db_type}

    @abstractmethod
    def add(self):
        """Insert data on database."""
        raise NotImplementedError

    @abstractmethod
    def _initialize(self):
        """Initialize the class. This method must be implemented by subclasses.

        This method is called during the deserialization process to ensure that the client
        is properly initialized after its state has been restored.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError
