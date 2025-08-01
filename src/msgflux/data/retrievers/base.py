from typing import Dict

from msgflux._private.client import BaseClient


class BaseRetriever(BaseClient):
    msgflux_type = "retriever"
    to_ignore = ["client", "documents"]

    def instance_type(self) -> Dict[str, str]:
        return {"retriever_type": self.retriever_type}
