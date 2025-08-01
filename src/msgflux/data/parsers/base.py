from typing import Dict

from msgflux._private.client import BaseClient


class BaseParser(BaseClient):
    msgflux_type = "parser"
    to_ignore = ["client"]

    def instance_type(self) -> Dict[str, str]:
        return {"parser_type": self.parser_type}
