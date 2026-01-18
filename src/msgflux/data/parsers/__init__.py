from msgflux.data.parsers.parser import Parser
from msgflux.utils.imports import autoload_package

autoload_package("msgflux.data.parsers.providers")

__all__ = ["Parser"]
