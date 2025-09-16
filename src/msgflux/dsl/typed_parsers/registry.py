from typing import Type
from msgflux.dsl.typed_parsers.base import BaseTypedParser

typed_parser_registry = {} # typed_parser_registry[type] = cls

def register_typed_parser(cls: Type[BaseTypedParser]):
    key = getattr(cls, "typed_parser_type")
    typed_parser_registry[key] = cls
    return cls
