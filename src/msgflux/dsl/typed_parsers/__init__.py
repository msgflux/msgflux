from msgflux.dsl.typed_parsers.registry import register_typed_parser, typed_parser_registry
from msgflux.utils.imports import autoload_package


autoload_package("msgflux.dsl.typed_parsers.providers")

__all__ = ["register_typed_parser", "typed_parser_registry"]
