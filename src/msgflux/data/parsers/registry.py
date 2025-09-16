from msgflux.data.parsers.base import BaseParser

parser_registry = {} # parser_registry[parser_type][provider] = cls


def register_parser(cls: type[BaseParser]):
    parser_type = getattr(cls, "parser_type", None)
    provider = getattr(cls, "provider", None)

    if not parser_type or not provider:
        raise ValueError(f"{cls.__name__} must define `parser_type` and `provider`.")

    parser_registry.setdefault(parser_type, {})[provider] = cls
    return cls
