from typing import Any, Mapping, Type

from msgflux.data.parsers.base import BaseParser
from msgflux.data.parsers.registry import parser_registry
from msgflux.data.parsers.types import (
    PdfParser,
    PptxParser,
    XlsxParser,
)


class Parser:

    @classmethod
    def providers(cls):
        return {k: list(v.keys()) for k, v in parser_registry.items()}

    @classmethod
    def parser_types(cls):
        return list(parser_registry.keys())

    @classmethod
    def _get_parser_class(cls, parser_type: str, provider: str) -> Type[BaseDB]:
        if parser_type not in parser_registry:
            raise ValueError(f"Parser type `{parser_type}` is not supported")
        if provider not in parser_registry[parser_type]:
            raise ValueError(
                f"Provider `{provider}` not registered for type `{parser_type}`"
            )
        parser_cls = parser_registry[parser_type][provider]
        return parser_cls

    @classmethod
    def _create_parser(
        cls, parser_type: str, provider: str, **kwargs
    ) -> Type[BaseParser]:
        parser_cls = cls._get_parser_class(parser_type, provider)
        return parser_cls(**kwargs)

    @classmethod
    def from_serialized(
        cls, provider: str, parser_type: str, params: Mapping[str, Any]
    ) -> Type[BaseParser]:
        """Creates a parser instance from serialized parameters.

        Args:
            provider:
                The parser provider (e.g., "pypdf").
            parser_type:
                The type of parser (e.g., "pdf", "pptx").
            params:
                Dictionary containing the serialized parser parameters.

        Returns:
            An instance of the appropriate parser class with restored state.
        """
        parser_cls = cls._get_parser_class(retriever_type, provider)
        # Create instance without calling __init__
        instance = object.__new__(parser_cls)
        # Restore the instance state
        instance.from_serialized(params)
        return instance

    @classmethod
    def pdf(cls, provider: str, **kwargs) -> PdfParser:
        return cls._create_parser("pdf", provider, **kwargs)

    @classmethod
    def pptx(cls, provider: str, **kwargs) -> PptxParser:
        return cls._create_parser("pptx", provider, **kwargs)

    @classmethod
    def xlsx(cls, provider: str, **kwargs) -> XlsxParser:
        return cls._create_parser("xlsx", provider, **kwargs)
