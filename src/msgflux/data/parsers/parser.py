from typing import Any, Dict, Type

from msgflux.data.retrievers.base import BaseRetriever
from msgflux.data.retrievers.types import (
    AnyParser,
    DocumentParser,
    EmailParser,
    MediaParser,
    PptxParser,
    WebParser,
    XlsxParser,
)
from msgflux.utils.imports import import_module_from_lib

_SUPPORTED_RETRIEVER_TYPES = [
    # "hybrid"
    "lexical",
    # "semantic",
]

_LEXICAL_RETRIEVER_PROVIDERS = ["bm25"]
_SEMANTIC_RETRIEVER_PROVIDERS = []
_HYBRID_RETRIEVER_PROVIDERS = []

_RETRIEVER_NAMESPACE_TRANSLATOR = {
    "bm25": "BM25",
}

_PROVIDERS_BY_RETRIEVER_TYPE = {
    "lexical": _LEXICAL_RETRIEVER_PROVIDERS,
}


class Parser:
    supported_retriever_types = _SUPPORTED_RETRIEVER_TYPES
    providers_by_retriever_type = _PROVIDERS_BY_RETRIEVER_TYPE

    @classmethod
    def _get_retriever_class(
        cls, retriever_type: str, provider: str
    ) -> Type[BaseRetriever]:
        if retriever_type not in cls.providers_by_retriever_type:
            raise ValueError(f"Retriever type `{retriever_type}` is not supported")

        providers = cls.providers_by_retriever_type[retriever_type]
        if provider not in providers:
            raise ValueError(
                f"Provider `{provider}` is not supported for {retriever_type}"
            )

        if len(db_type) <= 3:
            db_type = db_type.upper()
        else:
            db_type = db_type.title().replace("_", "")

        provider_class_name = (
            f"{_RETRIEVER_NAMESPACE_TRANSLATOR[provider]}{retriever_type}"
        )
        module_name = f"msgflux.data.parsers.providers.{provider}"
        return import_module_from_lib(provider_class_name, module_name)

    @classmethod
    def _create_retriever(
        cls, retriever_type: str, provider: str, **kwargs
    ) -> Type[BaseRetriever]:
        retriever_cls = cls._get_retriever_class(retriever_type, provider)
        return retriever_cls(**kwargs)

    @classmethod
    def from_serialized(
        cls, provider: str, retriever_type: str, params: Dict[str, Any]
    ) -> Type[BaseRetriever]:
        """Creates a retriever instance from serialized parameters without calling __init__.

        Args:
            provider: The retriever provider (e.g., "bm25")
            retriever_type: The type of model (e.g., "lexical", "semantic")
            params: Dictionary containing the serialized db parameters

        Returns:
            An instance of the appropriate retriever class with restored state
        """
        retriever_cls = cls._get_retriever_class(retriever_type, provider)
        # Create instance without calling __init__
        instance = object.__new__(retriever_cls)
        # Restore the instance state
        instance.from_serialized(params)
        return instance

    @classmethod
    def lexical(cls, provider: str, **kwargs) -> Type[LexicalRetriever]:
        return cls._create_retriever("lexical", provider, **kwargs)
