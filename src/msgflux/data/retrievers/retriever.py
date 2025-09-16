from typing import Any, Mapping, Type

from msgflux.data.retrievers.base import BaseRetriever
from msgflux.data.retrievers.registry import retriever_registry
from msgflux.data.retrievers.types import (
    LexicalRetriever,
    SemanticRetriever,
    WebRetriever,
)


class Retriever:

    @classmethod
    def providers(cls):
        return {k: list(v.keys()) for k, v in retriever_registry.items()}

    @classmethod
    def retriever_types(cls):
        return list(retriever_registry.keys())

    @classmethod
    def _get_retriever_class(cls, retriever_type: str, provider: str) -> Type[BaseRetriever]:
        if retriever_type not in retriever_registry:
            raise ValueError(f"Retriever type `{retriever_type}` is not supported")
        if provider not in retriever_registry[retriever_type]:
            raise ValueError(
                f"Provider `{provider}` not registered for type `{retriever_type}`"
            )
        retriever_cls = retriever_registry[retriever_type][provider]
        return retriever_cls

    @classmethod
    def _create_retriever(
        cls, retriever_type: str, provider: str, **kwargs
    ) -> Type[BaseRetriever]:
        retriever_cls = cls._get_retriever_class(retriever_type, provider)
        return retriever_cls(**kwargs)

    @classmethod
    def from_serialized(
        cls, provider: str, retriever_type: str, params: Mapping[str, Any]
    ) -> Type[BaseRetriever]:
        """Creates a retriever instance from serialized parameters.

        Args:
            provider:
                The retriever provider (e.g., "bm25", "rank_bm25").
            retriever_type:
                The type of retriever (e.g., "lexical", "web").
            params:
                Dictionary containing the serialized retriever parameters.

        Returns:
            An instance of the appropriate retriever class with restored state.
        """
        retriever_cls = cls._get_retriever_class(retriever_type, provider)
        # Create instance without calling __init__
        instance = object.__new__(retriever_cls)
        # Restore the instance state
        instance.from_serialized(params)
        return instance

    @classmethod
    def lexical(cls, provider: str, **kwargs) -> LexicalRetriever:
        return cls._create_retriever("lexical", provider, **kwargs)

    @classmethod
    def semantic(cls, provider: str, **kwargs) -> SemanticRetriever:
        return cls._create_retriever("semantic", provider, **kwargs)

    @classmethod
    def web(cls, provider: str, **kwargs) -> WebRetriever:
        return cls._create_retriever("web", provider, **kwargs)
