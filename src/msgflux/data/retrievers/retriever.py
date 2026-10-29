from typing import TYPE_CHECKING, Any, Mapping

from msgflux.data.retrievers.registry import retriever_registry

if TYPE_CHECKING:
    from msgflux.data.retrievers.base import BaseRetriever

from msgflux.data.retrievers.types import (
    FuzzyRetriever,
    LexicalRetriever,
    SemanticRetriever,
    WeatherRetriever,
    WebFetchRetriever,
    WebSearchRetriever,
)


class Retriever:
    @classmethod
    def providers(cls):
        return {k: list(v.keys()) for k, v in retriever_registry.items()}

    @classmethod
    def retriever_types(cls):
        return list(retriever_registry.keys())

    @classmethod
    def _get_retriever_class(
        cls, retriever_type: str, provider: str
    ) -> type["BaseRetriever"]:
        # Records serialized before the web search/fetch split used "web".
        if retriever_type == "web":
            retriever_type = "web_search"
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
    ) -> type["BaseRetriever"]:
        retriever_cls = cls._get_retriever_class(retriever_type, provider)
        return retriever_cls(**kwargs)

    @classmethod
    def from_serialized(
        cls, provider: str, retriever_type: str, params: Mapping[str, Any]
    ) -> type["BaseRetriever"]:
        """Creates a retriever instance from serialized parameters.

        Args:
            provider:
                The retriever provider (e.g., "bm25", "rank_bm25").
            retriever_type:
                The type of retriever (e.g., "lexical", "web_search").
            params:
                Dictionary containing the serialized retriever parameters.

        Returns:
            An instance of the appropriate retriever class with restored state.
        """
        retriever_cls = cls._get_retriever_class(retriever_type, provider)
        return retriever_cls.from_serialized(params)

    @classmethod
    def lexical(cls, provider: str, **kwargs) -> LexicalRetriever:
        return cls._create_retriever("lexical", provider, **kwargs)

    @classmethod
    def semantic(cls, provider: str, **kwargs) -> SemanticRetriever:
        return cls._create_retriever("semantic", provider, **kwargs)

    @classmethod
    def web(cls, provider: str, **kwargs) -> WebSearchRetriever:
        """Compatibility alias for :meth:`web_search`."""
        return cls.web_search(provider, **kwargs)

    @classmethod
    def web_search(cls, provider: str, **kwargs) -> WebSearchRetriever:
        """Create a retriever that searches web sources for a query."""
        return cls._create_retriever("web_search", provider, **kwargs)

    @classmethod
    def web_fetch(cls, provider: str, **kwargs) -> WebFetchRetriever:
        """Create a retriever that fetches known URLs as page content."""
        return cls._create_retriever("web_fetch", provider, **kwargs)

    @classmethod
    def weather(cls, provider: str, **kwargs) -> WeatherRetriever:
        return cls._create_retriever("weather", provider, **kwargs)

    @classmethod
    def fuzzy(cls, provider: str, **kwargs) -> FuzzyRetriever:
        return cls._create_retriever("fuzzy", provider, **kwargs)
