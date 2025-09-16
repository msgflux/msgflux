from msgflux.data.retrievers.base import BaseRetriever

retriever_registry = {} # retriever_registry[retriever_type][provider] = cls


def register_retriever(cls: type[BaseRetriever]):
    retriever_type = getattr(cls, "retriever_type", None)
    provider = getattr(cls, "provider", None)

    if not retriever_type or not provider:
        raise ValueError(f"{cls.__name__} must define `retriever_type` and `provider`.")

    retriever_registry.setdefault(retriever_type, {})[provider] = cls
    return cls
