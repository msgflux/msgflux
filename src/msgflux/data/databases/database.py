from typing import Any, Dict, Type

from msgflux.data.databases.base import BaseDB
from msgflux.data.databases.types import (
    # GeoDB,
    # GraphDB,
    KVDB,
    # NoSQLDB,
    # RelationalDB,
    # TimeSeriesDB,
    VectorDB,
)
from msgflux.utils.imports import import_module_from_lib

_SUPPORTED_DB_TYPES = [
    "kv",
    "vector",
    # "graph",
    # "relational",
    # "nosql",
    # "time_series",
    # "geo",
]
_DB_NAMESPACE_TRANSLATOR = {
    "cachetools": "CacheTools",
    "diskmanager": "DiskManager",
    "faiss": "FAISS",
    # "qdrant": "Qdrant",
    # "postgres": "Postgres",
    # "weavite": "Weavite",
    # "milvus": "Milvus",
    # "lance_db": "LanceDB",
    # "vespa": "Vespa",
    # "pinecone": "Pinecone",
    # "cassandra": "Cassandra",
    # "redis": "Redis",
    # "deep_lake": "DeepLake",
    # "marqo": "Marqo",
    # "azure": "AzureAISearch",
    # "supabase": "SupaBase",
    # "elastic_search": "ElasticSearch",
    # "chroma": "Chroma",
    # "sqlite": "SQLite",
    # "neo4j": "Neo4J",
}
_VECTOR_DB_PROVIDERS = [
    # "qdrant",
    # "postgres",
    # "weavite",
    # "milvus",
    # "lance_db",
    # "vespa",
    # "redis",
    # "cassandra",
    # "deep_lake",
    # "marqo",
    # "elastic_search",
    # "sqlite",
    "faiss",
    # "chroma",
]
# _GRAPH_DB_PROVIDERS = ["neo4j"]
_KV_DB_PROVIDERS = ["cachetools", "diskmanager"]

_PROVIDERS_BY_DB_TYPE = {
    "kv": _KV_DB_PROVIDERS,
    "vector": _VECTOR_DB_PROVIDERS,
}


class DataBase:
    supported_db_types = _SUPPORTED_DB_TYPES
    providers_by_db_type = _PROVIDERS_BY_DB_TYPE

    @classmethod
    def _get_db_class(cls, db_type: str, provider: str) -> Type[BaseDB]:
        if db_type not in cls.supported_db_types:
            raise ValueError(f"DB type `{db_type}` is not supported")

        providers = cls.providers_by_db_type[db_type]
        if provider not in providers:
            raise ValueError(f"Provider `{provider}` is not supported for {db_type}")

        if len(db_type) <= 3:
            db_type = db_type.upper()
        else:
            db_type = db_type.title().replace("_", "")

        provider_class_name = f"{_DB_NAMESPACE_TRANSLATOR[provider]}{db_type}DB"
        module_name = f"msgflux.data.databases.providers.{provider}"
        return import_module_from_lib(provider_class_name, module_name)

    @classmethod
    def _create_db(cls, db_type: str, provider: str, **kwargs) -> Type[BaseDB]:
        db_cls = cls._get_db_class(db_type, provider)
        return db_cls(**kwargs)

    @classmethod
    def from_serialized(
        cls, provider: str, db_type: str, params: Dict[str, Any]
    ) -> Type[BaseDB]:
        """Creates a db instance from serialized parameters without calling __init__.

        Args:
            provider: The db provider (e.g., "faiss", "cachetools")
            db_type: The type of db (e.g., "vector", "kv")
            params: Dictionary containing the serialized db parameters

        Returns:
            An instance of the appropriate db class with restored state
        """
        db_cls = cls._get_db_class(db_type, provider)
        # Create instance without calling __init__
        instance = object.__new__(db_cls)
        # Restore the instance state
        instance.from_serialized(params)
        return instance

    @classmethod
    def kv(cls, provider: str, **kwargs) -> KVDB:
        return cls._create_db("kv", provider, **kwargs)

    @classmethod
    def vector(cls, provider: str, **kwargs) -> VectorDB:
        return cls._create_db("vector", provider, **kwargs)
