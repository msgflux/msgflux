from typing import Any, Mapping, Type

from msgflux.data.dbs.base import BaseDB
from msgflux.data.dbs.registry import db_registry
from msgflux.data.dbs.types import KVDB, VectorDB


class DB:

    @classmethod
    def providers(cls):
        return {k: list(v.keys()) for k, v in db_registry.items()}

    @classmethod
    def db_types(cls):
        return list(db_registry.keys())

    @classmethod
    def _get_db_class(cls, db_type: str, provider: str) -> Type[BaseDB]:
        if db_type not in db_registry:
            raise ValueError(f"DB type `{db_type}` is not supported")
        if provider not in db_registry[db_type]:
            raise ValueError(
                f"Provider `{provider}` not registered for type `{db_type}`"
            )
        db_cls = db_registry[db_type][provider]
        return db_cls

    @classmethod
    def _create_db(cls, db_type: str, provider: str, **kwargs) -> Type[BaseDB]:
        db_cls = cls._get_db_class(db_type, provider)
        return db_cls(**kwargs)

    @classmethod
    def from_serialized(
        cls, provider: str, db_type: str, params: Mapping[str, Any]
    ) -> Type[BaseDB]:
        """Creates a db instance from serialized parameters.

        Args:
            provider:
                The db provider (e.g., "faiss", "cachetools").
            db_type:
                The type of db (e.g., "vector", "kv").
            params:
                Dictionary containing the serialized db parameters.

        Returns:
            An instance of the appropriate db class with restored state.
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
