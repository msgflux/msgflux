from typing import Any

from msgflux.tasks.registry import task_store_registry
from msgflux.tasks.types import InMemoryTaskStoreType, SQLiteTaskStoreType


class TaskStore:
    @classmethod
    def providers(cls):
        return {k: list(v.keys()) for k, v in task_store_registry.items()}

    @classmethod
    def task_store_types(cls):
        return list(task_store_registry.keys())

    @classmethod
    def _get_task_store_class(
        cls,
        task_store_type: str,
        provider: str,
    ) -> type[Any]:
        if task_store_type not in task_store_registry:
            raise ValueError(f"Task store type `{task_store_type}` is not supported")
        if provider not in task_store_registry[task_store_type]:
            raise ValueError(
                f"Provider `{provider}` not registered for type `{task_store_type}`"
            )
        return task_store_registry[task_store_type][provider]

    @classmethod
    def _create_task_store(
        cls,
        task_store_type: str,
        provider: str,
        **kwargs: Any,
    ) -> Any:
        store_cls = cls._get_task_store_class(task_store_type, provider)
        return store_cls(**kwargs)

    @classmethod
    def in_memory(
        cls,
        provider: str = "default",
        **kwargs: Any,
    ) -> InMemoryTaskStoreType:
        return cls._create_task_store("in_memory", provider, **kwargs)

    @classmethod
    def sqlite(
        cls,
        provider: str = "default",
        **kwargs: Any,
    ) -> SQLiteTaskStoreType:
        return cls._create_task_store("sqlite", provider, **kwargs)
