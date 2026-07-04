from typing import Any

from msgflux.utils.imports import AutoloadRegistry

task_store_registry = AutoloadRegistry("msgflux.tasks.providers")


def register_task_store(cls: type[Any]):
    task_store_type = getattr(cls, "task_store_type", None)
    provider = getattr(cls, "provider", None)

    if not task_store_type or not provider:
        raise ValueError(
            f"{cls.__name__} must define `task_store_type` and `provider`."
        )

    task_store_registry.setdefault(task_store_type, {})[provider] = cls
    return cls
