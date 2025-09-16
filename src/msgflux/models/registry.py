from typing import Type
from msgflux.models.base import BaseModel

model_registry = {} # model_registry[model_type][provider] = cls


def register_model(cls: Type[BaseModel]):
    model_type = getattr(cls, "model_type", None)
    provider = getattr(cls, "provider", None)

    if not model_type or not provider:
        raise ValueError(f"{cls.__name__} must define `model_type` and `provider`.")

    model_registry.setdefault(model_type, {})[provider] = cls
    return cls
