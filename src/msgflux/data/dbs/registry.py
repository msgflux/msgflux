from msgflux.data.dbs.base import BaseDB

db_registry = {} # db_registry[db_type][provider] = cls


def register_db(cls: type[BaseDB]):
    db_type = getattr(cls, "db_type", None)
    provider = getattr(cls, "provider", None)

    if not db_type or not provider:
        raise ValueError(f"{cls.__name__} must define `db_type` and `provider`.")

    db_registry.setdefault(db_type, {})[provider] = cls
    return cls
