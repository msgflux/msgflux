from msgflux.models.model import Model
from msgflux.utils.imports import autoload_package

autoload_package("msgflux.models.providers")

__all__ = ["Model"]
