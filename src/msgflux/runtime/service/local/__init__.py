"""Local process discovery and startup for AgentService backends."""

from msgflux.runtime.service.local.discovery import (
    connect_local_service,
    inspect_local_service,
    restart_local_service,
    stop_local_service,
)
from msgflux.runtime.service.local.records import LocalServiceStatus
from msgflux.runtime.service.local.runner import serve_local_service

__all__ = [
    "LocalServiceStatus",
    "connect_local_service",
    "inspect_local_service",
    "restart_local_service",
    "stop_local_service",
    "serve_local_service",
]
