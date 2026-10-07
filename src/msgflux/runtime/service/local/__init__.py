"""Local process discovery and startup for AgentService backends."""

from msgflux.runtime.service.local.discovery import connect_local_service
from msgflux.runtime.service.local.runner import serve_local_service

__all__ = ["connect_local_service", "serve_local_service"]
