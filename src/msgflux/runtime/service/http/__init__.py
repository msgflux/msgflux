"""Native AgentService HTTP/SSE transport; server dependencies are optional."""

from msgflux.runtime.service.http.client import (
    AgentServiceClient,
    AgentServiceHTTPError,
    AgentServiceProtocolError,
    RemoteThreadWatcher,
)
from msgflux.runtime.service.http.factory import create_service_app
from msgflux.runtime.service.http.records import (
    EventRecord,
    HealthRecord,
    SnapshotRecord,
)
from msgflux.runtime.service.http.session import AgentSessionClient

__all__ = [
    "AgentServiceClient",
    "AgentSessionClient",
    "AgentServiceHTTPError",
    "AgentServiceProtocolError",
    "RemoteThreadWatcher",
    "create_service_app",
    "EventRecord",
    "SnapshotRecord",
    "HealthRecord",
]
