"""Lazy loading of the optional AgentService server dependency."""

from msgflux.runtime.service import AgentService


def create_service_app(
    service: AgentService,
    *,
    token: str,
    close_service: bool = False,
    event_buffer_limit: int | None = 1024,
):
    """Create the Litestar app; install msgflux[service] for server support."""
    try:
        from msgflux.runtime.service.http.app import (  # noqa: PLC0415
            create_service_app as create_app,
        )
    except ModuleNotFoundError as error:
        if error.name and error.name.startswith("litestar"):
            raise ImportError(
                "Install server support with: uv add 'msgflux[service]'"
            ) from error
        raise
    return create_app(
        service,
        token=token,
        close_service=close_service,
        event_buffer_limit=event_buffer_limit,
    )
