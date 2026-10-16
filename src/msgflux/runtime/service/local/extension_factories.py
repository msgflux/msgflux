"""Compose configured extensions with a trusted custom session factory."""

from __future__ import annotations

import inspect
from functools import partial

from msgflux.nn.extensions.installation import AgentExtensionInstallation
from msgflux.runtime.service.api import AgentSession


def with_extensions(factory, names, catalog):
    """Keep custom Agent construction lazy and retain its cleanup ownership."""
    if not names:
        return factory

    return partial(_create_session, factory, tuple(names), catalog)


async def _create_session(factory, names, catalog, thread):
    session = factory(thread)
    if inspect.isawaitable(session):
        session = await session
    if not isinstance(session, AgentSession):
        raise TypeError("factory must return AgentSession")
    installation = AgentExtensionInstallation(catalog, session.agent)
    original_close = session._host_on_close if session._managed else session.on_close

    async def close():
        try:
            await installation.aclose()
        finally:
            if original_close is not None:
                result = original_close()
                if inspect.isawaitable(result):
                    await result

    if session._managed:
        # Managed sessions settle delegated work before invoking host cleanup.
        session._host_on_close = close
    else:
        session.on_close = close
    try:
        installation.install(names)
    except BaseException as error:
        try:
            await session.on_close()
        except BaseException as cleanup_error:
            error.add_note(f"Extension cleanup failed: {cleanup_error}")
        raise
    return session
