"""Compose configured extensions with a trusted custom session factory."""

from __future__ import annotations

import inspect
from functools import partial

from msgflux.nn.extensions.installation import AgentExtensionInstallation
from msgflux.runtime.service.api import AgentSession
from msgflux.runtime.service.extension_controls import own_installation
from msgflux.utils.imports import _factory_import_root


def with_extensions(factory, names, catalog, *, import_root):
    """Keep custom Agent construction lazy and retain its cleanup ownership."""
    return partial(_create_session, factory, tuple(names), catalog, import_root)


async def _create_session(factory, names, catalog, import_root, thread):
    session = factory(thread)
    if inspect.isawaitable(session):
        session = await session
    if not isinstance(session, AgentSession):
        raise TypeError("factory must return AgentSession")
    installation = AgentExtensionInstallation(catalog, session.agent)
    session.extension_catalog = catalog
    session.extension_import_root = import_root
    own_installation(session, installation)
    try:
        with _factory_import_root(import_root):
            installation.install(names)
    except BaseException as error:
        try:
            await session.on_close()
        except BaseException as cleanup_error:
            error.add_note(f"Extension cleanup failed: {cleanup_error}")
        raise
    return session
