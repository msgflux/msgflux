"""Service-owned extension composition; never an execution-time filter."""

from __future__ import annotations

import asyncio
import inspect
from contextlib import nullcontext
from urllib.parse import quote

from msgflux.nn.extensions.installation import AgentExtensionInstallation
from msgflux.nn.modules.agent import Agent
from msgflux.runtime.service.records import (
    AgentExtensionStatus,
    ExtensionSelection,
    ServiceBusyError,
    ServiceConflictError,
    ServiceRecoveryRequiredError,
    ThreadExtensions,
)
from msgflux.runtime.service.session_cache import _SessionLoadCleanupError
from msgflux.utils.imports import _factory_import_root


def agent_tree(agent, path="root", parent=None, ancestors=frozenset()):
    """Use composition positions, not user-assigned display names, as IDs."""
    if id(agent) in ancestors:
        raise ValueError("Cyclic Agent composition")
    yield path, parent, agent
    ancestors = ancestors | {id(agent)}
    for definition in agent.tool_library.registry.definitions():
        name = definition.name
        child = getattr(definition.executor, "impl", None)
        if isinstance(child, Agent):
            yield from agent_tree(
                child, f"{path}/{quote(name, safe='')}", path, ancestors
            )


def own_installation(session, installation):
    """Retain the existing factory's cleanup, including managed resource fencing."""
    original = session._host_on_close if session._managed else session.on_close

    async def close():
        try:
            await installation.aclose()
        finally:
            if original is not None:
                result = original()
                if inspect.isawaitable(result):
                    await result

    if session._managed:
        session._host_on_close = close
    else:
        session.on_close = close


def available_extensions(session):
    catalog = session.extension_catalog
    names = {info.name for info in catalog.list()} if catalog is not None else set()
    for values in session._extension_available.values():
        names.update(values)
    return names


def _unfinished_descendants(service, thread_id, session):
    stores = (
        {id(session.task_store): session.task_store}
        if session.task_store is not None
        else {}
    )
    for _path, _parent, agent in agent_tree(session.agent):
        store = getattr(agent.tool_library, "_task_store", None)
        if store is not None:
            stores[id(store)] = store
    # Inspect unfinished tasks only. Descendants can use a different thread ID
    # but retain the foreground root_run_id. No completed trajectory is loaded.
    for store in stores.values():
        for status in ("queued", "running", "paused"):
            for task in store.list(status=status):
                root_run = task.metadata.get("root_run_id")
                if task.metadata.get("thread_id") == thread_id or (
                    isinstance(root_run, str)
                    and service.store.get_for_run(thread_id, root_run) is not None
                ):
                    return "A descendant background task has not settled"
    return None


def busy_reason(service, thread_id, session):
    reason = service._release_reason(thread_id, session, mark_closing=False)
    if reason is not None:
        return reason
    for _path, _parent, agent in agent_tree(session.agent):
        with agent._resource_lock:
            if any(agent._thread_activity.values()):
                return "An Agent or subagent call is still active"
        reason = agent._active_background_future_reason((agent.tool_library,))
        if reason is not None:
            return reason
    record = service.store.latest_admission(thread_id)
    if record is not None and record.receipt.status in {
        "accepted",
        "running",
        "paused",
    }:
        return "The thread has unfinished admitted work"
    return _unfinished_descendants(service, thread_id, session)


def inventory(service, thread_id, session):
    blocked = busy_reason(service, thread_id, session)
    return ThreadExtensions(
        thread_id=thread_id,
        revision=session._extension_revision,
        agents=tuple(
            AgentExtensionStatus(
                path,
                agent.get_module_name(),
                parent,
                "busy" if blocked is not None else "idle",
                tuple(sorted(agent.extensions.keys())),
            )
            for path, parent, agent in agent_tree(session.agent)
        ),
        available=tuple(sorted(available_extensions(session))),
    )


def names(value, label):
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(name, str) or not name.strip() for name in value
    ):
        raise ValueError(f"{label} must contain non-empty names")
    if len(set(value)) != len(value):
        raise ValueError(f"Duplicate {label}")
    return set(value)


def desired_selection(session, *, scope, agent_ids, enable, disable):
    enable = names(enable, "enable")
    disable = names(disable, "disable")
    if not enable and not disable:
        raise ValueError("Provide at least one extension to enable or disable")
    if enable & disable:
        raise ValueError("An extension cannot be enabled and disabled together")
    tree = {path: agent for path, _parent, agent in agent_tree(session.agent)}
    if scope == "tree" and agent_ids is None:
        targets = set(tree)
    elif scope is None and agent_ids is not None:
        targets = names(agent_ids, "agent_ids")
        if not targets or targets - tree.keys():
            raise ValueError(
                "Unknown or empty agent_ids; inspect the thread extensions first"
            )
    else:
        raise ValueError("Use scope='tree' or agent_ids, exclusively")
    available = (
        {info.name for info in session.extension_catalog.list()}
        if session.extension_catalog is not None
        else set()
    )
    if disable - available_extensions(session):
        raise ValueError("Unknown registered extension")
    # A fresh factory can reconstruct its own extensions even without a catalog.
    for path in targets:
        known = available | set(session._extension_available.get(path, ()))
        if enable - known:
            raise ValueError(f"Unknown registered extension for Agent {path!r}")
    selections = {
        path: tuple(sorted((set(agent.extensions) | enable) - disable))
        if path in targets
        else tuple(sorted(agent.extensions))
        for path, agent in tree.items()
    }
    return selections, targets


async def apply_selection(session, selection, *, strict=True):
    """Compose only fresh, unpublished sessions using existing lifecycle APIs."""
    session._extension_revision = selection.revision
    if not selection.agents:
        return
    seen = set()
    for path, _parent, agent in agent_tree(session.agent):
        selected = selection.agents.get(path)
        if selected is None:
            if strict:
                raise ServiceRecoveryRequiredError(
                    "The saved Agent composition changed"
                )
            continue
        seen.add(path)
        current = set(agent.extensions)
        for name in current - set(selected):
            await agent.aremove_extension(name)
        added = set(selected) - current
        if added:
            catalog = session.extension_catalog
            if catalog is None:
                raise ValueError(
                    "The host must register a catalog to enable this extension"
                )
            installation = AgentExtensionInstallation(catalog, agent)
            own_installation(session, installation)
            with (
                _factory_import_root(session.extension_import_root)
                if session.extension_import_root is not None
                else nullcontext()
            ):
                installation.install(sorted(added))
    if strict and seen != set(selection.agents):
        raise ServiceRecoveryRequiredError("The saved Agent composition changed")


async def update_extensions(
    service,
    thread_id,
    *,
    scope=None,
    agent_ids=None,
    enable=(),
    disable=(),
    expected_revision=None,
):
    if expected_revision is not None and (
        type(expected_revision) is not int or expected_revision < 0
    ):
        raise ValueError("expected_revision must be a non-negative integer or None")
    async with service._lock:
        service._require_open()
        service._require_no_maintenance(thread_id)
        service.store.thread(thread_id)
        task = service._cache._create_task(
            _update(
                service, thread_id, scope, agent_ids, enable, disable, expected_revision
            )
        )
        service._extension_updates[thread_id] = task
    return await asyncio.shield(task)


async def _close(session):
    if session.on_close is not None:
        result = session.on_close()
        if inspect.isawaitable(result):
            await result


async def _prepare_update(
    service, thread_id, scope, agent_ids, enable, disable, expected
):
    lease = await service._cache.acquire(thread_id)
    try:
        old = lease.session
        current = service.store.extension_selection(thread_id)
        if old._extension_revision != current.revision:
            raise ServiceConflictError("The loaded extension configuration changed")
        if expected is not None and expected != current.revision:
            raise ServiceConflictError("The thread extension configuration changed")
        reason = busy_reason(service, thread_id, old)
        if reason is not None:
            raise ServiceBusyError(reason)
        desired, targets = desired_selection(
            old, scope=scope, agent_ids=agent_ids, enable=enable, disable=disable
        )
        actual = {
            path: tuple(sorted(agent.extensions))
            for path, _parent, agent in agent_tree(old.agent)
        }
        return (
            current,
            desired,
            targets,
            (inventory(service, thread_id, old) if desired == actual else None),
            tuple(agent for _path, _parent, agent in agent_tree(old.agent)),
        )
    finally:
        await lease.aclose()


def _reconstructed_selection(
    service, thread_id, candidate, old_agents, targets, revision
):
    if candidate.namespace != old_agents[0].get_module_name():
        raise ServiceConflictError("Factories must preserve the thread namespace")
    if any(
        agent is old
        for _path, _parent, agent in agent_tree(candidate.agent)
        for old in old_agents
    ):
        raise ValueError("Factories must reconstruct fresh Agents and subagents")
    reason = busy_reason(service, thread_id, candidate)
    if reason is not None:
        raise ServiceBusyError(reason)
    composed = {
        path: tuple(sorted(agent.extensions))
        for path, _parent, agent in agent_tree(candidate.agent)
    }
    if not targets.issubset(composed):
        raise ValueError("A targeted Agent was removed by this extension change")
    return ExtensionSelection(thread_id, revision, composed)


async def _update(service, thread_id, scope, agent_ids, enable, disable, expected):
    candidate = None
    published = False
    try:
        current, desired, targets, unchanged, old_agents = await _prepare_update(
            service, thread_id, scope, agent_ids, enable, disable, expected
        )
        if unchanged is not None:
            return unchanged
        # The cache rejects other pins, fences background admission and joins cleanup.
        await service._cache.release(thread_id)
        proposed = ExtensionSelection(thread_id, current.revision, desired)
        candidate = await service._create_session(
            thread_id, extension_selection=proposed, strict_extensions=False
        )
        proposed = _reconstructed_selection(
            service,
            thread_id,
            candidate,
            old_agents,
            targets if scope is None else {"root"},
            current.revision,
        )
        # Admit no callers until publication and the SQLite CAS both finish.
        await service._cache.install(thread_id, candidate)
        published = True
        service._notify_before_write()
        saved = service.store.update_extension_selection(
            proposed, expected_revision=current.revision
        )
        candidate._extension_revision = saved.revision
        return inventory(service, thread_id, candidate)
    except _SessionLoadCleanupError as error:
        await service._cache.install(thread_id, error.session, close_error=error.error)
        raise ServiceRecoveryRequiredError(
            "Extension cleanup failed; session quarantined"
        ) from error
    except BaseException:
        if published:
            await service._cache.release(thread_id)
        elif candidate is not None:
            try:
                await _close(candidate)
            except BaseException as error:
                await service._cache.install(thread_id, candidate, close_error=error)
                raise ServiceRecoveryRequiredError(
                    "Extension cleanup failed; session quarantined"
                ) from error
        raise
    finally:
        service._extension_updates.pop(thread_id, None)
