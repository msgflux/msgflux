from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Literal, Mapping, Protocol
from uuid import uuid4

from msgflux.runtime.context import ExecutionScope, get_execution_scope
from msgflux.vulcano.events import EventDraft, EventType

__all__ = [
    "ExtensionPermissionApi",
    "PermissionActionDecision",
    "PermissionDecision",
    "PermissionManager",
    "PermissionRequest",
    "PermissionResult",
    "PermissionSource",
]


PermissionActionDecision = Literal["allow_once", "allow_session", "deny"]
PermissionDecision = Literal[
    "allow_once",
    "allow_session",
    "deny",
    "cancelled",
]
PermissionSource = Literal["user", "session", "headless", "cancelled"]
PermissionEventEmitter = Callable[
    [EventDraft, str | None],
    Awaitable[None],
]


class _ActiveAssertion(Protocol):
    def __call__(self) -> None: ...


@dataclass(frozen=True)
class PermissionRequest:
    """Serializable description of one privileged operation."""

    request_id: str
    owner: str
    operation: str
    description: str
    resource: str | None
    remember_key: str
    allow_session: bool
    scope: ExecutionScope
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_payload(self, *, requires_confirmation: bool) -> dict[str, object]:
        options: list[PermissionActionDecision] = ["allow_once"]
        if self.allow_session:
            options.append("allow_session")
        options.append("deny")
        return {
            "request_id": self.request_id,
            "owner": self.owner,
            "operation": self.operation,
            "description": self.description,
            "resource": self.resource,
            "remember_key": self.remember_key,
            "options": options,
            "scope": self.scope.to_dict(),
            "metadata": dict(self.metadata),
            "requires_confirmation": requires_confirmation,
        }


@dataclass(frozen=True)
class PermissionResult:
    """Decision returned to a command, tool, hook, or background task."""

    request: PermissionRequest
    decision: PermissionDecision
    source: PermissionSource

    @property
    def allowed(self) -> bool:
        return self.decision in {"allow_once", "allow_session"}

    @property
    def remembered(self) -> bool:
        return self.decision == "allow_session"

    def to_payload(self) -> dict[str, object]:
        payload = self.request.to_payload(requires_confirmation=False)
        payload.update(
            {
                "decision": self.decision,
                "source": self.source,
                "allowed": self.allowed,
                "remembered": self.remembered,
            }
        )
        return payload


@dataclass
class _PendingPermission:
    request: PermissionRequest
    future: asyncio.Future[PermissionDecision]


class PermissionManager:
    """Runtime-owned broker between privileged operations and clients."""

    def __init__(
        self,
        emit: PermissionEventEmitter,
        *,
        interactive: Callable[[], bool],
    ) -> None:
        self._emit = emit
        self._interactive = interactive
        self._pending: dict[str, _PendingPermission] = {}
        self._session_grants: set[tuple[str, str, str]] = set()

    @property
    def pending(self) -> tuple[PermissionRequest, ...]:
        return tuple(item.request for item in self._pending.values())

    async def request(
        self,
        owner: str,
        operation: str,
        description: str,
        *,
        resource: str | None = None,
        remember_key: str | None = None,
        allow_session: bool = True,
        metadata: Mapping[str, object] | None = None,
        scope: ExecutionScope | None = None,
        correlation_id: str | None = None,
    ) -> PermissionResult:
        resolved_owner = _required_text(owner, "Permission owner")
        resolved_operation = _required_text(operation, "Permission operation")
        resolved_description = _required_text(
            description,
            "Permission description",
        )
        if scope is not None and not isinstance(scope, ExecutionScope):
            raise TypeError("scope must be an ExecutionScope or None")
        resolved_scope = scope or get_execution_scope()
        resolved_resource = resource if resource is None else str(resource)
        resolved_key = (
            _required_text(remember_key, "Permission remember key")
            if remember_key is not None
            else f"{resolved_operation}:{resolved_resource or resolved_description}"
        )
        request = PermissionRequest(
            request_id=uuid4().hex,
            owner=resolved_owner,
            operation=resolved_operation,
            description=resolved_description,
            resource=resolved_resource,
            remember_key=resolved_key,
            allow_session=allow_session,
            scope=resolved_scope,
            metadata=dict(metadata or {}),
        )
        grant_key = (
            resolved_scope.thread_id or "",
            resolved_owner,
            resolved_key,
        )

        if grant_key in self._session_grants:
            await self._emit_requested(
                request,
                correlation_id,
                requires_confirmation=False,
            )
            result = PermissionResult(request, "allow_session", "session")
            await self._emit_resolved(result, correlation_id)
            return result

        if not self._interactive():
            await self._emit_requested(
                request,
                correlation_id,
                requires_confirmation=False,
            )
            result = PermissionResult(request, "deny", "headless")
            await self._emit_resolved(result, correlation_id)
            return result

        loop = asyncio.get_running_loop()
        pending = _PendingPermission(
            request=request,
            future=loop.create_future(),
        )
        self._pending[request.request_id] = pending
        await self._emit_requested(
            request,
            correlation_id,
            requires_confirmation=True,
        )
        try:
            decision = await pending.future
        except asyncio.CancelledError:
            result = PermissionResult(request, "cancelled", "cancelled")
            await self._emit_resolved(result, correlation_id)
            raise
        finally:
            self._pending.pop(request.request_id, None)

        if decision == "allow_session":
            self._session_grants.add(grant_key)
        source: PermissionSource = "cancelled" if decision == "cancelled" else "user"
        result = PermissionResult(request, decision, source)
        await self._emit_resolved(result, correlation_id)
        return result

    def resolve(
        self,
        request_id: str,
        decision: PermissionActionDecision,
    ) -> bool:
        if decision not in {"allow_once", "allow_session", "deny"}:
            raise ValueError(f"Unsupported permission decision: {decision!r}")
        pending = self._pending.get(request_id)
        if pending is None or pending.future.done():
            return False
        if decision == "allow_session" and not pending.request.allow_session:
            raise ValueError("This permission cannot be remembered for the session")
        pending.future.set_result(decision)
        return True

    def cancel_all(self) -> None:
        for pending in tuple(self._pending.values()):
            if not pending.future.done():
                pending.future.set_result("cancelled")

    async def _emit_requested(
        self,
        request: PermissionRequest,
        correlation_id: str | None,
        *,
        requires_confirmation: bool,
    ) -> None:
        await self._emit(
            EventDraft(
                EventType.PERMISSION_REQUESTED,
                request.to_payload(
                    requires_confirmation=requires_confirmation,
                ),
            ),
            correlation_id,
        )

    async def _emit_resolved(
        self,
        result: PermissionResult,
        correlation_id: str | None,
    ) -> None:
        await self._emit(
            EventDraft(EventType.PERMISSION_RESOLVED, result.to_payload()),
            correlation_id,
        )


class ExtensionPermissionApi:
    """Owner-aware permission facade exposed by ExtensionApi."""

    def __init__(
        self,
        manager: PermissionManager,
        *,
        owner: str,
        assert_active: _ActiveAssertion,
    ) -> None:
        self._manager = manager
        self._owner = owner
        self._assert_active = assert_active

    async def request(
        self,
        operation: str,
        description: str,
        *,
        resource: str | None = None,
        remember_key: str | None = None,
        allow_session: bool = True,
        metadata: Mapping[str, object] | None = None,
        scope: ExecutionScope | None = None,
        correlation_id: str | None = None,
    ) -> PermissionResult:
        self._assert_active()
        result = await self._manager.request(
            self._owner,
            operation,
            description,
            resource=resource,
            remember_key=remember_key,
            allow_session=allow_session,
            metadata=metadata,
            scope=scope,
            correlation_id=correlation_id,
        )
        self._assert_active()
        return result


def _required_text(value: str, label: str) -> str:
    resolved = str(value).strip()
    if not resolved:
        raise ValueError(f"{label} cannot be empty")
    return resolved
