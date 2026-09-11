"""Durable conversation context scope transitions."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from msgflux.chat_messages import ChatMessages
from msgflux.runtime.agent_run import AgentRun, get_agent_run

_SCOPE_KEY = "context_scopes"
_RUNTIME_KEY = "runtime"
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")


@dataclass(frozen=True)
class ContextScopeCommand:
    """Typed command emitted by a scope tool and applied at a settled boundary."""

    action: str
    name: str | None = None
    summary: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": "context_scope_transition",
            "action": self.action,
            "name": self.name,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class ScopeTransition:
    action: str
    scope: str
    branch_id: str
    changed: bool
    revision: int
    summary: str | None = None


class ContextScopeConflictError(RuntimeError):
    """Raised when a scope transition uses stale durable context metadata."""


class ContextScopeController:
    """Manage nested branch views inside one Agent run.

    A transition is deliberately local to a ``ChatMessages`` object. It does
    not allocate a new execution scope, thread, run, or budget. The caller is
    responsible for invoking it only after all tool calls and outputs in a
    batch have settled.
    """

    def __init__(self, *, run: AgentRun | None = None) -> None:
        self.run = run or get_agent_run()

    @staticmethod
    def _metadata(messages: ChatMessages) -> dict[str, Any]:
        from msgflux.chat_messages import ChatMessages  # noqa: PLC0415

        if not isinstance(messages, ChatMessages):
            raise TypeError("Context scopes require a ChatMessages history")
        runtime = messages.metadata.setdefault(_RUNTIME_KEY, {})
        if not isinstance(runtime, dict):
            raise ValueError("ChatMessages runtime metadata is corrupted")
        scopes = runtime.setdefault(_SCOPE_KEY, {})
        if not isinstance(scopes, dict):
            raise ValueError("ChatMessages context scope metadata is corrupted")
        scopes.setdefault("revision", 0)
        scopes.setdefault("active", "root")
        scopes.setdefault("stack", [])
        scopes.setdefault(
            "branches",
            {
                "root": {
                    "parent": None,
                    "name": "root",
                    "items": deepcopy(messages._items),
                }
            },
        )
        return scopes

    @classmethod
    def restore(
        cls, messages: ChatMessages, *, run: AgentRun | None = None
    ) -> ContextScopeController:
        controller = cls(run=run)
        scopes = cls._metadata(messages)
        active = scopes["active"]
        branch = scopes["branches"].get(active)
        if not isinstance(branch, Mapping) or not isinstance(branch.get("items"), list):
            raise ValueError(f"Context branch `{active}` is corrupted")
        # The checkpoint already contains the active branch timeline. Branch
        # snapshots are metadata used only when a transition needs to swap views.
        if controller.run is not None:
            controller.run.branch_id = active
        return controller

    @staticmethod
    def _validate_name(name: str) -> str:
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            raise ValueError("Scope names must match [A-Za-z][A-Za-z0-9_-]{0,63}")
        return name

    @classmethod
    def _check_revision(
        cls, scopes: Mapping[str, Any], expected_revision: int | None
    ) -> None:
        if (
            expected_revision is not None
            and scopes.get("revision", 0) != expected_revision
        ):
            raise ContextScopeConflictError(
                f"Context scope revision conflict: expected {expected_revision}, "
                f"found {scopes.get('revision', 0)}."
            )

    @classmethod
    def _sync_active(cls, messages: ChatMessages, scopes: dict[str, Any]) -> None:
        active = scopes["active"]
        scopes["branches"][active]["items"] = deepcopy(messages._items)

    def open(
        self,
        messages: ChatMessages,
        name: str,
        *,
        summary: str | None = None,
        expected_revision: int | None = None,
    ) -> ScopeTransition:
        name = self._validate_name(name)
        scopes = self._metadata(messages)
        self._check_revision(scopes, expected_revision)
        active = scopes["active"]
        if active != "root" and scopes["stack"] and scopes["stack"][-1]["name"] == name:
            return ScopeTransition(
                "open", name, active, False, scopes["revision"], summary
            )
        branch_id = f"{active}/{name}" if active != "root" else name
        if branch_id in scopes["branches"]:
            raise ContextScopeConflictError(
                f"Context branch `{branch_id}` already exists"
            )
        if summary:
            messages.add_assistant_response(content=summary)
        self._sync_active(messages, scopes)
        parent_items = deepcopy(messages._items)
        scopes["stack"].append({"name": name, "branch_id": branch_id, "parent": active})
        scopes["branches"][branch_id] = {
            "parent": active,
            "name": name,
            "items": parent_items,
            "origin_item_id": parent_items[-1].get("item_id") if parent_items else None,
        }
        scopes["active"] = branch_id
        scopes["revision"] += 1
        messages.metadata[_RUNTIME_KEY][_SCOPE_KEY] = scopes
        messages._items = parent_items
        self._sync_active(messages, scopes)
        if self.run is not None:
            self.run.branch_id = branch_id
        return ScopeTransition(
            "open", name, branch_id, True, scopes["revision"], summary
        )

    def close(
        self,
        messages: ChatMessages,
        name: str | None = None,
        *,
        summary: str | None = None,
        expected_revision: int | None = None,
    ) -> ScopeTransition:
        scopes = self._metadata(messages)
        self._check_revision(scopes, expected_revision)
        active = scopes["active"]
        if name is not None and any(
            branch.get("name") == name
            and branch.get("parent") == active
            and branch.get("closed") is True
            for branch in scopes["branches"].values()
        ):
            return ScopeTransition(
                "close", name, active, False, scopes["revision"], summary
            )
        if active == "root":
            if name in (None, "root"):
                return ScopeTransition(
                    "close", name or "root", "root", False, scopes["revision"], summary
                )
            raise ContextScopeConflictError(f"Context scope `{name}` is not active")
        frame = scopes["stack"][-1]
        if name is not None and name != frame["name"]:
            raise ContextScopeConflictError(
                f"Cannot close scope `{name}` while `{frame['name']}` is active"
            )
        self._sync_active(messages, scopes)
        parent = frame["parent"]
        parent_branch = scopes["branches"].get(parent)
        if not isinstance(parent_branch, dict):
            raise ValueError(f"Parent context branch `{parent}` is missing")
        scopes["stack"].pop()
        scopes["active"] = parent
        # Keep closed branches durable for inspection and recovery.
        scopes["branches"][active]["closed"] = True
        scopes["revision"] += 1
        messages._items = deepcopy(parent_branch["items"])
        close_summary = summary or f"Context scope `{frame['name']}` completed."
        messages.add_assistant_response(content=close_summary)
        self._sync_active(messages, scopes)
        messages.metadata[_RUNTIME_KEY][_SCOPE_KEY] = scopes
        if self.run is not None:
            self.run.branch_id = parent
        return ScopeTransition(
            "close", frame["name"], parent, True, scopes["revision"], close_summary
        )

    def apply_command(
        self,
        messages: ChatMessages,
        command: ContextScopeCommand | Mapping[str, Any],
        *,
        expected_revision: int | None = None,
    ) -> ScopeTransition:
        if isinstance(command, Mapping):
            if command.get("type") != "context_scope_transition":
                raise ValueError("Unknown context scope command type")
            command = ContextScopeCommand(
                action=command.get("action"),
                name=command.get("name"),
                summary=command.get("summary"),
            )
        if not isinstance(command, ContextScopeCommand):
            raise TypeError("Expected ContextScopeCommand")
        if command.action == "open":
            if command.name is None:
                raise ValueError("Opening a context scope requires a name")
            return self.open(
                messages,
                command.name,
                summary=command.summary,
                expected_revision=expected_revision,
            )
        if command.action == "close":
            return self.close(
                messages,
                command.name,
                summary=command.summary,
                expected_revision=expected_revision,
            )
        raise ValueError(f"Unknown context scope action `{command.action}`")

    @classmethod
    def active_scope(cls, messages: ChatMessages) -> str:
        return cls._metadata(messages)["active"]

    @classmethod
    def durable_state(cls, messages: ChatMessages) -> Mapping[str, Any]:
        return deepcopy(cls._metadata(messages))


__all__ = [
    "ContextScopeCommand",
    "ContextScopeConflictError",
    "ContextScopeController",
    "ScopeTransition",
]
