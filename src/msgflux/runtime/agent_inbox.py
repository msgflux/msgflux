from __future__ import annotations

import json
import sqlite3
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping
from uuid import uuid4
from xml.sax.saxutils import escape

from msgflux.data.stores.registry import register_store
from msgflux.runtime.context import ExecutionScope
from msgflux.utils.console import cprint

# --- Module Utilities ---


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- Notification Model ---


@dataclass
class AgentNotification:
    notification_id: str
    source: str
    ref: str | None = None
    status: str | None = None
    hint: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    dedupe_key: str | None = None
    created_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AgentInboxStore(ABC):
    """Persistent storage boundary for pending agent inbox notifications."""

    @abstractmethod
    def load_notifications(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
    ) -> List[Mapping[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def save_notifications(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
        notifications: Iterable[Mapping[str, Any]],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def clear(
        self,
        namespace: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        *,
        older_than: float | None = None,
    ) -> int:
        raise NotImplementedError


@register_store("agent_inbox", "in_memory")
class InMemoryAgentInboxStore(AgentInboxStore):
    """In-memory inbox store for tests and local prototyping."""

    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
        self._lock = RLock()

    def _get_run(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
    ) -> Dict[str, Any] | None:
        return self._data.get(namespace, {}).get(session_id, {}).get(run_id)

    def load_notifications(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
    ) -> List[Mapping[str, Any]]:
        with self._lock:
            run = self._get_run(namespace, session_id, run_id)
            if run is None:
                return []
            return deepcopy(run["notifications"])

    def save_notifications(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
        notifications: Iterable[Mapping[str, Any]],
    ) -> None:
        with self._lock:
            ns = self._data.setdefault(namespace, {})
            session = ns.setdefault(session_id, {})
            existing = session.get(run_id)
            created_at = existing["created_at"] if existing else time.time()
            session[run_id] = {
                "notifications": deepcopy([dict(n) for n in notifications]),
                "created_at": created_at,
                "updated_at": time.time(),
            }

    def clear(
        self,
        namespace: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        *,
        older_than: float | None = None,
    ) -> int:
        cutoff = time.time() - older_than if older_than is not None else None
        removed = 0
        with self._lock:
            namespaces = (
                [namespace] if namespace is not None else list(self._data.keys())
            )
            for ns in namespaces:
                ns_data = self._data.get(ns)
                if ns_data is None:
                    continue
                sessions = (
                    [session_id] if session_id is not None else list(ns_data.keys())
                )
                for sid in sessions:
                    session = ns_data.get(sid)
                    if session is None:
                        continue
                    run_ids = [run_id] if run_id is not None else list(session.keys())
                    for rid in run_ids:
                        run = session.get(rid)
                        if run is None:
                            continue
                        if cutoff is not None and run["updated_at"] >= cutoff:
                            continue
                        del session[rid]
                        removed += 1
                    if not session:
                        del ns_data[sid]
                if not ns_data:
                    del self._data[ns]
        return removed


_CREATE_INBOX_TABLES = """\
CREATE TABLE IF NOT EXISTS agent_inboxes (
    namespace      TEXT NOT NULL,
    session_id     TEXT NOT NULL,
    run_id         TEXT NOT NULL,
    notifications  TEXT NOT NULL,
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL,
    PRIMARY KEY (namespace, session_id, run_id)
);

CREATE INDEX IF NOT EXISTS idx_agent_inboxes_session
    ON agent_inboxes(namespace, session_id, updated_at DESC);
"""

_UPSERT_INBOX = """\
INSERT INTO agent_inboxes
    (namespace, session_id, run_id, notifications, created_at, updated_at)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(namespace, session_id, run_id) DO UPDATE SET
    notifications = excluded.notifications,
    updated_at = excluded.updated_at
"""


@register_store("agent_inbox", "sqlite")
class SQLiteAgentInboxStore(AgentInboxStore):
    """SQLite-backed inbox store."""

    def __init__(self, path: str = ".msgflux/agent-inboxes.sqlite3") -> None:
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_CREATE_INBOX_TABLES)
        self._conn.commit()

    @staticmethod
    def _serialize(notifications: Iterable[Mapping[str, Any]]) -> str:
        return json.dumps(list(notifications), ensure_ascii=False, default=str)

    @staticmethod
    def _deserialize(text: str) -> List[Mapping[str, Any]]:
        data = json.loads(text)
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, Mapping)]

    def load_notifications(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
    ) -> List[Mapping[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT notifications FROM agent_inboxes
                WHERE namespace=? AND session_id=? AND run_id=?
                """,
                (namespace, session_id, run_id),
            ).fetchone()
        if row is None:
            return []
        return self._deserialize(row[0])

    def save_notifications(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
        notifications: Iterable[Mapping[str, Any]],
    ) -> None:
        with self._lock:
            now = time.time()
            created_at = self._conn.execute(
                """
                SELECT created_at FROM agent_inboxes
                WHERE namespace=? AND session_id=? AND run_id=?
                """,
                (namespace, session_id, run_id),
            ).fetchone()
            self._conn.execute(
                _UPSERT_INBOX,
                (
                    namespace,
                    session_id,
                    run_id,
                    self._serialize(notifications),
                    created_at[0] if created_at else now,
                    now,
                ),
            )
            self._conn.commit()

    def clear(
        self,
        namespace: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        *,
        older_than: float | None = None,
    ) -> int:
        query = "DELETE FROM agent_inboxes WHERE 1=1"
        params: List[Any] = []
        if namespace is not None:
            query += " AND namespace=?"
            params.append(namespace)
        if session_id is not None:
            query += " AND session_id=?"
            params.append(session_id)
        if run_id is not None:
            query += " AND run_id=?"
            params.append(run_id)
        if older_than is not None:
            query += " AND updated_at < ?"
            params.append(time.time() - older_than)

        with self._lock:
            deleted = self._conn.execute(query, tuple(params)).rowcount
            self._conn.commit()
        return deleted or 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()


@dataclass(frozen=True)
class AgentControlMessage:
    """Runtime control signal delivered through an AgentInbox."""

    command: str
    reason: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    control_id: str = field(default_factory=lambda: uuid4().hex[:8])
    created_at: str = field(default_factory=_utc_now)

    def to_notification(self) -> AgentNotification:
        metadata = deepcopy(self.metadata)
        if self.reason:
            metadata["reason"] = self.reason
        return AgentNotification(
            notification_id=self.control_id,
            source="control",
            status=self.command,
            hint=self.reason,
            metadata=metadata,
            dedupe_key=f"control:{self.command}",
            created_at=self.created_at,
        )


class ToolNotificationHandle:
    """Controlled notification publisher injected into tools."""

    def __init__(
        self,
        agent_inbox: AgentInbox | None,
        *,
        source: str = "tool_status",
        ref: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ):
        self._agent_inbox = agent_inbox
        self._source = source
        self._ref = ref
        self._metadata = deepcopy(dict(metadata or {}))

    # --- Notification Publishing ---

    def publish(
        self,
        *,
        status: str,
        hint: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        dedupe_key: str | None = None,
        source: str | None = None,
        ref: str | None = None,
    ) -> AgentNotification | None:
        if self._agent_inbox is None:
            return None

        payload = deepcopy(self._metadata)
        if metadata:
            payload.update(dict(metadata))

        return self._agent_inbox.publish(
            AgentNotification(
                notification_id=uuid4().hex[:8],
                source=source or self._source,
                ref=self._ref if ref is None else ref,
                status=status,
                hint=hint,
                metadata=payload,
                dedupe_key=dedupe_key,
                created_at=_utc_now(),
            )
        )

    def notify(
        self,
        *,
        status: str,
        hint: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        dedupe_key: str | None = None,
        source: str | None = None,
        ref: str | None = None,
    ) -> AgentNotification | None:
        return self.publish(
            status=status,
            hint=hint,
            metadata=metadata,
            dedupe_key=dedupe_key,
            source=source,
            ref=ref,
        )

    def update(
        self,
        status: str,
        *,
        hint: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        dedupe_key: str | None = None,
        source: str | None = None,
        ref: str | None = None,
    ) -> AgentNotification | None:
        return self.publish(
            status=status,
            hint=hint,
            metadata=metadata,
            dedupe_key=dedupe_key,
            source=source,
            ref=ref,
        )


class AgentInbox:
    """Small inbox for notifications delivered to an agent runtime."""

    def __init__(
        self,
        *,
        verbose: bool = False,
        owner: str | None = None,
        store: AgentInboxStore | None = None,
        namespace: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
    ):
        self._lock = RLock()
        self._notifications: List[AgentNotification] = []
        self.verbose = verbose
        self.owner = owner
        self.store = store
        self.namespace = namespace or owner or "default"
        self.session_id = session_id or "default"
        self.run_id = run_id or "default"

    # --- Scope Binding ---

    def bind(
        self,
        *,
        namespace: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> AgentInbox:
        with self._lock:
            if namespace:
                self.namespace = namespace
            if session_id:
                self.session_id = session_id
            if run_id:
                self.run_id = run_id
        return self

    def bind_scope(
        self,
        scope: ExecutionScope,
        *,
        namespace: str | None = None,
    ) -> AgentInbox:
        if not isinstance(scope, ExecutionScope):
            raise TypeError(f"`scope` must be an ExecutionScope, given `{type(scope)}`")
        return self.bind(
            namespace=namespace or scope.namespace,
            session_id=scope.session_id,
            run_id=scope.run_id or "default",
        )

    def fork(
        self,
        *,
        owner: str | None = None,
        namespace: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> AgentInbox:
        return AgentInbox(
            verbose=self.verbose,
            owner=owner or self.owner,
            store=self.store,
            namespace=namespace or self.namespace,
            session_id=session_id or self.session_id,
            run_id=run_id or self.run_id,
        )

    # --- Configuration ---

    def set_verbose(self, verbose: bool) -> None:  # noqa: FBT001
        self.verbose = bool(verbose)

    # --- Queue Operations ---

    def publish(
        self,
        notification: AgentNotification | AgentControlMessage | Mapping[str, Any],
    ) -> AgentNotification:
        normalized = self._normalize(notification)
        with self._lock:
            notifications = self._load_notifications_locked()
            if normalized.dedupe_key:
                for index, existing in enumerate(notifications):
                    if existing.dedupe_key == normalized.dedupe_key:
                        notifications[index] = normalized
                        self._save_notifications_locked(notifications)
                        if self.verbose:
                            self._print_verbose_event(
                                "notification_replace",
                                self._render_notification_payload(normalized),
                                suffix=(
                                    f"\ndedupe_key: {normalized.dedupe_key}"
                                    if normalized.dedupe_key
                                    else ""
                                ),
                            )
                        return deepcopy(normalized)
            notifications.append(normalized)
            self._save_notifications_locked(notifications)
        if self.verbose:
            self._print_verbose_event(
                "notification_publish",
                self._render_notification_payload(normalized),
            )
        return deepcopy(normalized)

    def control(
        self,
        command: str,
        *,
        reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AgentNotification:
        return self.publish(
            AgentControlMessage(
                command=command,
                reason=reason,
                metadata=deepcopy(dict(metadata or {})),
            ).to_notification()
        )

    def stop(self, *, reason: str | None = None) -> AgentNotification:
        return self.control("stop", reason=reason)

    def cancel(self, *, reason: str | None = None) -> AgentNotification:
        return self.control("cancel", reason=reason)

    def pause(self, *, reason: str | None = None) -> AgentNotification:
        return self.control("pause", reason=reason)

    def user_message(
        self,
        content: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> AgentNotification:
        return self.publish(
            {
                "source": "incoming_user_message",
                "hint": content,
                "metadata": deepcopy(dict(metadata or {})),
            }
        )

    def publish_many(
        self,
        notifications: Iterable[
            AgentNotification | AgentControlMessage | Mapping[str, Any]
        ],
    ) -> List[AgentNotification]:
        published: List[AgentNotification] = []
        for notification in notifications:
            published.append(self.publish(notification))
        return published

    def peek(self) -> List[AgentNotification]:
        with self._lock:
            return deepcopy(self._load_notifications_locked())

    def drain(self) -> List[AgentNotification]:
        with self._lock:
            notifications = deepcopy(self._load_notifications_locked())
            self._save_notifications_locked([])
        if self.verbose and notifications:
            rendered_messages = self.render_messages(notifications)
            self._print_verbose_event(
                "notification_drain",
                "\n\n".join(message["content"] for message in rendered_messages),
                prefix=f"{len(notifications)} notification(s)\n",
            )
        return notifications

    def ack(self, notification_ids: Iterable[str]) -> None:
        ids = set(notification_ids)
        if not ids:
            return
        with self._lock:
            notifications = [
                notification
                for notification in self._load_notifications_locked()
                if notification.notification_id not in ids
            ]
            self._save_notifications_locked(notifications)

    # --- Rendering ---

    def render(
        self,
        notifications: Iterable[AgentNotification | Mapping[str, Any]],
    ) -> Dict[str, str] | List[Dict[str, str]] | None:
        rendered_messages = self.render_messages(notifications)
        if not rendered_messages:
            return None
        if len(rendered_messages) == 1:
            return rendered_messages[0]
        return rendered_messages

    def render_messages(  # noqa: C901
        self,
        notifications: Iterable[AgentNotification | Mapping[str, Any]],
    ) -> List[Dict[str, str]]:
        normalized = [self._normalize(notification) for notification in notifications]
        if not normalized:
            return []

        incoming_messages = [
            notification
            for notification in normalized
            if notification.source == "incoming_user_message"
        ]
        system_notifications = [
            notification
            for notification in normalized
            if notification.source != "incoming_user_message"
        ]

        rendered_messages: List[Dict[str, str]] = []
        lines: List[str] = []
        lines.append("<system_note>")
        for notification in system_notifications:
            body_lines: List[str] = []
            body_lines.append(f"source: {self._escape_text(notification.source)}")
            if notification.ref:
                body_lines.append(f"ref: {self._escape_text(notification.ref)}")
            if notification.status:
                body_lines.append(f"status: {self._escape_text(notification.status)}")
            for key, value in sorted(notification.metadata.items()):
                rendered_value = self._escape_text(self._stringify(value))
                body_lines.append(
                    f"{self._escape_text(key)}: {rendered_value}"
                )
            if notification.hint:
                body_lines.append(f"hint: {self._escape_text(notification.hint)}")

            lines.append("<notification>")
            lines.extend(body_lines)
            lines.append("</notification>")
        lines.append("</system_note>")
        if system_notifications:
            rendered_messages.append({"role": "system", "content": "\n".join(lines)})

        lines = []
        for notification in incoming_messages:
            lines.append("<incoming_user_message>")
            if notification.hint:
                lines.append(self._escape_text(notification.hint))
            lines.append("</incoming_user_message>")

        if lines:
            rendered_messages.append({"role": "user", "content": "\n".join(lines)})
        return rendered_messages

    # --- Normalization Helpers ---

    def _normalize(
        self,
        notification: AgentNotification | AgentControlMessage | Mapping[str, Any],
    ) -> AgentNotification:
        if isinstance(notification, AgentNotification):
            return deepcopy(notification)
        if isinstance(notification, AgentControlMessage):
            return notification.to_notification()

        payload = dict(notification)
        source = payload.get("source")
        if not isinstance(source, str) or not source:
            raise ValueError("`AgentNotification.source` must be a non-empty string.")

        notification_id = payload.get("notification_id")
        if not isinstance(notification_id, str) or not notification_id:
            notification_id = uuid4().hex[:8]

        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise TypeError(
                "`AgentNotification.metadata` must be a mapping, "
                f"given `{type(metadata)}`"
            )

        created_at = payload.get("created_at")
        if not isinstance(created_at, str) or not created_at:
            created_at = _utc_now()

        return AgentNotification(
            notification_id=notification_id,
            source=source,
            ref=payload.get("ref"),
            status=payload.get("status"),
            hint=payload.get("hint"),
            metadata=deepcopy(dict(metadata)),
            dedupe_key=payload.get("dedupe_key"),
            created_at=created_at,
        )

    def _load_notifications_locked(self) -> List[AgentNotification]:
        if self.store is None:
            return deepcopy(self._notifications)
        return [
            self._normalize(notification)
            for notification in self.store.load_notifications(
                self.namespace,
                self.session_id,
                self.run_id,
            )
        ]

    def _save_notifications_locked(
        self,
        notifications: Iterable[AgentNotification],
    ) -> None:
        normalized = [self._normalize(notification) for notification in notifications]
        if self.store is None:
            self._notifications = deepcopy(normalized)
            return
        self.store.save_notifications(
            self.namespace,
            self.session_id,
            self.run_id,
            [notification.to_dict() for notification in normalized],
        )

    # --- Escaping Helpers ---

    @staticmethod
    def _stringify(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    # --- Verbose Helpers ---

    def _print_verbose_event(
        self,
        label: str,
        text: str,
        *,
        prefix: str = "",
        suffix: str = "",
    ) -> None:
        cprint(
            f"[{self.owner or 'unknown'}][{label}]\n{prefix}{text}{suffix}",
            bc="b",
            ls="b",
        )

    def _render_notification_payload(self, notification: AgentNotification) -> str:
        rendered = self.render([notification])
        if rendered is None:
            return ""
        if isinstance(rendered, list):
            rendered = rendered[0]
        content = rendered["content"]
        lines = content.splitlines()
        if len(lines) >= 4:
            return "\n".join(lines[2:-2])
        return content

    @staticmethod
    def _escape_text(value: Any) -> str:
        return escape(str(value), {'"': "&quot;"})

    @staticmethod
    def _escape_attr(value: Any) -> str:
        return escape(str(value), {'"': "&quot;"})
