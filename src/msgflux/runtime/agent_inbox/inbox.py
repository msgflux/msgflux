from __future__ import annotations

import json
import re
from copy import deepcopy
from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping, Tuple
from uuid import uuid4
from xml.sax.saxutils import escape, quoteattr

from msgflux.runtime.agent_inbox.base import AgentInboxStore
from msgflux.runtime.agent_inbox.dataclasses import (
    AgentControlMessage,
    AgentNotification,
)
from msgflux.runtime.context import (
    DEFAULT_NAMESPACE,
    ExecutionScope,
    new_run_id,
    new_thread_id,
)
from msgflux.utils.console import cprint
from msgflux.utils.time import utc_now_isoformat

_NOTIFICATION_ATTRIBUTE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]*$")
_RESERVED_NOTIFICATION_ATTRIBUTES = {"source", "ref", "status", "metadata"}


class AgentInbox:
    """Small inbox for notifications delivered to an agent runtime."""

    def __init__(
        self,
        *,
        verbose: bool = False,
        owner: str | None = None,
        store: AgentInboxStore | None = None,
        namespace: str | None = None,
        thread_id: str | None = None,
        run_id: str | None = None,
    ):
        if store is None:
            raise ValueError(
                "`store` is required when creating AgentInbox directly. "
                "Pass an AgentInboxStore with `store=...`, or let Agent create "
                "its default memory-backed inbox."
            )
        self._lock = RLock()
        self._claims: Dict[str, str] = {}
        self._receipts: set[str] = set()
        self._scope_bound = thread_id is not None or run_id is not None
        self.verbose = verbose
        self.owner = owner
        self.store = store
        self.namespace = namespace or owner or DEFAULT_NAMESPACE
        self.thread_id = thread_id or new_thread_id()
        self.run_id = run_id or new_run_id()
        self._scope_key_by_thread: Dict[
            Tuple[str, str],
            Tuple[str, str, str],
        ] = {self._thread_key(): self._scope_key()}

    # --- Scope Binding ---

    def bind(
        self,
        *,
        namespace: str | None = None,
        thread_id: str | None = None,
        run_id: str | None = None,
    ) -> AgentInbox:
        with self._lock:
            previous_key = self._scope_key()
            was_bound = self._scope_bound
            if namespace:
                self.namespace = namespace
            if thread_id:
                self.thread_id = thread_id
            if run_id:
                self.run_id = run_id
            self._scope_bound = True
            current_key = self._scope_key()
            if not was_bound or previous_key[:2] == current_key[:2]:
                self._move_notifications(previous_key, current_key)
            else:
                previous_thread_key = self._scope_key_by_thread.get(self._thread_key())
                if previous_thread_key is not None:
                    self._move_notifications(previous_thread_key, current_key)
            self._scope_key_by_thread[self._thread_key()] = current_key
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
            thread_id=scope.thread_id,
            run_id=scope.run_id or new_run_id(),
        )

    def fork(
        self,
        *,
        owner: str | None = None,
        namespace: str | None = None,
        thread_id: str | None = None,
        run_id: str | None = None,
    ) -> AgentInbox:
        return AgentInbox(
            verbose=self.verbose,
            owner=owner or self.owner,
            store=self.store,
            namespace=namespace or self.namespace,
            thread_id=thread_id or self.thread_id,
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
            stored = self.store.publish_notification(
                self.namespace, self.thread_id, self.run_id, normalized.to_dict()
            )
            normalized = self._normalize(stored)
        if self.verbose:
            label = (
                "notification_replace"
                if normalized.dedupe_key
                else "notification_publish"
            )
            self._print_verbose_event(
                label,
                self._render_notification_payload(normalized),
                suffix=(
                    f"\ndedupe_key: {normalized.dedupe_key}"
                    if label == "notification_replace"
                    else ""
                ),
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

    def interrupt(self, *, reason: str | None = None) -> AgentNotification:
        return self.control("interrupt", reason=reason)

    def pause(self, *, reason: str | None = None) -> AgentNotification:
        return self.control("pause", reason=reason)

    def user_message(
        self,
        content: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> AgentNotification:
        payload = deepcopy(dict(metadata or {}))
        payload["content"] = content
        return self.publish(
            {
                "source": "incoming_user_message",
                "metadata": payload,
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

    def claim(
        self,
        *,
        lease_seconds: float = 60.0,
        limit: int | None = None,
    ) -> List[AgentNotification]:
        lease_id = uuid4().hex
        with self._lock:
            raw = self.store.claim_notifications(
                self.namespace,
                self.thread_id,
                self.run_id,
                lease_id=lease_id,
                lease_seconds=lease_seconds,
                limit=limit,
            )
            notifications = [self._normalize(item) for item in raw]
            for notification in notifications:
                self._claims[notification.notification_id] = lease_id
            return deepcopy(notifications)

    def ack(
        self,
        notification_ids: Iterable[str],
        *,
        lease_id: str | None = None,
    ) -> None:
        ids = set(notification_ids)
        if not ids:
            return
        with self._lock:
            grouped: Dict[str | None, set[str]] = {}
            for notification_id in ids:
                claim = lease_id or self._claims.get(notification_id)
                grouped.setdefault(claim, set()).add(notification_id)
            for claim, claimed_ids in grouped.items():
                self.store.ack_notifications(
                    self.namespace,
                    self.thread_id,
                    self.run_id,
                    claimed_ids,
                    lease_id=claim,
                )
                for notification_id in claimed_ids:
                    self._claims.pop(notification_id, None)
                    self._receipts.discard(notification_id)

    def claimed_ids(self) -> set[str]:
        with self._lock:
            return set(self._claims)

    def mark_delivered(self, notification_ids: Iterable[str]) -> None:
        with self._lock:
            self._receipts.update(
                notification_id
                for notification_id in notification_ids
                if notification_id in self._claims
            )

    def delivered_ids(self) -> set[str]:
        with self._lock:
            return set(self._receipts)

    def release(
        self,
        *,
        lease_id: str | None = None,
        except_ids: Iterable[str] = (),
    ) -> None:
        preserved = set(except_ids)
        with self._lock:
            selected = {
                notification_id: current_lease
                for notification_id, current_lease in self._claims.items()
                if notification_id not in preserved
                and (lease_id is None or current_lease == lease_id)
            }
            by_lease: Dict[str, set[str]] = {}
            for notification_id, current_lease in selected.items():
                by_lease.setdefault(current_lease, set()).add(notification_id)
            for current_lease, notification_ids in by_lease.items():
                self.store.release_notifications(
                    self.namespace,
                    self.thread_id,
                    self.run_id,
                    lease_id=current_lease,
                    notification_ids=notification_ids,
                )
            self._claims = {
                notification_id: current_lease
                for notification_id, current_lease in self._claims.items()
                if notification_id in preserved
                or (lease_id is not None and current_lease != lease_id)
            }
            self._receipts.intersection_update(self._claims)

    def drain(self) -> List[AgentNotification]:
        notifications = self.claim()
        self.ack(notification.notification_id for notification in notifications)
        if self.verbose and notifications:
            rendered_messages = self.render_messages(notifications)
            self._print_verbose_event(
                "notification_drain",
                "\n\n".join(message["content"] for message in rendered_messages),
                prefix=f"{len(notifications)} notification(s)\n",
            )
        return notifications

    def clear_user_messages(self) -> int:
        """Remove pending incoming user messages while preserving runtime signals."""
        with self._lock:
            notifications = self._load_notifications_locked()
            kept = [
                notification
                for notification in notifications
                if notification.source != "incoming_user_message"
            ]
            removed = len(notifications) - len(kept)
            if removed:
                self._save_notifications_locked(kept)
            return removed

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

    def render_messages(
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
        if system_notifications:
            rendered_messages.append(
                {
                    "role": "system",
                    "content": self._render_system_notifications(system_notifications),
                }
            )

        incoming_content = self._render_incoming_user_messages(incoming_messages)
        if incoming_content is not None:
            rendered_messages.append({"role": "user", "content": incoming_content})
        return rendered_messages

    def _render_system_notifications(
        self,
        notifications: List[AgentNotification],
    ) -> str:
        return "\n".join(
            self._render_notification(notification) for notification in notifications
        )

    def _render_incoming_user_messages(
        self,
        incoming_messages: List[AgentNotification],
    ) -> str | None:
        lines: List[str] = []
        for notification in incoming_messages:
            lines.append("<incoming_user_message>")
            content = notification.metadata.get("content")
            if content:
                lines.append(self._escape_text(content))
            lines.append("</incoming_user_message>")
        if not lines:
            return None
        return "\n".join(lines)

    def _render_notification(self, notification: AgentNotification) -> str:
        attributes: List[Tuple[str, Any]] = [("source", notification.source)]
        if notification.ref:
            attributes.append(("ref", notification.ref))
        if notification.status:
            attributes.append(("status", notification.status))
        nested_metadata: Dict[str, Any] = {}
        for key, value in sorted(
            notification.metadata.items(),
            key=lambda item: str(item[0]),
        ):
            if (
                isinstance(key, str)
                and _NOTIFICATION_ATTRIBUTE_PATTERN.fullmatch(key)
                and key not in _RESERVED_NOTIFICATION_ATTRIBUTES
            ):
                attributes.append((key, value))
            else:
                nested_metadata[str(key)] = value
        if nested_metadata:
            attributes.append(("metadata", nested_metadata))
        rendered = " ".join(
            f"{key}={quoteattr(self._stringify(value))}" for key, value in attributes
        )
        return f"<notification {rendered}/>"

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
            created_at = utc_now_isoformat()

        return AgentNotification(
            notification_id=notification_id,
            source=source,
            ref=payload.get("ref"),
            status=payload.get("status"),
            metadata=deepcopy(dict(metadata)),
            dedupe_key=payload.get("dedupe_key"),
            created_at=created_at,
        )

    def _load_notifications_locked(self) -> List[AgentNotification]:
        return [
            self._normalize(notification)
            for notification in self.store.load_notifications(
                self.namespace,
                self.thread_id,
                self.run_id,
            )
        ]

    def _save_notifications_locked(
        self,
        notifications: Iterable[AgentNotification],
    ) -> None:
        normalized = [self._normalize(notification) for notification in notifications]
        self._scope_key_by_thread[self._thread_key()] = self._scope_key()
        self.store.save_notifications(
            self.namespace,
            self.thread_id,
            self.run_id,
            [notification.to_dict() for notification in normalized],
        )

    def _scope_key(self) -> Tuple[str, str, str]:
        return (self.namespace, self.thread_id, self.run_id)

    def _thread_key(self) -> Tuple[str, str]:
        return (self.namespace, self.thread_id)

    def _move_notifications(
        self,
        previous_key: Tuple[str, str, str],
        current_key: Tuple[str, str, str],
    ) -> None:
        if previous_key == current_key:
            return
        self.store.move_notifications(previous_key, current_key)

    # --- Escaping Helpers ---

    @staticmethod
    def _stringify(value: Any) -> str:
        if isinstance(value, (Mapping, list, tuple)):
            return json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
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
        return self._render_notification(notification)

    @staticmethod
    def _escape_text(value: Any) -> str:
        return escape(str(value), {'"': "&quot;"})
