from __future__ import annotations

from copy import deepcopy
from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping, Tuple
from uuid import uuid4
from xml.sax.saxutils import escape

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
        lines: List[str] = ["<notifications>"]
        for notification in notifications:
            lines.append(self._render_notification(notification))
        lines.append("</notifications>")
        return "\n".join(lines)

    def _render_incoming_user_messages(
        self,
        incoming_messages: List[AgentNotification],
    ) -> str | None:
        lines: List[str] = []
        for notification in incoming_messages:
            lines.append("<incoming_user_message>")
            if notification.hint:
                lines.append(self._escape_text(notification.hint))
            lines.append("</incoming_user_message>")
        if not lines:
            return None
        return "\n".join(lines)

    def _render_notification(self, notification: AgentNotification) -> str:
        fields = [f"source={self._escape_text(notification.source)}"]
        if notification.ref:
            fields.append(f"ref={self._escape_text(notification.ref)}")
        if notification.status:
            fields.append(f"status={self._escape_text(notification.status)}")

        for key, value in sorted(notification.metadata.items()):
            fields.append(
                f"{self._escape_text(key)}="
                f"{self._escape_text(self._stringify(value))}"
            )
        rendered = " ".join(fields)
        if notification.hint:
            rendered += f" | {self._escape_text(notification.hint)}"
        return rendered

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
            hint=payload.get("hint"),
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
        notifications = [
            self._normalize(notification)
            for notification in self.store.load_notifications(*previous_key)
        ]
        if not notifications:
            return
        current = [
            self._normalize(notification)
            for notification in self.store.load_notifications(*current_key)
        ]
        self.store.save_notifications(
            *current_key,
            [notification.to_dict() for notification in notifications + current],
        )
        self.store.clear(*previous_key)

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
        if len(lines) >= 3 and lines[0] == "<notifications>":
            return "\n".join(lines[1:-1])
        return content

    @staticmethod
    def _escape_text(value: Any) -> str:
        return escape(str(value), {'"': "&quot;"})
