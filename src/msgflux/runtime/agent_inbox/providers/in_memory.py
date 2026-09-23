from __future__ import annotations

import time
from copy import deepcopy
from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping
from uuid import uuid4

from msgflux.data.stores.registry import register_store
from msgflux.runtime.agent_inbox.base import AgentInboxStore


@register_store()
class InMemoryAgentInboxStore(AgentInboxStore):
    """In-memory inbox store for tests and local prototyping."""

    provider = "in_memory"

    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
        self._lock = RLock()
        self._routing_id = uuid4().hex

    @property
    def routing_id(self) -> str:
        return f"memory:{self._routing_id}"

    def _get_run(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
    ) -> Dict[str, Any] | None:
        return self._data.get(namespace, {}).get(thread_id, {}).get(run_id)

    def load_notifications(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
    ) -> List[Mapping[str, Any]]:
        with self._lock:
            run = self._get_run(namespace, thread_id, run_id)
            if run is None:
                return []
            return deepcopy(run["notifications"])

    def save_notifications(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        notifications: Iterable[Mapping[str, Any]],
    ) -> None:
        with self._lock:
            ns = self._data.setdefault(namespace, {})
            thread = ns.setdefault(thread_id, {})
            existing = thread.get(run_id)
            created_at = existing["created_at"] if existing else time.time()
            old_claims = existing.get("claims", {}) if existing else {}
            payloads = deepcopy([dict(n) for n in notifications])
            valid_ids = {item.get("notification_id") for item in payloads}
            thread[run_id] = {
                "notifications": payloads,
                "created_at": created_at,
                "updated_at": time.time(),
                "claims": {
                    key: value for key, value in old_claims.items() if key in valid_ids
                },
            }

    def publish_notification(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        notification: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        with self._lock:
            ns = self._data.setdefault(namespace, {})
            thread = ns.setdefault(thread_id, {})
            run = thread.setdefault(
                run_id,
                {
                    "notifications": [],
                    "created_at": time.time(),
                    "updated_at": time.time(),
                    "claims": {},
                },
            )
            dedupe_key = notification.get("dedupe_key")
            if dedupe_key:
                for index, item in enumerate(run["notifications"]):
                    if item.get("dedupe_key") == dedupe_key:
                        run["notifications"][index] = deepcopy(dict(notification))
                        run["updated_at"] = time.time()
                        return deepcopy(dict(notification))
            run["notifications"].append(deepcopy(dict(notification)))
            run["updated_at"] = time.time()
            return deepcopy(dict(notification))

    def claim_notifications(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        *,
        lease_id: str,
        lease_seconds: float,
        limit: int | None = None,
    ) -> List[Mapping[str, Any]]:
        now = time.time()
        with self._lock:
            run = self._get_run(namespace, thread_id, run_id)
            if run is None:
                return []
            claims = run.setdefault("claims", {})
            selected = []
            for notification in run["notifications"]:
                notification_id = notification.get("notification_id")
                claim = claims.get(notification_id)
                if claim is not None and claim["expires_at"] > now:
                    continue
                claims[notification_id] = {
                    "lease_id": lease_id,
                    "expires_at": now + max(0.001, lease_seconds),
                }
                selected.append(deepcopy({**notification, "_lease_id": lease_id}))
                if limit is not None and len(selected) >= limit:
                    break
            return selected

    def ack_notifications(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        notification_ids: Iterable[str],
        *,
        lease_id: str | None = None,
    ) -> None:
        ids = set(notification_ids)
        if not ids:
            return
        with self._lock:
            run = self._get_run(namespace, thread_id, run_id)
            if run is None:
                return
            claims = run.setdefault("claims", {})
            kept = []
            for notification in run["notifications"]:
                notification_id = notification.get("notification_id")
                claim = claims.get(notification_id)
                owned = (
                    lease_id is None or claim is None or claim["lease_id"] == lease_id
                )
                if notification_id in ids and owned:
                    claims.pop(notification_id, None)
                    continue
                kept.append(notification)
            run["notifications"] = kept
            run["updated_at"] = time.time()

    def release_notifications(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        *,
        lease_id: str,
        notification_ids: Iterable[str] | None = None,
    ) -> None:
        ids = set(notification_ids) if notification_ids is not None else None
        with self._lock:
            run = self._get_run(namespace, thread_id, run_id)
            if run is None:
                return
            claims = run.setdefault("claims", {})
            for notification_id, claim in list(claims.items()):
                if claim.get("lease_id") == lease_id and (
                    ids is None or notification_id in ids
                ):
                    claims.pop(notification_id, None)

    def move_notifications(
        self,
        previous: tuple[str, str, str],
        current: tuple[str, str, str],
    ) -> None:
        if previous == current:
            return
        with self._lock:
            source = self._get_run(*previous)
            if source is None or not source["notifications"]:
                return
            ns = self._data.setdefault(current[0], {})
            thread = ns.setdefault(current[1], {})
            target = thread.setdefault(
                current[2],
                {
                    "notifications": [],
                    "created_at": time.time(),
                    "updated_at": time.time(),
                    "claims": {},
                },
            )
            existing_ids = {
                item.get("notification_id") for item in target["notifications"]
            }
            target["notifications"].extend(
                deepcopy(
                    [
                        item
                        for item in source["notifications"]
                        if item.get("notification_id") not in existing_ids
                    ]
                )
            )
            target["updated_at"] = time.time()
            del source["notifications"][:]
            source["claims"].clear()

    def clear(
        self,
        namespace: str | None = None,
        thread_id: str | None = None,
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
                threads = [thread_id] if thread_id is not None else list(ns_data.keys())
                for sid in threads:
                    thread = ns_data.get(sid)
                    if thread is None:
                        continue
                    run_ids = [run_id] if run_id is not None else list(thread.keys())
                    for rid in run_ids:
                        run = thread.get(rid)
                        if run is None:
                            continue
                        if cutoff is not None and run["updated_at"] >= cutoff:
                            continue
                        del thread[rid]
                        removed += 1
                    if not thread:
                        del ns_data[sid]
                if not ns_data:
                    del self._data[ns]
        return removed
