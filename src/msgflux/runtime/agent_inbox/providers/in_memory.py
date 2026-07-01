from __future__ import annotations

import time
from copy import deepcopy
from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping

from msgflux.data.stores.registry import register_store
from msgflux.runtime.agent_inbox.base import AgentInboxStore


@register_store()
class InMemoryAgentInboxStore(AgentInboxStore):
    """In-memory inbox store for tests and local prototyping."""

    provider = "in_memory"

    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
        self._lock = RLock()

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
            thread[run_id] = {
                "notifications": deepcopy([dict(n) for n in notifications]),
                "created_at": created_at,
                "updated_at": time.time(),
            }

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
                threads = (
                    [thread_id] if thread_id is not None else list(ns_data.keys())
                )
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
