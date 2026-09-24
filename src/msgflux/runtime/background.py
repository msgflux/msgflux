from __future__ import annotations

from concurrent.futures import CancelledError as FutureCancelledError
from contextlib import nullcontext
from functools import partial
from threading import Lock
from typing import Any, Dict, Mapping
from uuid import uuid4

from msgflux._private.executor import Executor
from msgflux.exceptions import (
    TaskIdCollisionError,
    TaskInterruptRequestedError,
    TaskLeaseLostError,
    TaskPauseRequestedError,
)
from msgflux.logger import logger
from msgflux.runtime.agent_inbox import AgentInbox, AgentNotification
from msgflux.runtime.context import (
    ExecutionScope,
    execution_context,
    get_execution_context,
    new_run_id,
    new_thread_id,
)
from msgflux.runtime.events import (
    EventType,
    _capture_events,
    _hub_event_sink,
    _is_capturing_events,
    emit_event,
    event_source,
)
from msgflux.runtime.permissions import require_permissions
from msgflux.runtime.task_leases import TaskLeaseHeartbeats
from msgflux.tasks import TaskActivityRecorder, TaskHandle
from msgflux.tools.builtin.task_tool import (
    build_background_dispatch_result,
    truncate_activity_text,
)
from msgflux.tools.handles import ToolBucketHandle
from msgflux.tools.responses import ToolCall
from msgflux.tools.types import ToolBackground, ToolBucket
from msgflux.utils.msgspec import lossless_json_roundtrip


class BackgroundTaskDispatcher:
    """Dispatches tools into the task runtime and tracks live executions."""

    def __init__(self, library_handle: Any):
        self.library_handle = library_handle
        self._task_futures: Dict[str, Any] = {}
        self._task_futures_lock = Lock()
        self.lease_seconds = 60.0

    def clear(self) -> None:
        with self._task_futures_lock:
            self._task_futures.clear()

    def register_task_future(self, task_id: str, future: Any) -> None:
        with self._task_futures_lock:
            self._task_futures[task_id] = future

    def get_task_future(self, task_id: str) -> Any | None:
        with self._task_futures_lock:
            return self._task_futures.get(task_id)

    def cleanup_task_future(self, task_id: str, future: Any) -> None:
        with self._task_futures_lock:
            current = self._task_futures.get(task_id)
            if current is future:
                self._task_futures.pop(task_id, None)

    def get_task_inbox(
        self,
        task_id: str,
        *,
        task_store: Any,
        agent_inbox: AgentInbox,
    ) -> AgentInbox | None:
        """Resolve a task's current inbox without retaining a per-task view."""
        task = task_store.get(task_id)
        if task is None or task.metadata.get("task_kind") != "agent":
            return None
        return self._resolve_task_inbox(task, agent_inbox=agent_inbox)

    @staticmethod
    def _resolve_task_inbox(task: Any, *, agent_inbox: AgentInbox) -> AgentInbox:
        metadata = task.metadata
        expected_store = metadata.get("inbox_store_id")
        if not isinstance(expected_store, str) or not expected_store:
            raise RuntimeError(f"Task `{task.task_id}` has no inbox store binding.")
        if agent_inbox.store.routing_id != expected_store:
            raise RuntimeError(
                f"Task `{task.task_id}` inbox store does not match the current "
                "runtime binding."
            )
        namespace = metadata.get("checkpoint_namespace")
        thread_id = metadata.get("checkpoint_thread_id")
        run_id = metadata.get("checkpoint_run_id")
        if not all(
            isinstance(value, str) and value for value in (namespace, thread_id, run_id)
        ):
            raise RuntimeError(f"Task `{task.task_id}` has an incomplete inbox route.")
        return agent_inbox.fork(
            owner=f"{task.tool_name}:{task.task_id}",
            namespace=namespace,
            thread_id=thread_id,
            run_id=run_id,
        )

    def _effective_checkpoint_store(
        self,
        *,
        tool: Any,
        resume_params: Mapping[str, Any],
    ) -> Any | None:
        """Match the child agent's configured-store precedence at dispatch/resume."""
        impl = getattr(tool, "impl", None)
        if isinstance(impl, ToolBucket):
            child_param = getattr(impl, "task_checkpoint_namespace_param", None)
            child_name = resume_params.get(child_param) if child_param else None
            if isinstance(child_name, str):
                child = self.library_handle.get_tool_definition(child_name).executor
                impl = getattr(child, "impl", None)
        configured = getattr(impl, "checkpoint_store", None)
        if configured is not None:
            return configured
        return get_execution_context().get("checkpoint_store")

    @staticmethod
    def _validate_checkpoint_binding(task: Any, store: Any | None) -> None:
        expected = task.metadata.get("checkpoint_store_id")
        actual = store.routing_id if store is not None else None
        if expected != actual:
            raise RuntimeError(
                f"Task `{task.task_id}` checkpoint store does not match the "
                "current runtime binding."
            )

    @staticmethod
    def _validate_recovery_checkpoint(
        task: Any,
        checkpoint_store: Any | None,
        namespace: str,
        thread_id: str,
        run_id: str,
    ) -> Mapping[str, Any] | None:
        if task.status != "running":
            raise RuntimeError(f"Task `{task.task_id}` is not running.")
        checkpoint = (
            checkpoint_store.load_state(namespace, thread_id, run_id)
            if checkpoint_store is not None
            else None
        )
        if checkpoint is None and not isinstance(
            task.metadata.get("initial_call_params"), dict
        ):
            raise RuntimeError(
                f"Task `{task.task_id}` has no checkpoint or durable initial input "
                "to recover."
            )
        return checkpoint

    @staticmethod
    def _durable_initial_params(visible_params: Mapping[str, Any]) -> dict | None:
        """Keep only inputs that can survive a JSON-backed task store."""
        original = dict(visible_params)
        lossless, decoded = lossless_json_roundtrip(original)
        return decoded if lossless and isinstance(decoded, dict) else None

    def _get_task_resume_params(
        self,
        *,
        tool: Any,
        call_params: Mapping[str, Any],
    ) -> Dict[str, Any]:
        impl = getattr(tool, "impl", None)
        param_names = getattr(impl, "task_resume_params", ())
        if not param_names:
            return {}
        return {name: call_params[name] for name in param_names if name in call_params}

    def _get_checkpoint_namespace(
        self,
        *,
        tool_name: str,
        tool: Any,
        task_resume_params: Mapping[str, Any],
    ) -> str:
        impl = getattr(tool, "impl", None)
        namespace_param = getattr(impl, "task_checkpoint_namespace_param", None)
        if isinstance(namespace_param, str):
            value = task_resume_params.get(namespace_param)
            if isinstance(value, str) and value:
                return value
        if hasattr(getattr(tool, "impl", None), "get_module_name"):
            return tool.impl.get_module_name()
        return tool_name

    def run_tool(
        self,
        *,
        tool: Any,
        task_handle: TaskHandle,
        tool_name: str,
        call_params: Dict[str, Any],
        execution_scope: Dict[str, Any] | None = None,
        agent_inbox: AgentInbox | None = None,
        required_permissions: tuple[str, ...] = (),
        required_resources: tuple = (),
        recover_expired: bool = False,
    ) -> Any:
        scope = execution_scope or {}
        capture = (
            nullcontext()
            if _is_capturing_events()
            else _capture_events(_hub_event_sink())
        )
        with (
            execution_context(**scope),
            capture,
            event_source(tool_name, "background"),
        ):
            if not task_handle.has_worker_lease:
                task_handle.start_worker(
                    lease_seconds=self.lease_seconds,
                    recover_expired=recover_expired,
                )
            elif not task_handle.renew_worker(lease_seconds=self.lease_seconds):
                TaskLeaseHeartbeats.unregister(task_handle)
                raise TaskLeaseLostError(task_handle.task_id)
            TaskLeaseHeartbeats.register(task_handle, lease_seconds=self.lease_seconds)
            try:
                require_permissions(required_permissions, required_resources)
                result = tool(**call_params)
            except TaskLeaseLostError:
                raise
            except TaskInterruptRequestedError as exc:
                task_handle.interrupt(reason=str(exc))
                self.publish_task_notification(
                    task_id=task_handle.task_id,
                    tool_name=tool_name,
                    status="interrupted",
                    agent_inbox=agent_inbox,
                )
                raise
            except TaskPauseRequestedError as exc:
                task_handle.pause(reason=str(exc))
                self.publish_task_notification(
                    task_id=task_handle.task_id,
                    tool_name=tool_name,
                    status="paused",
                    agent_inbox=agent_inbox,
                )
                raise
            except Exception as exc:
                task_handle.fail(exc)
                self.publish_task_notification(
                    task_id=task_handle.task_id,
                    tool_name=tool_name,
                    status="failed",
                    agent_inbox=agent_inbox,
                )
                raise
            else:
                task_handle.complete(result)
                self.publish_task_notification(
                    task_id=task_handle.task_id,
                    tool_name=tool_name,
                    status="completed",
                    agent_inbox=agent_inbox,
                )
                return result
            finally:
                TaskLeaseHeartbeats.unregister(task_handle)

    def resume_agent_task(  # noqa: C901 - restart and expired-worker recovery share routing
        self,
        *,
        task: Any,
        message: str,
        recover_expired: bool = False,
        reconcile_terminal: bool = False,
    ) -> str:
        task_store = self.library_handle.get_task_store()
        tool_name = task.tool_name
        tool = self.library_handle.get_tool(tool_name)
        definition = self.library_handle.get_tool_definition(tool_name)
        required_permissions = definition.required_permissions
        required_resources = definition.required_resources
        require_permissions(required_permissions, required_resources)
        checkpoint_namespace = task.metadata.get("checkpoint_namespace")
        if checkpoint_namespace is None:
            checkpoint_namespace = (
                tool.impl.get_module_name()
                if hasattr(tool, "impl") and hasattr(tool.impl, "get_module_name")
                else tool.get_module_name()
            )

        checkpoint_store = self._effective_checkpoint_store(
            tool=tool,
            resume_params=task.metadata.get("task_resume_params") or {},
        )
        self._validate_checkpoint_binding(task, checkpoint_store)
        thread_id = task.metadata.get("checkpoint_thread_id")
        if not isinstance(thread_id, str) or not thread_id:
            thread_id = new_thread_id()
        root_inbox = get_execution_context().get("agent_inbox")
        if root_inbox is None:
            root_inbox = self.library_handle.get_agent_inbox()
        task_inbox = self._resolve_task_inbox(task, agent_inbox=root_inbox)
        run_id = task.metadata.get("checkpoint_run_id") or task.task_id
        checkpoint = None
        if recover_expired:
            checkpoint = self._validate_recovery_checkpoint(
                task, checkpoint_store, checkpoint_namespace, thread_id, run_id
            )
            if reconcile_terminal:
                if checkpoint is None or checkpoint.get("status") != "completed":
                    raise RuntimeError(
                        f"Task `{task.task_id}` has no completed checkpoint "
                        "to reconcile."
                    )
                result = checkpoint.get("task_result")
                if not isinstance(result, Mapping) or "value" not in result:
                    raise RuntimeError(
                        f"Task `{task.task_id}` terminal checkpoint has no "
                        "recorded result."
                    )
            elif checkpoint is not None and checkpoint.get("status") in {
                "completed",
                "interrupted",
            }:
                raise RuntimeError(
                    f"Task `{task.task_id}` has a terminal checkpoint; "
                    "use `reconcile_agent_task` for completed runs."
                )
        elif task.status in {"queued", "running"}:
            raise RuntimeError(f"Task `{task.task_id}` is already active.")
        next_run_id = None
        if not recover_expired and task.status in {"completed", "interrupted"}:
            run_id = new_run_id()
            next_run_id = run_id
        if not recover_expired:
            updated_task = task_store.requeue(
                task.task_id,
                expected_status=task.status,
                expected_generation=task.metadata.get("resume_generation", 0),
                run_id=next_run_id,
            )
            if updated_task is None:
                raise RuntimeError(f"Task `{task.task_id}` changed during resume.")
            task = updated_task
            task_inbox = self._resolve_task_inbox(task, agent_inbox=root_inbox)
            emit_event(
                EventType.TASK_START,
                {
                    "task_id": task.task_id,
                    "tool_name": tool_name,
                    "status": "queued",
                },
            )
        activity_recorder = TaskActivityRecorder(task.task_id, task_store)
        task_handle = TaskHandle(
            task.task_id,
            task_store,
            tool_name=tool_name,
            agent_inbox=root_inbox,
        )
        if reconcile_terminal:
            task_handle.start_worker(
                lease_seconds=self.lease_seconds,
                recover_expired=True,
            )
            emit_event(
                EventType.TASK_START,
                {
                    "task_id": task.task_id,
                    "tool_name": tool_name,
                    "status": "running",
                    "reconciled": True,
                },
            )
            task_handle.complete(result["value"])
            self.publish_task_notification(
                task_id=task.task_id,
                tool_name=tool_name,
                status="completed",
                agent_inbox=root_inbox,
            )
            return "Completed background agent result reconciled from checkpoint."
        execution_scope = {
            "thread_id": thread_id
            if isinstance(thread_id, str) and thread_id
            else None,
            "run_id": run_id,
            "parent_run_id": task.metadata.get("parent_run_id"),
            "root_run_id": task.metadata.get("root_run_id"),
            "checkpoint_store": checkpoint_store,
            "agent_inbox": task_inbox,
            "task_handle": task_handle,
            "task_activity_recorder": activity_recorder,
        }
        initial_params = (
            dict(task.metadata["initial_call_params"])
            if recover_expired and checkpoint is None
            else {
                **dict(task.metadata.get("task_resume_params") or {}),
                "message": message,
            }
        )
        resume_params = {
            **initial_params,
            "scope": ExecutionScope(
                thread_id=thread_id,
                namespace=checkpoint_namespace,
                run_id=run_id,
                parent_run_id=task.metadata.get("parent_run_id"),
                root_run_id=task.metadata.get("root_run_id"),
            ),
        }
        if recover_expired and checkpoint is None:
            resume_params["tool_call_id"] = task.metadata.get("tool_call_id")
        if isinstance(getattr(tool, "impl", None), ToolBucket):
            resume_params["handle"] = self.library_handle.for_tool(
                tool_name=tool_name,
                agent_inbox=task_inbox,
                task_store=task_store,
                message=message,
                tool_call_id=task.metadata.get("tool_call_id"),
                activity_recorder=activity_recorder,
            )

        if recover_expired:
            task_handle.start_worker(
                lease_seconds=self.lease_seconds,
                recover_expired=True,
            )
            emit_event(
                EventType.TASK_START,
                {
                    "task_id": task.task_id,
                    "tool_name": tool_name,
                    "status": "running",
                    "recovered": True,
                },
            )
            TaskLeaseHeartbeats.register(task_handle, lease_seconds=self.lease_seconds)
        try:
            if recover_expired and checkpoint is None:
                task_store.enqueue_message(task.task_id, uuid4().hex, message)
            task_store.add_activity(
                task.task_id,
                kind="message",
                summary=(f"Root message: {truncate_activity_text(message)}"),
                metadata={
                    "direction": "root_to_task",
                    "resume": True,
                    "run_id": run_id,
                },
            )
            future = Executor.get_instance().submit(
                partial(
                    self.run_tool,
                    tool=tool,
                    task_handle=task_handle,
                    tool_name=tool_name,
                    call_params=resume_params,
                    required_permissions=required_permissions,
                    required_resources=required_resources,
                    execution_scope=execution_scope,
                    agent_inbox=root_inbox,
                    recover_expired=recover_expired,
                )
            )
        except BaseException:
            if recover_expired:
                TaskLeaseHeartbeats.unregister(task_handle)
                task_handle.fail("Worker submission failed")
            raise
        self.register_task_future(task.task_id, future)
        future.add_done_callback(partial(self.cleanup_task_future, task.task_id))
        if recover_expired:
            future.add_done_callback(
                partial(self._cleanup_cancelled_recovery, task_handle)
            )
        future.add_done_callback(self.log_task_failure)
        return (
            "Message scheduled and expired background agent recovered."
            if recover_expired
            else "Message scheduled and background agent resumed."
        )

    @staticmethod
    def _cleanup_cancelled_recovery(task_handle: TaskHandle, future: Any) -> None:
        if not future.cancelled():
            return
        TaskLeaseHeartbeats.unregister(task_handle)
        try:
            task_handle.fail("Recovered worker was cancelled before execution")
        except TaskLeaseLostError:
            pass

    def log_task_failure(self, future: Any) -> None:
        try:
            future.result()
        except FutureCancelledError:
            return
        except TaskInterruptRequestedError:
            return
        except TaskPauseRequestedError:
            return
        except Exception as exc:
            logger.error(f"Background task error: {exc!s}", exc_info=True)

    def dispatch(
        self,
        *,
        tool: Any,
        definition: Any,
        tool_id: str,
        tool_name: str,
        call_params: Dict[str, Any],
        visible_params: Mapping[str, Any],
    ) -> Any:
        require_permissions(
            definition.required_permissions, definition.required_resources
        )
        task_kind = definition.kind
        task_capabilities = ToolBackground.get_background_capabilities(definition)
        task_store = self.library_handle.get_task_store()
        task_resume_params = self._get_task_resume_params(
            tool=tool,
            call_params=call_params,
        )
        is_agent_task = ToolBackground.is_agent_source(tool)
        checkpoint_namespace = self._get_checkpoint_namespace(
            tool_name=tool_name,
            tool=tool,
            task_resume_params=task_resume_params,
        )
        context = get_execution_context()
        thread_id = context.get("thread_id")
        if not isinstance(thread_id, str) or not thread_id:
            thread_id = new_thread_id()
        parent_run_id = context.get("run_id")
        root_run_id = context.get("root_run_id")
        checkpoint_store = self._effective_checkpoint_store(
            tool=tool,
            resume_params=task_resume_params,
        )
        root_agent_inbox = context.get("agent_inbox")
        if root_agent_inbox is None:
            root_agent_inbox = self.library_handle.get_agent_inbox()
        task_metadata = {
            "tool_call_id": tool_id,
            "task_kind": "agent" if is_agent_task else task_kind,
            "checkpoint_namespace": checkpoint_namespace if is_agent_task else None,
            "inbox_store_id": (
                root_agent_inbox.store.routing_id if is_agent_task else None
            ),
            "checkpoint_store_id": (
                checkpoint_store.routing_id
                if is_agent_task and checkpoint_store is not None
                else None
            ),
            "task_resume_params": task_resume_params,
            "initial_call_params": (
                self._durable_initial_params(visible_params) if is_agent_task else None
            ),
            "thread_id": thread_id,
            "parent_run_id": parent_run_id,
            "root_run_id": root_run_id,
            "checkpoint_thread_id": thread_id,
            "background_capabilities": list(task_capabilities),
            "interrupt_requested": False,
        }
        while True:
            task_id = uuid4().hex[:8]
            try:
                task = task_store.create(
                    task_id=task_id,
                    tool_name=tool_name,
                    metadata={
                        **task_metadata,
                        "checkpoint_run_id": task_id if is_agent_task else None,
                    },
                )
            except TaskIdCollisionError:
                continue
            break

        emit_event(
            EventType.TASK_START,
            {
                "task_id": task.task_id,
                "tool_name": tool_name,
                "status": task.status,
            },
        )

        task_inbox = None
        if is_agent_task:
            task_inbox = root_agent_inbox.fork(
                owner=f"{checkpoint_namespace}:{task.task_id}",
                namespace=checkpoint_namespace,
                thread_id=thread_id if isinstance(thread_id, str) else None,
                run_id=task.task_id,
            )
        runner_params = dict(call_params)
        if is_agent_task:
            runner_params["scope"] = ExecutionScope(
                thread_id=thread_id,
                namespace=checkpoint_namespace,
                run_id=task.task_id,
                parent_run_id=(
                    parent_run_id
                    if isinstance(parent_run_id, str) and parent_run_id
                    else None
                ),
                root_run_id=(
                    root_run_id
                    if isinstance(root_run_id, str) and root_run_id
                    else task.task_id
                ),
            )
        runner_params["tool_call_id"] = tool_id
        activity_recorder = TaskActivityRecorder(task.task_id, task_store)
        task_handle = TaskHandle(
            task.task_id,
            task_store,
            tool_name=tool_name,
            agent_inbox=root_agent_inbox,
        )
        execution_scope = {
            "thread_id": thread_id
            if isinstance(thread_id, str) and thread_id
            else None,
            "run_id": task.task_id,
            "parent_run_id": (
                parent_run_id
                if isinstance(parent_run_id, str) and parent_run_id
                else None
            ),
            "root_run_id": (
                root_run_id if isinstance(root_run_id, str) and root_run_id else None
            ),
            "checkpoint_store": checkpoint_store,
            "agent_inbox": task_inbox or root_agent_inbox,
            "task_handle": task_handle,
            "task_activity_recorder": activity_recorder,
        }
        bucket_handle = runner_params.get("handle")
        if isinstance(bucket_handle, ToolBucketHandle):
            runner_params["handle"] = bucket_handle.with_runtime(
                agent_inbox=task_inbox or root_agent_inbox,
                task_store=task_store,
                activity_recorder=activity_recorder,
            )
        future = Executor.get_instance().submit(
            partial(
                self.run_tool,
                tool=tool,
                task_handle=task_handle,
                tool_name=tool_name,
                call_params=runner_params,
                required_permissions=definition.required_permissions,
                required_resources=definition.required_resources,
                execution_scope=execution_scope,
                agent_inbox=root_agent_inbox,
            )
        )
        self.register_task_future(task.task_id, future)
        future.add_done_callback(partial(self.cleanup_task_future, task.task_id))
        future.add_done_callback(self.log_task_failure)
        return ToolCall(
            id=tool_id,
            name=tool_name,
            parameters=dict(visible_params),
            result=build_background_dispatch_result(
                task_id=task.task_id,
                tool_name=tool_name,
                task_capabilities=task_capabilities,
            ),
        )

    def publish_task_notification(
        self,
        *,
        task_id: str,
        tool_name: str,
        status: str,
        agent_inbox: AgentInbox | None = None,
    ) -> AgentNotification | None:
        inbox = agent_inbox
        if inbox is None:
            inbox = self.library_handle.get_agent_inbox()
        if inbox is None:
            return None
        return inbox.publish(
            AgentNotification(
                notification_id=uuid4().hex[:8],
                source="task",
                ref=task_id,
                status=status,
                metadata={"tool": tool_name},
                dedupe_key=f"task:{task_id}:{status}",
            )
        )
