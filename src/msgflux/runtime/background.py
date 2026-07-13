from __future__ import annotations

from concurrent.futures import CancelledError as FutureCancelledError
from functools import partial
from threading import Lock
from typing import Any, Dict, Mapping
from uuid import uuid4

from msgflux._private.executor import Executor
from msgflux.exceptions import (
    TaskIdCollisionError,
    TaskInterruptRequestedError,
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
from msgflux.tasks import TaskActivityRecorder, TaskHandle
from msgflux.tools.builtin.task_tool import (
    build_background_dispatch_result,
    truncate_activity_text,
)
from msgflux.tools.helpers import build_call_parameters_for_response
from msgflux.tools.responses import ToolCall
from msgflux.tools.types import ToolBackground


class BackgroundTaskDispatcher:
    """Dispatches tools into the task runtime and tracks live executions."""

    def __init__(self, library_handle: Any):
        self.library_handle = library_handle
        self._task_futures: Dict[str, Any] = {}
        self._task_futures_lock = Lock()
        self._task_inboxes: Dict[str, AgentInbox] = {}
        self._task_inboxes_lock = Lock()
        self._task_checkpoint_stores: Dict[str, Any] = {}
        self._task_checkpoint_stores_lock = Lock()

    def clear(self) -> None:
        with self._task_futures_lock:
            self._task_futures.clear()
        with self._task_inboxes_lock:
            self._task_inboxes.clear()
        with self._task_checkpoint_stores_lock:
            self._task_checkpoint_stores.clear()

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

    def register_task_inbox(self, task_id: str, inbox: AgentInbox) -> None:
        with self._task_inboxes_lock:
            self._task_inboxes[task_id] = inbox

    def get_task_inbox(self, task_id: str) -> AgentInbox | None:
        with self._task_inboxes_lock:
            return self._task_inboxes.get(task_id)

    def register_task_checkpoint_store(
        self,
        task_id: str,
        checkpoint_store: Any,
    ) -> None:
        if checkpoint_store is None:
            return
        with self._task_checkpoint_stores_lock:
            self._task_checkpoint_stores[task_id] = checkpoint_store

    def get_task_checkpoint_store(self, task_id: str) -> Any | None:
        with self._task_checkpoint_stores_lock:
            return self._task_checkpoint_stores.get(task_id)

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
    ) -> Any:
        scope = execution_scope or {}
        with execution_context(**scope):
            task_handle.set_running()
            try:
                result = tool(**call_params)
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
            task_handle.complete(result)
            self.publish_task_notification(
                task_id=task_handle.task_id,
                tool_name=tool_name,
                status="completed",
                agent_inbox=agent_inbox,
            )
            return result

    def resume_agent_task(
        self,
        *,
        task: Any,
        message: str,
    ) -> str:
        task_store = self.library_handle.get_task_store()
        tool_name = task.tool_name
        tool = self.library_handle.get_tool(tool_name)
        checkpoint_namespace = task.metadata.get("checkpoint_namespace")
        if checkpoint_namespace is None:
            checkpoint_namespace = (
                tool.impl.get_module_name()
                if hasattr(tool, "impl") and hasattr(tool.impl, "get_module_name")
                else tool.get_module_name()
            )

        checkpoint_store = get_execution_context().get(
            "checkpoint_store"
        ) or self.get_task_checkpoint_store(task.task_id)
        thread_id = task.metadata.get("checkpoint_thread_id")
        if not isinstance(thread_id, str) or not thread_id:
            thread_id = new_thread_id()
        run_id = task.metadata.get("checkpoint_run_id") or task.task_id
        if task.status in {"completed", "interrupted"}:
            run_id = new_run_id()
            updated_task = task_store.update_metadata(
                task.task_id,
                {"checkpoint_run_id": run_id},
            )
            if updated_task is not None:
                task = updated_task

        root_inbox = get_execution_context().get("agent_inbox")
        if root_inbox is None:
            root_inbox = self.library_handle.get_agent_inbox()
        task_inbox = self.get_task_inbox(task.task_id)
        if task_inbox is None:
            task_inbox = root_inbox.fork(
                owner=f"{tool_name}:{task.task_id}",
                namespace=checkpoint_namespace,
                thread_id=(
                    thread_id if isinstance(thread_id, str) and thread_id else None
                ),
                run_id=run_id,
            )
            self.register_task_inbox(task.task_id, task_inbox)

        task_store.requeue(task.task_id)
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

        execution_scope = {
            "thread_id": thread_id
            if isinstance(thread_id, str) and thread_id
            else None,
            "run_id": run_id,
            "parent_run_id": task.metadata.get("parent_run_id"),
            "root_run_id": task.metadata.get("root_run_id"),
            "checkpoint_store": checkpoint_store,
            "agent_inbox": task_inbox,
            "task_activity_recorder": TaskActivityRecorder(task.task_id, task_store),
        }
        future = Executor.get_instance().submit(
            partial(
                self.run_tool,
                tool=tool,
                task_handle=TaskHandle(
                    task.task_id,
                    task_store,
                    tool_name=tool_name,
                    agent_inbox=root_inbox,
                ),
                tool_name=tool_name,
                call_params={
                    **dict(task.metadata.get("task_resume_params") or {}),
                    "message": message,
                    "scope": ExecutionScope(
                        thread_id=thread_id,
                        namespace=checkpoint_namespace,
                        run_id=run_id,
                        parent_run_id=task.metadata.get("parent_run_id"),
                        root_run_id=task.metadata.get("root_run_id"),
                    ),
                },
                execution_scope=execution_scope,
                agent_inbox=root_inbox,
            )
        )
        self.register_task_future(task.task_id, future)
        future.add_done_callback(partial(self.cleanup_task_future, task.task_id))
        future.add_done_callback(self.log_task_failure)
        return "Message scheduled and background agent resumed."

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
        tool_id: str,
        tool_name: str,
        call_params: Dict[str, Any],
        config: Mapping[str, Any],
    ) -> Any:
        task_kind = config.get("tool_kind", "tool")
        task_capabilities = ToolBackground.get_background_capabilities(tool, config)
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
        checkpoint_store = context.get("checkpoint_store")
        root_agent_inbox = context.get("agent_inbox")
        if root_agent_inbox is None:
            root_agent_inbox = self.library_handle.get_agent_inbox()
        task_metadata = {
            "tool_call_id": tool_id,
            "task_kind": "agent" if is_agent_task else task_kind,
            "checkpoint_namespace": checkpoint_namespace if is_agent_task else None,
            "task_resume_params": task_resume_params,
            "thread_id": thread_id,
            "namespace": context.get("namespace"),
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

        task_inbox = None
        if is_agent_task:
            task_inbox = root_agent_inbox.fork(
                owner=f"{checkpoint_namespace}:{task.task_id}",
                namespace=checkpoint_namespace,
                thread_id=thread_id if isinstance(thread_id, str) else None,
                run_id=task.task_id,
            )
            self.register_task_inbox(task.task_id, task_inbox)
            self.register_task_checkpoint_store(task.task_id, checkpoint_store)
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
            "task_handle": TaskHandle(
                task.task_id,
                task_store,
                tool_name=tool_name,
                agent_inbox=root_agent_inbox,
            ),
            "task_activity_recorder": TaskActivityRecorder(task.task_id, task_store),
        }
        future = Executor.get_instance().submit(
            partial(
                self.run_tool,
                tool=tool,
                task_handle=TaskHandle(
                    task.task_id,
                    task_store,
                    tool_name=tool_name,
                    agent_inbox=root_agent_inbox,
                ),
                tool_name=tool_name,
                call_params=runner_params,
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
            parameters=build_call_parameters_for_response(call_params),
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
        hint: str | None = None,
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
                hint=hint,
                metadata={"tool": tool_name},
                dedupe_key=f"task:{task_id}",
            )
        )
