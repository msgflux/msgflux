"""Control exceptions must bypass the built-in retry policy."""

import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent


def test_default_tool_retry_does_not_repeat_action_after_interrupt():
    code = dedent(
        """
        import asyncio
        import json

        from msgflux.exceptions import TaskInterruptRequestedError
        from msgflux.nn.modules.tool import ToolLibrary
        from msgflux.runtime.context import execution_context
        from msgflux.tasks import InMemoryTaskStore, TaskHandle
        from msgflux.tools.config import tool_config
        from msgflux.tools.types import Hidden
        from msgflux.utils.tenacity import apply_retry, default_model_retry

        async def main():
            effects = []
            effects_after_interrupt = []
            started = asyncio.Event()
            release = asyncio.Event()
            store = InMemoryTaskStore()
            store.create("worker", task_id="retry-stop")
            task_handle = TaskHandle("retry-stop", store)

            @tool_config(runtime_inputs=["handle"])
            async def external_action(handle: Hidden) -> str:
                effects.append("action")
                task = store.get(handle.get_task_id())
                if task.metadata.get("interrupt_requested"):
                    effects_after_interrupt.append("action")
                if len(effects) == 1:
                    started.set()
                    await release.wait()
                handle.raise_if_interrupted()
                return "done"

            attempts = []

            @tool_config()
            async def retry_ordinary_failure() -> str:
                attempts.append("attempt")
                if len(attempts) < 3:
                    raise RuntimeError("temporary failure")
                return "ok"

            cancelled_attempts = []

            @tool_config()
            async def cancelled_action() -> str:
                cancelled_attempts.append("attempt")
                raise asyncio.CancelledError("caller cancelled")

            library = ToolLibrary(
                "control-retry-probe",
                tools=[external_action, retry_ordinary_failure, cancelled_action],
                task_store=store,
            )
            ordinary_result = await library.arun("retry_ordinary_failure", {})
            try:
                await library.arun("cancelled_action", {})
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("caller cancellation was swallowed")

            model_attempts = []
            async def model_control_error():
                model_attempts.append("attempt")
                raise TaskInterruptRequestedError("model-task", "stop")

            retrying_model_call = apply_retry(
                model_control_error, default=default_model_retry
            )
            try:
                await retrying_model_call()
            except TaskInterruptRequestedError as error:
                model_task_id = error.task_id
            else:
                raise AssertionError("model interruption was swallowed")
            with execution_context(task_store=store, task_handle=task_handle):
                run = asyncio.create_task(library.arun("external_action", {}))
                await asyncio.wait_for(started.wait(), timeout=2)
                store.request_interrupt(task_handle.task_id)
                release.set()
                try:
                    await asyncio.wait_for(run, timeout=2)
                except TaskInterruptRequestedError as error:
                    task_id = error.task_id
                    control_type = type(error).__name__
                else:
                    raise AssertionError("task interruption was swallowed")

            print(json.dumps({
                "ordinary_result": ordinary_result,
                "ordinary_attempts": len(attempts),
                "cancelled_attempts": len(cancelled_attempts),
                "model_attempts": len(model_attempts),
                "model_task_id": model_task_id,
                "interrupt_type": control_type,
                "interrupt_task_id": task_id,
                "effects": len(effects),
                "effects_after_interrupt": len(effects_after_interrupt),
            }))

        asyncio.run(main())
        """
    )
    env = os.environ.copy()
    env["msgflux_TOOL_STOP_AFTER_DELAY"] = "10"
    env["msgflux_MODEL_STOP_AFTER_DELAY"] = "10"
    result = subprocess.run(  # noqa: S603 - fixed interpreter and test script
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        check=True,
        text=True,
        timeout=10,
    )
    outcome = json.loads(result.stdout)

    assert outcome == {
        "ordinary_result": "ok",
        "ordinary_attempts": 3,
        "cancelled_attempts": 1,
        "model_attempts": 1,
        "model_task_id": "model-task",
        "interrupt_type": "TaskInterruptRequestedError",
        "interrupt_task_id": "retry-stop",
        "effects": 1,
        "effects_after_interrupt": 0,
    }
