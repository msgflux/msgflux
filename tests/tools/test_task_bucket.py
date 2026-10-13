import pytest

import msgflux as mf
from msgflux.nn import ToolLibrary
from msgflux.tools.builtin import TaskTool


@pytest.mark.asyncio
async def test_grouped_task_controls_execute_and_follow_dynamic_sources():
    @mf.tool_config(allow_background=True)
    async def job(value: int) -> int:
        """Double the input."""
        return value * 2

    library = ToolLibrary(name="grouped", tools=[job])
    extension = library.extensions["background_tasks"]
    extension.enable_grouping()
    assert "task" in library.library
    assert isinstance(library.library["task"].impl, TaskTool)
    assert "task_status" not in library.library
    assert "task_activity" not in library.get_bucket_tool_names("task")
    dispatch = await library.aforward(
        [("bg", "job", {"value": 21, "run_in_background": True})]
    )
    dispatch_result = dispatch.tool_calls[0].result
    assert dispatch_result.startswith("task_id='")
    assert " " not in dispatch_result
    listed = await library.aforward(
        [("ls", "task", {"request": {"operation": "list", "status": None}})]
    )
    task_id = listed.tool_calls[0].result[0]["task_id"]
    result = await library.aforward(
        [
            (
                "wait",
                "task",
                {"request": {"operation": "wait", "task_id": task_id, "timeout": 2.0}},
            )
        ]
    )
    assert result.tool_calls[0].result == 42
    repeated = await library.aforward(
        [("output", "task", {"request": {"operation": "output", "task_id": task_id}})]
    )
    assert repeated.tool_calls[0].result == 42
    library.add(activity_job)
    assert "task_activity" in library.get_bucket_tool_names("task")
    library.remove("activity_job")
    assert "task_activity" not in library.get_bucket_tool_names("task")
    library.remove("job")
    assert "task" not in library.library
    library.add(job)
    assert "task" in library.library


@mf.tool_config(allow_background=True, background_capabilities=["activity"])
def activity_job() -> str:
    """Activity-capable job."""
    return "ok"


def test_task_grouping_extension_removal_clears_captured_controls():
    library = ToolLibrary(name="removal", tools=[activity_job])
    library.extensions["background_tasks"].enable_grouping()
    assert "task" in library.library
    library.remove_extension("background_tasks")
    assert "task" not in library.library
    assert list(library.library) == ["activity_job"]
    assert not any(name.startswith("task_") for name in library.get_tool_names())


def test_disabling_all_grouped_controls_removes_empty_bucket_and_resets_with_sources():
    library = ToolLibrary(name="empty", tools=[activity_job])
    library.extensions["background_tasks"].enable_grouping()
    for name in tuple(library.get_bucket_tool_names("task")):
        library.remove(name)
    assert "task" not in library.library
    library.remove("activity_job")
    library.add(activity_job)
    assert "task" in library.library


def test_grouped_controls_cover_deferred_background_sources_before_loading():
    @mf.tool_config(allow_background=True, defer_loading=True)
    def later() -> str:
        """Deferred job."""
        return "done"

    library = ToolLibrary(name="deferred", tools=[later])
    assert "task_status" not in library.library
    library.extensions["background_tasks"].enable_grouping()
    assert "task" in library.library
    assert "task_wait" in library.get_bucket_tool_names("task")
    assert "later" not in library.library
    library.remove("later")
    assert "task" not in library.library


@pytest.mark.asyncio
async def test_renamed_controller_keeps_guidance_out_of_dispatch_results():
    class Jobs(TaskTool):
        name = "jobs"

    library = ToolLibrary(name="renamed", tools=[activity_job, Jobs])
    library.apply_default_usage_guidance()
    guidance = library.get_tool_usage_guidance()
    assert any(
        item["name"] == "jobs" and "task_id" in item["guidance"] for item in guidance
    )
    dispatch = await library.aforward(
        [("bg", "activity_job", {"run_in_background": True})]
    )
    task_id = dispatch.tool_calls[0].result.split("'")[1]
    assert dispatch.tool_calls[0].result == f"task_id='{task_id}'"
    result = await library.aforward(
        [
            (
                "wait",
                "jobs",
                {"request": {"operation": "wait", "task_id": task_id, "timeout": 2.0}},
            )
        ]
    )
    assert result.tool_calls[0].result == "ok"
