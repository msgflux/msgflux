"""Synchronous policy bridges preserve control exceptions during preparation."""

import asyncio

import pytest

from msgflux.exceptions import (
    AbortRequestedError,
    TaskInterruptRequestedError,
    TaskPauseRequestedError,
)
from msgflux.nn import ToolLibrary
from msgflux.nn.modules.tool_runtime import ToolPolicy


def _control_errors():
    return [
        AbortRequestedError("abort"),
        TaskPauseRequestedError("task", "pause"),
        TaskInterruptRequestedError("task", "interrupt"),
    ]


@pytest.mark.parametrize("phase", ["before_tool", "before_dispatch"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "control",
    _control_errors(),
    ids=["abort", "pause", "interrupt"],
)
def test_prepare_policy_controls_propagate_without_running_tool(
    phase, asynchronous, control
):
    effects = []

    def effect() -> str:
        effects.append("ran")
        return "unexpected"

    class ControlPolicy(ToolPolicy):
        def __init__(self):
            super().__init__(f"control_{phase}")

        async def before_tool(self, payload):
            if phase == "before_tool":
                raise control
            return payload

        async def before_dispatch(self, payload):
            if phase == "before_dispatch":
                raise control
            return payload

    library = ToolLibrary("control-prepare", [effect])
    library.runtime_extensions.register(ControlPolicy())
    try:
        with pytest.raises(type(control)) as raised:
            if asynchronous:
                asyncio.run(library.arun("effect", {}))
            else:
                library.run("effect", {})
        assert raised.value is control
        assert effects == []
    finally:
        library.runtime_extensions.remove(f"control_{phase}")
