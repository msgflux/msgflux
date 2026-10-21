"""ModelGateway must not route control exceptions to another deployment."""

import pytest

from msgflux.exceptions import (
    AbortRequestedError,
    TaskInterruptRequestedError,
    TaskPauseRequestedError,
)
from msgflux.models.base import BaseModel
from msgflux.models.gateway import ModelGateway


class _ControlModel(BaseModel):
    model_type = "chat_completion"
    provider = "control-test"

    def __init__(self, model_id, *, error=None, failing_operation=None):
        self.model_id = model_id
        self.error = error
        self.failing_operation = failing_operation
        self.calls = []

    def _initialize(self):
        pass

    def _record(self, operation):
        self.calls.append(operation)
        if operation == self.failing_operation:
            raise self.error
        return f"{self.model_id}:{operation}"

    def __call__(self, **_kwargs):
        return self._record("generate")

    async def acall(self, **_kwargs):
        return self._record("agenerate")

    def count_context_tokens(self, **_kwargs):
        return self._record("count_context_tokens")

    async def acompact_context(self, **_kwargs):
        return self._record("acompact_context")

    def serialize(self):
        return {"model_id": self.model_id}


def _call(gateway, operation):
    if operation == "generate":
        return gateway()
    if operation == "agenerate":
        return gateway.acall()
    if operation == "count_context_tokens":
        return gateway.count_context_tokens(messages=[])
    if operation == "acompact_context":
        return gateway.acompact_context(messages=[])
    raise AssertionError(operation)


@pytest.mark.parametrize(
    "error",
    [
        AbortRequestedError("stop"),
        TaskInterruptRequestedError("task-1", "stop"),
        TaskPauseRequestedError("task-1", "pause"),
    ],
)
@pytest.mark.parametrize(
    "operation",
    ["generate", "agenerate", "count_context_tokens", "acompact_context"],
)
def test_control_exception_does_not_invoke_fallback(error, operation):
    primary = _ControlModel("primary", error=error, failing_operation=operation)
    fallback = _ControlModel("fallback")
    gateway = ModelGateway(
        models=[
            {"model_name": "primary", "model": primary},
            {"model_name": "fallback", "model": fallback},
        ]
    )

    if operation in {"agenerate", "acompact_context"}:

        async def run():
            with pytest.raises(type(error)) as raised:
                await _call(gateway, operation)
            assert raised.value is error

        import asyncio

        asyncio.run(run())
    else:
        with pytest.raises(type(error)) as raised:
            _call(gateway, operation)
        assert raised.value is error

    assert primary.calls == [operation]
    assert fallback.calls == []
