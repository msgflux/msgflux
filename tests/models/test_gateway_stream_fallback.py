"""ModelGateway must fall back on streaming setup failures before output."""

import asyncio

import pytest

from msgflux.exceptions import (
    AbortRequestedError,
    ModelRouterError,
    TaskInterruptRequestedError,
    TaskPauseRequestedError,
)
from msgflux.models.base import BaseModel
from msgflux.models.gateway import ModelGateway
from msgflux.models.response import LMStreamEvent, ModelStreamResponse

OPERATIONS = ["sync", "async"]


def _stream(
    mode,
    *,
    error=None,
    response_type=None,
    text=None,
    reasoning=None,
    reasoning_summary=None,
    commentary=None,
    data=None,
    event=False,
    finish=True,
):
    stream = ModelStreamResponse(mode=mode)
    if response_type is not None:
        stream.set_response_type(response_type)
    if text is not None:
        stream.add(text)
    if reasoning is not None:
        stream.add_reasoning(reasoning)
    if reasoning_summary is not None:
        stream.add_reasoning_summary(reasoning_summary)
    if commentary is not None:
        stream.add_commentary(commentary)
    if data is not None:
        stream.data = data
    if event:
        stream._add_event(LMStreamEvent(type="tools.updated", data={}))
    if error is not None:
        stream.set_error(error)
    if finish:
        stream.finish(status="failed" if error is not None else "completed")
    return stream


class _StreamModel(BaseModel):
    model_type = "chat_completion"
    provider = "stream-test"

    def __init__(self, model_id, factory):
        self.model_id = model_id
        self.factory = factory
        self.calls = 0

    def _initialize(self):
        pass

    def _call(self, mode):
        self.calls += 1
        return self.factory(mode)

    def __call__(self, **_kwargs):
        return self._call("sync")

    async def acall(self, **_kwargs):
        return self._call("async")


def _gateway(models, *, fallback=True):
    return ModelGateway(
        models=[{"model_name": model_id, "model": model} for model_id, model in models],
        fallback=fallback,
    )


def _call(gateway, operation):
    if operation == "sync":
        return gateway(messages=[], stream=True)
    return asyncio.run(gateway.acall(messages=[], stream=True))


@pytest.mark.parametrize("operation", OPERATIONS)
def test_stream_setup_failure_falls_back(operation):
    original = RuntimeError("primary stream setup failed")
    primary = _StreamModel("primary", lambda mode: _stream(mode, error=original))
    healthy = _StreamModel(
        "healthy",
        lambda mode: _stream(
            mode, response_type="text_generation", text=f"{mode} answer"
        ),
    )
    gateway = _gateway([("primary", primary), ("healthy", healthy)])

    response = _call(gateway, operation)

    assert response.error is None
    assert response.response_type == "text_generation"
    assert response.data == f"{operation} answer"
    assert primary.calls == 1
    assert healthy.calls == 1


@pytest.mark.parametrize("operation", OPERATIONS)
def test_stream_setup_failure_respects_fallback_false(operation):
    original = RuntimeError("primary stream setup failed")
    primary = _StreamModel("primary", lambda mode: _stream(mode, error=original))
    fallback = _StreamModel("fallback", lambda mode: _stream(mode, text="unused"))
    gateway = _gateway([("primary", primary), ("fallback", fallback)], fallback=False)

    with pytest.raises(ModelRouterError) as raised:
        _call(gateway, operation)

    assert raised.value.exceptions == [original]
    assert raised.value.model_info == [("primary", "stream-test", original)]
    assert "primary (stream-test): RuntimeError: primary stream setup failed" in str(
        raised.value
    )
    assert fallback.calls == 0


@pytest.mark.parametrize("operation", OPERATIONS)
def test_stream_setup_failures_preserve_all_error_details(operation):
    errors = [RuntimeError("primary down"), ValueError("secondary down")]
    primary = _StreamModel("primary", lambda mode: _stream(mode, error=errors[0]))
    secondary = _StreamModel("secondary", lambda mode: _stream(mode, error=errors[1]))
    gateway = _gateway([("primary", primary), ("secondary", secondary)])

    with pytest.raises(ModelRouterError) as raised:
        _call(gateway, operation)

    assert raised.value.exceptions == errors
    assert raised.value.model_info == [
        ("primary", "stream-test", errors[0]),
        ("secondary", "stream-test", errors[1]),
    ]
    rendered = str(raised.value)
    assert "primary (stream-test): RuntimeError: primary down" in rendered
    assert "secondary (stream-test): ValueError: secondary down" in rendered


@pytest.mark.parametrize(
    "error",
    [
        AbortRequestedError("stop"),
        TaskInterruptRequestedError("task-1", "stop"),
        TaskPauseRequestedError("task-1", "pause"),
    ],
)
@pytest.mark.parametrize("operation", OPERATIONS)
def test_stream_control_error_does_not_fall_back(error, operation):
    primary = _StreamModel("primary", lambda mode: _stream(mode, error=error))
    fallback = _StreamModel("fallback", lambda mode: _stream(mode, text="unused"))
    gateway = _gateway([("primary", primary), ("fallback", fallback)])

    with pytest.raises(type(error)) as raised:
        _call(gateway, operation)

    assert raised.value is error
    assert primary.calls == 1
    assert fallback.calls == 0


@pytest.mark.parametrize(
    "error",
    [KeyboardInterrupt(), SystemExit(2), asyncio.CancelledError()],
)
@pytest.mark.parametrize("operation", OPERATIONS)
def test_stream_base_exception_does_not_fall_back(error, operation):
    primary = _StreamModel("primary", lambda mode: _stream(mode, error=error))
    fallback = _StreamModel("fallback", lambda mode: _stream(mode, text="unused"))
    gateway = _gateway([("primary", primary), ("fallback", fallback)])

    with pytest.raises(type(error)) as raised:
        _call(gateway, operation)

    assert raised.value is error
    assert primary.calls == 1
    assert fallback.calls == 0


@pytest.mark.parametrize(
    "activity",
    [
        pytest.param({"response_type": "text_generation"}, id="response-type"),
        pytest.param({"text": "partial"}, id="content"),
        pytest.param({"commentary": "partial"}, id="commentary"),
        pytest.param({"reasoning": "thinking"}, id="reasoning"),
        pytest.param({"reasoning_summary": "summary"}, id="reasoning-summary"),
        pytest.param({"data": object()}, id="tool-payload"),
        pytest.param({"event": True}, id="stream-event"),
    ],
)
@pytest.mark.parametrize("operation", OPERATIONS)
def test_stream_failure_after_activity_is_not_retried(activity, operation):
    original = RuntimeError("stream failed after activity")
    built = []

    def primary_factory(mode):
        stream = _stream(mode, error=original, **activity)
        built.append(stream)
        return stream

    primary = _StreamModel("primary", primary_factory)
    fallback = _StreamModel("fallback", lambda mode: _stream(mode, text="unused"))
    gateway = _gateway([("primary", primary), ("fallback", fallback)])

    response = _call(gateway, operation)

    assert response is built[0]
    assert response.error is original
    assert primary.calls == 1
    assert fallback.calls == 0


@pytest.mark.parametrize("field", ["reasoning", "reasoning_summary"])
@pytest.mark.parametrize("operation", OPERATIONS)
def test_stream_final_reasoning_fields_block_retry(field, operation):
    """Custom adapters may set final fields without calling add_reasoning."""
    original = RuntimeError("stream failed after reasoning")
    built = []

    def primary_factory(mode):
        stream = ModelStreamResponse(mode=mode)
        setattr(stream, field, "already produced")
        stream.set_error(original)
        stream.finish(status="failed")
        built.append(stream)
        return stream

    primary = _StreamModel("primary", primary_factory)
    fallback = _StreamModel("fallback", lambda mode: _stream(mode, text="unused"))
    gateway = _gateway([("primary", primary), ("fallback", fallback)])

    response = _call(gateway, operation)

    assert response is built[0]
    assert response.error is original
    assert getattr(response, field) == "already produced"
    assert not response.has_reasoning
    assert not response.has_reasoning_summary
    assert primary.calls == 1
    assert fallback.calls == 0


@pytest.mark.parametrize("operation", OPERATIONS)
def test_stream_setup_failure_leaves_finalization_to_producer(operation):
    original = RuntimeError("primary stream setup failed")
    finalized = []
    built = []

    def primary_factory(mode):
        stream = _stream(mode, error=original, finish=False)
        stream.add_finalizer(finalized.append)
        built.append(stream)
        return stream

    primary = _StreamModel("primary", primary_factory)
    healthy = _StreamModel(
        "healthy",
        lambda mode: _stream(mode, response_type="text_generation", text="ok"),
    )
    gateway = _gateway([("primary", primary), ("healthy", healthy)])

    response = _call(gateway, operation)

    assert response.error is None
    assert healthy.calls == 1
    assert built[0].error is original
    assert finalized == []
