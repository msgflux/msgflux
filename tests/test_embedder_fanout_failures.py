"""Embedder should preserve failures returned by its fanout helpers."""

import asyncio

import pytest

from msgflux.exceptions import (
    AbortRequestedError,
    TaskInterruptRequestedError,
    TaskPauseRequestedError,
)
from msgflux.models.base import BaseModel
from msgflux.models.response import ModelResponse
from msgflux.nn.modules.embedder import Embedder


class FailingNonBatchEmbedderModel(BaseModel):
    model_type = "text_embedder"
    provider = "test"
    batch_support = False

    def __init__(self, error):
        self.model_id = "failing-non-batch-embedder"
        self._api_key = None
        self.model = None
        self.processor = None
        self.client = None
        self.error = error
        self.completed = []

    def _initialize(self):
        pass

    def __call__(self, *, data, **kwargs):
        return self._respond(data)

    async def acall(self, *, data, **kwargs):
        return self._respond(data)

    def _respond(self, data):
        if data == "bad":
            raise self.error
        self.completed.append(data)
        response = ModelResponse()
        response.set_response_type("embeddings")
        response.add([[1.0, 2.0]])
        return response


@pytest.mark.parametrize("api", ["sync", "async"])
@pytest.mark.parametrize(
    "error",
    [
        ValueError("embedding failed"),
        AbortRequestedError("abort"),
        TaskPauseRequestedError("task", "pause"),
        TaskInterruptRequestedError("task", "interrupt"),
    ],
    ids=["ordinary", "abort", "pause", "interrupt"],
)
def test_non_batch_fanout_reraises_original_model_error(api, error):
    model = FailingNonBatchEmbedderModel(error)
    embedder = Embedder(model)

    if api == "sync":
        with pytest.raises(type(error)) as raised:
            embedder(["ok", "bad"])
    else:

        async def run():
            with pytest.raises(type(error)) as raised:
                await embedder.acall(["ok", "bad"])
            return raised.value

        raised_error = asyncio.run(run())
        assert raised_error is error
        assert model.completed == ["ok"]
        return

    assert raised.value is error
    assert model.completed == ["ok"]
