"""Tests for Embedder module with batch support."""

import pytest

from msgflux.core.message import Message
from msgflux.models.response import ModelResponse
from msgflux.nn.modules import Embedder


class MockBatchEmbedder:
    """Mock embedder that supports batch processing."""

    batch_support = True
    model_type = "text_embedder"

    def __init__(self):
        self.call_count = 0
        self.last_data = None

    def __call__(self, data=None, **kwargs):
        self.call_count += 1
        self.last_data = data

        # Simulate batch embedding: return list of embeddings
        response = ModelResponse()
        response.set_response_type("text_embedding")

        if isinstance(data, list):
            embeddings = [[0.1 * i, 0.2 * i, 0.3 * i] for i in range(len(data))]
        else:
            embeddings = [[0.1, 0.2, 0.3]]

        response.add(embeddings)
        return response

    async def acall(self, data=None, **kwargs):
        return self.__call__(data=data, **kwargs)


class MockNonBatchEmbedder:
    """Mock embedder that does NOT support batch processing."""

    batch_support = False
    model_type = "text_embedder"

    def __init__(self):
        self.call_count = 0
        self.all_calls = []

    def __call__(self, data=None, **kwargs):
        self.call_count += 1
        self.all_calls.append(data)

        # Simulate single embedding
        response = ModelResponse()
        response.set_response_type("text_embedding")

        # Return an embedding derived from the input so tests can verify order.
        text = data if isinstance(data, str) else data[0]
        suffix = int(text[-1]) if isinstance(text, str) and text[-1].isdigit() else 0
        embedding = [[0.5 + suffix, 0.6 + suffix, 0.7 + suffix]]
        response.add(embedding)
        return response

    async def acall(self, data=None, **kwargs):
        return self.__call__(data=data, **kwargs)


def test_embedder_with_batch_model_single_text():
    """Test Embedder with batch-supporting model on single text."""
    model = MockBatchEmbedder()
    embedder = Embedder(model=model)

    # Single text input
    result = embedder("Hello world")

    # Should call model once
    assert model.call_count == 1

    # Should pass as list to batch model
    assert model.last_data == ["Hello world"]

    # Should return single embedding (not list of lists)
    assert isinstance(result, list)
    assert len(result) == 3
    assert result == [0.0, 0.0, 0.0]


def test_embedder_with_batch_model_multiple_texts():
    """Test Embedder with batch-supporting model on multiple texts."""
    model = MockBatchEmbedder()
    embedder = Embedder(model=model)

    # Multiple texts
    texts = ["text1", "text2", "text3"]
    result = embedder(texts)

    # Should call model once (batch mode)
    assert model.call_count == 1

    # Should pass all texts at once
    assert model.last_data == texts

    # Should return list of embeddings
    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(emb, list) for emb in result)


def test_embedder_with_non_batch_model_multiple_texts():
    """Test Embedder with non-batch model on multiple texts."""
    model = MockNonBatchEmbedder()
    embedder = Embedder(model=model)

    # Multiple texts
    texts = ["text1", "text2", "text3"]
    result = embedder(texts)

    # Should call model 3 times (once per text via F.map_gather)
    assert model.call_count == 3

    # Parallel execution should process every text exactly once.
    assert len(model.all_calls) == 3
    assert sorted(model.all_calls) == sorted(texts)

    # Results should preserve the input order.
    assert isinstance(result, list)
    assert len(result) == 3
    assert [embedding[0] for embedding in result] == [
        [1.5, 1.6, 1.7],
        [2.5, 2.6, 2.7],
        [3.5, 3.6, 3.7],
    ]


def test_embedder_with_non_batch_model_single_text():
    """Test Embedder with non-batch model on single text."""
    model = MockNonBatchEmbedder()
    embedder = Embedder(model=model)

    # Single text
    result = embedder("Hello world")

    # Should call model once
    assert model.call_count == 1

    # Should pass as single item (batch check short-circuits for single items)
    assert model.all_calls[0] == ["Hello world"]

    # Should return single embedding
    assert isinstance(result, list)
    assert len(result) == 3


def test_embedder_with_message_object():
    """Test Embedder with Message object using message_fields."""
    model = MockBatchEmbedder()
    embedder = Embedder(model=model, message_fields={"task": "texts"})

    # Create message with texts field
    msg = Message(texts=["text1", "text2"])
    result = embedder(msg)

    # Should extract texts from message
    assert model.call_count == 1
    assert model.last_data == ["text1", "text2"]

    # Should return embeddings
    assert isinstance(result, list)
    assert len(result) == 2


def test_embedder_with_config():
    """Test Embedder with config parameters."""
    model = MockBatchEmbedder()
    embedder = Embedder(model=model, config={"normalize": True, "truncate": True})

    result = embedder("Hello")

    # Should work with config
    assert model.call_count == 1
    assert isinstance(result, list)


def test_embedder_async():
    """Test Embedder async execution."""
    import asyncio

    model = MockBatchEmbedder()
    embedder = Embedder(model=model)

    async def run_test():
        result = await embedder.aforward(["text1", "text2"])
        return result

    result = asyncio.run(run_test())

    # Should call model once
    assert model.call_count == 1

    # Should return embeddings
    assert isinstance(result, list)
    assert len(result) == 2


def test_embedder_response_mode_plain():
    """Test Embedder with default response_mode (None)."""
    model = MockBatchEmbedder()
    embedder = Embedder(model=model, response_mode=None)

    result = embedder("Hello")

    # Should return plain embeddings (not Message)
    assert isinstance(result, list)
    assert not isinstance(result, Message)


def test_embedder_invalid_model_type():
    """Test Embedder with invalid model type."""

    class InvalidModel:
        model_type = "chat_completion"  # Wrong type

    with pytest.raises(TypeError, match="requires be `embedder` model"):
        Embedder(model=InvalidModel())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
