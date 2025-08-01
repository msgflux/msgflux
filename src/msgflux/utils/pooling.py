from typing import Literal

import numpy as np


def apply_pooling(
    embeddings: np.ndarray, strategy: Literal["mean", "max", "cls"]
) -> np.ndarray:
    """Applies different pooling strategies to embeddings.

    Args:
        embeddings:
            Array of embeddings with shape (sequence_length, embedding_dim)
            or (batch_size, sequence_length, embedding_dim)
        strategy:
            Pooling strategy to use ("mean", "max", or "cls")

    Returns:
        Embeddings after pooling with shape (embedding_dim)
        or (batch_size, embedding_dim).

    Raises:
        ValueError:
            If the pooling strategy is not recognized or if the
            input dimensions are not supported.
    """
    if embeddings.ndim == 1:
        embeddings = embeddings[np.newaxis, :]

    batch_dim = 3

    is_batch = embeddings.ndim == batch_dim
    if not is_batch:
        embeddings = embeddings[np.newaxis, :, :]

    if strategy == "mean":
        pooled = np.mean(embeddings, axis=1)
    elif strategy == "max":
        pooled = np.max(embeddings, axis=1)
    elif strategy == "cls":
        pooled = embeddings[:, 0, :]
    else:
        raise ValueError(
            f"Unrecognized pooling strategy `{strategy}`."
            "Use `mean`, `max` or `cls` instead."
        )

    if not is_batch:
        pooled = pooled.squeeze(0)

    return pooled
