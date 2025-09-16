import os
import pickle
import platform
from typing import Dict, List, Optional, Union

try:
    import faiss
    import numpy as np
except ImportError:
    faiss = None
    np = None

from msgflux.data.dbs.base import BaseDB, BaseVector
from msgflux.data.dbs.registry import register_db
from msgflux.data.dbs.types import VectorDB

if platform.system() == "Windows":
    # Solve OpenMP duplicate versions in Windows
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@register_db
class FAISSVectorDB(BaseVector, BaseDB, VectorDB):
    """FAISS Vector DB."""

    provider = "faiss"

    def __init__(
        self,
        dimension: int,
        metric_type: Optional[str] = "cosine",
        index_type: Optional[str] = "flat",
    ):
        """Args:
        dimension:
            Dimensionality of the embeddings.
        metric_type:
            Distance metric for similarity search ('cosine', 'l2', 'ip').
        index_type:
            Type of FAISS index ('flat', 'ivf', etc.).
        """
        if faiss is None:
            raise ImportError(
                "`faiss` client is not available. "
                "Install with `pip install faiss-cpu` "
                "or `pip install faiss-gpu`"
            )

        valid_metrics = {
            "cosine": faiss.METRIC_INNER_PRODUCT,
            "l2": faiss.METRIC_L2,
            "ip": faiss.METRIC_INNER_PRODUCT,
        }
        if metric_type not in valid_metrics:
            raise ValueError(
                f"Invalid metric type. Choose from {list(valid_metrics.keys())}"
            )

        self.dimension = dimension
        self.index_type = index_type
        self.metric_type = metric_type
        self._initialize()

    def _initialize(self):
        if self.index_type == "flat":
            if self.metric_type == "cosine":
                # For cosine similarity, use IndexFlatIP with normalized vectors
                self.index = faiss.IndexFlatIP(self.dimension)
            elif self.metric_type == "l2":
                self.index = faiss.IndexFlatL2(self.dimension)
            else:  # inner product
                self.index = faiss.IndexFlatIP(self.dimension)
        else:
            raise ValueError("Currently only 'flat' index type is supported")
        self.documents = []
        # TODO: support to restore data

    def add(
        self,
        embeddings: Union[List[List[float]], "np.ndarray"],
        documents: List[Dict],
    ):
        """Add embeddings and their corresponding metadata to the vector db.

        Args:
            embeddings:
                List or NumPy array of embeddings (N x dimension).
            documents:
                List of metadata.
        """
        # Convert to NumPy array if needed
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=np.float32)

        # Ensure correct dtype and dimensionality
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension must be {self.dimension}")

        # Normalize vectors if using cosine similarity
        if self.metric_type == "cosine":
            # Create a copy to avoid modifying the original array
            embeddings = embeddings.copy()

            # Use numpy for normalization to avoid FAISS normalization issues
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Avoid division by zero
            norms[norms == 0] = 1
            embeddings /= norms

        if len(documents) != embeddings.shape[0]:
            raise ValueError("Number of documents must match number of embeddings")

        self.index.add(embeddings)
        self.documents.extend(documents)

    def _search(
        self,
        queries: Union[List[List[float]], "np.ndarray"],
        top_k: Optional[int] = 4,
        threshold: Optional[float] = None,
        return_score: Optional[bool] = False,
    ) -> List[List[Dict]]:
        if len(np.array(queries).shape) == 1:
            queries = [queries]

        if not isinstance(queries, np.ndarray):
            queries = np.array(queries, dtype=np.float32)

        if queries.dtype != np.float32:
            queries = queries.astype(np.float32)

        if queries.shape[1] != self.dimension:
            raise ValueError(f"Query embedding dimension must be {self.dimension}")

        if self.metric_type == "cosine":
            queries = queries.copy()
            norms = np.linalg.norm(queries, axis=1, keepdims=True)
            norms[norms == 0] = 1
            queries /= norms

        all_results = []

        for query_embedding in queries:
            query_vec = query_embedding.reshape(1, -1)
            distances, indices = self.index.search(query_vec, top_k)

            query_results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:
                    continue

                if threshold is not None and dist < threshold:
                    continue

                result = {"data": self.documents[idx]}
                if return_score:
                    result["score"] = float(dist)
                query_results.append(result)

            all_results.append(query_results)

        return all_results

    def save(self, path: str):
        """Save the vector store to disk.

        Args:
            path:
                Path where the index and metadata will be saved.
        """
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        # Save FAISS index
        index_path = os.path.join(path, "faiss_index.bin")
        faiss.write_index(self.index, index_path)

        # Save metadata and configuration
        metadata = {
            "dimension": self.dimension,
            "metric_type": self.metric_type,
            "documents": self.documents,
        }
        metadata_path = os.path.join(path, "metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

    @classmethod
    def load(cls, path: str) -> "FAISSVectorDB":
        """Load a vector store from disk.

        Args:
            path:
                Path where the index and metadata are saved

        Returns:
            FAISSVectorDB instance
        """
        # Load metadata
        metadata_path = os.path.join(directory, "metadata.pkl")
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        # Create instance
        instance = cls(
            dimension=metadata["dimension"], metric_type=metadata["metric_type"]
        )

        # Load FAISS index
        index_path = os.path.join(directory, "faiss_index.bin")
        instance.index = faiss.read_index(index_path)

        # Restore metadata
        instance.documents = metadata["documents"]

        return instance
