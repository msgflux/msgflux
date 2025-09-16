from abc import abstractmethod
from typing import List, Mapping, Optional, Union

import msgspec

from msgflux._private.client import BaseClient
from msgflux.data.dbs.response import DBResponse
from msgflux.nn import functional as F
from msgflux.utils.convert import convert_str_to_hash


class BaseDB(BaseClient):
    msgflux_type = "db"
    to_ignore = ["client", "documents"]

    def instance_type(self) -> Mapping[str, str]:
        return {"db_type": self.db_type}

    @abstractmethod
    def _initialize(self):
        """Initialize the class. This method must be implemented by subclasses.

        This method is called during the deserialization process to ensure that the client
        is properly initialized after its state has been restored.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError

class BaseKV:
    def _single_search(self, query: str) -> str:
        if self.hash_key:
            query = convert_str_to_hash(query)
        v = self.client.get(query)
        if v is not None:
            v = msgspec.msgpack.decode(v)
        return v

    def _search(self, queries: Union[str, List[str]]):
        if not isinstance(queries, list):
            queries = [queries]
        args_list = [(query,) for query in queries]
        results = F.map_gather(
            self._single_search, args_list=args_list,
        )
        return results

    def __call__(self, queries: Union[str, List[str]]) -> DBResponse:
        """
        Executes a search in the key-value database for the given query or queries.

        This method searches the underlying key-value store using the provided query string
        or list of query strings. If a single query is provided, the search result is returned
        directly. If multiple queries are provided, a list of results is returned.

        Args:
            queries:
                A single query string or a list of query strings to search for in the db.

        Returns:
            An object containing the search results.

        Raises:
            ValueError:
                If `queries` is empty or contains unsupported types.
        """
        results = self._search(queries)
        if len(results) == 1:
            results = results[0]
        response = DBResponse()
        response.set_response_type("key_search")
        response.add(results)
        return response


class BaseVector:
    def __call__(
        self,
        queries: Union[List[List[float]], "np.ndarray"],
        *,
        top_k: Optional[int] = 4,
        threshold: Optional[float] = None,
        return_score: Optional[bool] = False,
    ) -> DBResponse:
        """
        Executes a vector similarity search against the vector database.

        This method searches for the closest vectors to the given query vector(s)
        in the underlying vector store. If a single vector is provided, it is
        automatically wrapped into a list for processing.

        Args:
            queries:
                A single query vector or a list/array of query vectors, where each
                vector is represented as a list of floats or a NumPy array.
            top_k:
                The maximum number of closest matches to return for each query.
            threshold:
                A similarity score threshold; results with scores below this value
                will be filtered out.
            return_score:
                If True, include similarity scores in the returned results.

        Returns:
            An object containing the search results.

        Raises:
            ValueError: If the query vectors are not in a supported format or are empty.
        """
        results = self._search(queries, top_k, threshold, return_score)
        response = DBResponse()
        response.set_response_type("vector_search")
        response.add(results)
        return response
