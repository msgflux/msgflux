from typing import Any, Dict, List, Optional, Union

import msgspec
from cachetools import TTLCache

from msgflux.data.databases.base import BaseDB
from msgflux.data.databases.types import KVDB
from msgflux.utils.convert import convert_str_to_hash


class CacheToolsKVDB(BaseDB, KVDB):
    """CacheTools Key-Value DB."""

    provider = "cachetools"

    def __init__(
        self,
        ttl: Optional[int] = 3600,
        maxsize: Optional[int] = 10000,
        hash_key: Optional[bool] = True,
    ):
        """Args:
        ttl:
            The time-to-live (TTL) for each cache entry in seconds.
        maxsize:
            The maximum number of items the cache can store.
        hash_key:
            Whether to hash the keys before storing them in the cache.
        """
        self.hash_key = hash_key
        self.maxsize = maxsize
        self.ttl = ttl
        self._initialize()

    def _initialize(self):
        self.client = TTLCache(maxsize=self.maxsize, ttl=self.ttl)

    def add(self, documents: Union[List[Dict[str, Any]], Dict[str, Any]]):
        if not isinstance(documents, list):
            documents = [documents]
        for document in documents:
            for k, v in document.items():
                if self.hash_key:
                    k = convert_str_to_hash(k)
                v = msgspec.msgpack.encode(v)
                self.client[k] = v

    def _search(self, queries: Union[str, List[str]]):
        if not isinstance(queries, list):
            queries = [queries]
        query_results = []
        for query in queries:
            if self.hash_key:
                query = convert_str_to_hash(query)
            v = self.client.get(query)
            if v is not None:
                v = msgspec.msgpack.decode(v)
            query_results.append(v)
        return query_results

    def __call__(self, queries: Union[str, List[str]]) -> Union[Any, List[Any]]:
        query_results = self._search(queries)
        if len(query_results) == 1:
            return query_results[0]
        return query_results
