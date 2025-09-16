from typing import Any, Dict, List, Optional, Union

import msgspec

try:
    from diskcache import Cache
except ImportError:
    Cache = None

from msgflux.data.dbs.base import BaseDB, BaseKV
from msgflux.data.dbs.registry import register_db
from msgflux.data.dbs.types import KVDB
from msgflux.utils.convert import convert_str_to_hash


@register_db
class DiskCacheKVDB(BaseKV, BaseDB, KVDB):
    """DiskCache Key-Value DB."""

    provider = "diskcache"

    def __init__(self, *, ttl: Optional[int] = 3600, hash_key: Optional[bool] = True):
        """Args:
        ttl:
            The time-to-live (TTL) for each cache entry in seconds.
        hash_key:
            Whether to hash the keys before storing them in the cache.
        """
        if Cache is None:
            raise ImportError(
                "`diskcache` client is not available. Install with "
                "`pip install diskcache`"
            )
        self.hash_key = hash_key
        self.ttl = ttl
        self._initialize()

    def _initialize(self):
        self.client = Cache(timeout=1)

    def add(self, documents: Union[List[Dict[str, Any]], Dict[str, Any]]):
        if not isinstance(documents, list):
            documents = [documents]
        for document in documents:
            for k, v in document.items():
                if self.hash_key:
                    k = convert_str_to_hash(k)
                v = msgspec.msgpack.encode(v)
                self.client.set(k, v, expire=self.ttl)
