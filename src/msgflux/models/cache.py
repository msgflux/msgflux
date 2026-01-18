"""Cache system for model responses using Python's builtin lru_cache."""

import hashlib
import json
from typing import Any, Dict, Optional

import msgspec


def generate_cache_key(**kwargs) -> str:
    """Generate a deterministic cache key from keyword arguments.

    Args:
        **kwargs: Keyword arguments

    Returns:
        A SHA256 hash string as cache key
    """
    # Filter out non-cacheable items
    cacheable_kwargs = {}
    for key, value in kwargs.items():
        # Skip stream_response and other non-deterministic params
        if key in ["stream_response"]:
            continue

        # Handle typed_parser and generation_schema specially
        if key == "generation_schema" and value is not None:
            # Include schema name/type for caching
            cacheable_kwargs["generation_schema_type"] = (
                value.__name__ if hasattr(value, "__name__") else str(type(value))
            )
            continue
        elif key == "typed_parser":
            cacheable_kwargs[key] = value
            continue

        # Convert msgspec Structs to dicts
        if hasattr(value, "__msgspec_fields__"):
            cacheable_kwargs[key] = msgspec.structs.asdict(value)
        elif isinstance(value, (list, dict, str, int, float, bool, type(None))):
            cacheable_kwargs[key] = value
        else:
            # For complex objects, use their string representation
            cacheable_kwargs[key] = str(value)

    # Create a deterministic string representation
    try:
        cache_data = json.dumps(cacheable_kwargs, sort_keys=True)
    except (TypeError, ValueError):
        # Fallback to str representation
        cache_data = str(sorted(cacheable_kwargs.items()))

    # Generate hash
    cache_key = hashlib.sha256(cache_data.encode()).hexdigest()
    return cache_key


class ResponseCache:
    """Simple LRU cache for model responses using OrderedDict.

    This cache is designed to be used as an instance attribute,
    allowing each model instance to have its own cache.
    """

    def __init__(self, maxsize: Optional[int] = 128):
        """Initialize response cache.

        Args:
            maxsize: Maximum number of cached responses
        """
        self.maxsize = maxsize
        self._cache = {}
        self.hits = 0
        self.misses = 0

    def get(self, cache_key: str) -> tuple[bool, Any]:
        """Get value from cache.

        Args:
            cache_key: Cache key to lookup

        Returns:
            Tuple of (hit: bool, value: Any).
            If hit is False, value is None.
        """
        if cache_key in self._cache:
            self.hits += 1
            return (True, self._cache[cache_key])
        else:
            self.misses += 1
            return (False, None)

    def set(self, cache_key: str, value: Any):
        """Set value in cache.

        Args:
            cache_key: Cache key
            value: Value to store
        """
        # If cache is full, remove oldest entry (simple FIFO for now)
        if len(self._cache) >= self.maxsize:
            # Remove first inserted key (FIFO)
            first_key = next(iter(self._cache))
            del self._cache[first_key]

        self._cache[cache_key] = value

    def cache_info(self) -> Dict[str, int]:
        """Get cache statistics.

        Returns:
            Dict with hits, misses, maxsize, and currsize
        """
        return {
            "hits": self.hits,
            "misses": self.misses,
            "maxsize": self.maxsize,
            "currsize": len(self._cache),
        }

    def cache_clear(self):
        """Clear the cache."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0
