"""Tests for model response caching system."""

import pytest

from msgflux.models.cache import ResponseCache, generate_cache_key
from msgflux.tools import ToolDefinitions


class TestCacheKeyGeneration:
    """Tests for cache key generation."""

    def test_generate_cache_key_basic(self):
        """Test basic cache key generation."""
        key1 = generate_cache_key(messages=["Hello"], model="gpt-4")
        key2 = generate_cache_key(messages=["Hello"], model="gpt-4")
        assert key1 == key2

    def test_generate_cache_key_order_independent(self):
        """Test that kwarg order doesn't affect cache key."""
        key1 = generate_cache_key(messages=["Hello"], model="gpt-4", temperature=0.7)
        key2 = generate_cache_key(temperature=0.7, model="gpt-4", messages=["Hello"])
        assert key1 == key2

    def test_generate_cache_key_different_values(self):
        """Test that different values produce different keys."""
        key1 = generate_cache_key(messages=["Hello"], model="gpt-4")
        key2 = generate_cache_key(messages=["Hi"], model="gpt-4")
        assert key1 != key2

    def test_generate_cache_key_filters_stream_response(self):
        """Test that stream_response is filtered from cache key."""
        key1 = generate_cache_key(messages=["Hello"], stream_response="dummy")
        key2 = generate_cache_key(messages=["Hello"])
        assert key1 == key2

    def test_generate_cache_key_with_lists(self):
        """Test cache key with list values."""
        key1 = generate_cache_key(messages=[{"role": "user", "content": "Hello"}])
        key2 = generate_cache_key(messages=[{"role": "user", "content": "Hello"}])
        assert key1 == key2

    def test_generate_cache_key_with_nested_dict(self):
        """Test cache key with nested dict values."""
        key1 = generate_cache_key(
            messages=[{"role": "user", "content": "Hello"}],
            tool_definitions=ToolDefinitions(
                schemas=[{"name": "get_weather", "parameters": {"type": "object"}}]
            ),
        )
        key2 = generate_cache_key(
            messages=[{"role": "user", "content": "Hello"}],
            tool_definitions=ToolDefinitions(
                schemas=[{"name": "get_weather", "parameters": {"type": "object"}}]
            ),
        )
        assert key1 == key2


class TestResponseCache:
    """Tests for ResponseCache class."""

    def test_cache_initialization(self):
        """Test cache initialization."""
        cache = ResponseCache(maxsize=100)
        assert cache.maxsize == 100
        assert cache.hits == 0
        assert cache.misses == 0

    def test_cache_set_and_get(self):
        """Test basic cache set and get operations."""
        cache = ResponseCache(maxsize=10)

        # Set a value
        cache.set("key1", "value1")

        # Get the value
        hit, value = cache.get("key1")
        assert hit is True
        assert value == "value1"
        assert cache.hits == 1
        assert cache.misses == 0

    def test_cache_miss(self):
        """Test cache miss."""
        cache = ResponseCache(maxsize=10)

        # Try to get non-existent key
        hit, value = cache.get("nonexistent")
        assert hit is False
        assert value is None
        assert cache.hits == 0
        assert cache.misses == 1

    def test_cache_multiple_values(self):
        """Test caching multiple values."""
        cache = ResponseCache(maxsize=10)

        # Set multiple values
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Get all values
        hit1, value1 = cache.get("key1")
        hit2, value2 = cache.get("key2")
        hit3, value3 = cache.get("key3")

        assert all([hit1, hit2, hit3])
        assert value1 == "value1"
        assert value2 == "value2"
        assert value3 == "value3"

    def test_cache_info(self):
        """Test cache statistics."""
        cache = ResponseCache(maxsize=5)

        # Add some values
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Access them
        cache.get("key1")
        cache.get("key2")
        cache.get("nonexistent")

        info = cache.cache_info()
        assert info["hits"] == 2
        assert info["misses"] == 1
        assert info["maxsize"] == 5

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = ResponseCache(maxsize=10)

        # Set values
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Verify they exist
        hit1, _ = cache.get("key1")
        assert hit1 is True

        # Clear cache
        cache.cache_clear()

        # Verify cache is empty and stats are reset
        info = cache.cache_info()
        assert info["currsize"] == 0
        assert info["hits"] == 0
        assert info["misses"] == 0

        # Verify key is no longer in cache (this will increment misses)
        hit2, _ = cache.get("key1")
        assert hit2 is False
        assert cache.misses == 1  # This get() call increments misses

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = ResponseCache(maxsize=3)

        # Fill cache
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Add one more (should evict least recently used)
        cache.set("key4", "value4")

        # Check cache info
        info = cache.cache_info()
        assert info["currsize"] <= 3

    def test_cache_with_complex_objects(self):
        """Test caching complex objects."""
        cache = ResponseCache(maxsize=10)

        complex_obj = {
            "data": [1, 2, 3],
            "metadata": {"key": "value"},
            "nested": {"deep": {"value": 42}},
        }

        cache.set("complex", complex_obj)
        hit, retrieved = cache.get("complex")

        assert hit is True
        assert retrieved == complex_obj

    def test_cache_overwrite(self):
        """Test overwriting cached values."""
        cache = ResponseCache(maxsize=10)

        cache.set("key1", "value1")
        cache.set("key1", "value2")  # Overwrite

        hit, value = cache.get("key1")
        assert hit is True
        assert value == "value2"


class TestCacheIntegration:
    """Integration tests for cache with models."""

    @pytest.mark.skipif(
        True,  # Skip by default as it requires OpenAI API key
        reason="Integration test requires OpenAI API key",
    )
    def test_model_cache_integration(self):
        """Test cache integration with actual model."""
        from msgflux.models import Model

        # Create model with cache
        model = Model.chat_completion(
            "openai/gpt-4o-mini",
            enable_cache=True,
            cache_size=10,
            temperature=0.0,  # Deterministic
        )

        # First call (cache miss)
        response1 = model("What is 1+1?")
        result1 = response1.consume()

        # Second identical call (cache hit)
        response2 = model("What is 1+1?")
        result2 = response2.consume()

        # Results should be identical from cache
        assert result1 == result2

        # Check cache stats
        if model._response_cache:
            stats = model._response_cache.cache_info()
            assert stats["hits"] >= 1
