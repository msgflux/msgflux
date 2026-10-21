"""Tests for msgflux.models.registry module."""

import pytest

from msgflux.models.base import BaseModel
from msgflux.models.registry import model_registry, register_model


class ValidModel(BaseModel):
    """Valid model for testing registration."""

    model_type = "test_type"
    provider = "test_provider"

    def _initialize(self):
        pass

    def __call__(self):
        return "test"


class InvalidModelNoType(BaseModel):
    """Invalid model without model_type."""

    provider = "test_provider"

    def _initialize(self):
        pass

    def __call__(self):
        return "test"


class InvalidModelNoProvider(BaseModel):
    """Invalid model without provider."""

    model_type = "test_type"

    def _initialize(self):
        pass

    def __call__(self):
        return "test"


class TestRegisterModel:
    """Test suite for register_model function."""

    def setup_method(self):
        """Clear registry before each test."""
        self._original_registry = {
            key: dict(value) for key, value in model_registry.items()
        }
        model_registry.clear()

    def teardown_method(self):
        model_registry.clear()
        model_registry.update(self._original_registry)

    def test_register_valid_model(self):
        """Test registering a valid model."""
        register_model(ValidModel)

        assert "test_type" in model_registry
        assert "test_provider" in model_registry["test_type"]
        assert model_registry["test_type"]["test_provider"] == ValidModel

    def test_register_model_returns_class(self):
        """Test that register_model returns the class (for decorator usage)."""
        result = register_model(ValidModel)
        assert result == ValidModel

    def test_register_multiple_providers_same_type(self):
        """Test registering multiple providers for the same model type."""

        class Provider1(BaseModel):
            model_type = "multi_test"
            provider = "provider1"

            def _initialize(self):
                pass

            def __call__(self):
                return "provider1"

        class Provider2(BaseModel):
            model_type = "multi_test"
            provider = "provider2"

            def _initialize(self):
                pass

            def __call__(self):
                return "provider2"

        register_model(Provider1)
        register_model(Provider2)

        assert "multi_test" in model_registry
        assert "provider1" in model_registry["multi_test"]
        assert "provider2" in model_registry["multi_test"]
        assert model_registry["multi_test"]["provider1"] == Provider1
        assert model_registry["multi_test"]["provider2"] == Provider2

    def test_register_model_without_type(self):
        """Test that registering a model without model_type raises ValueError."""
        with pytest.raises(
            ValueError,
            match="InvalidModelNoType must define `model_type` and `provider`",
        ):
            register_model(InvalidModelNoType)

    def test_register_model_without_provider(self):
        """Test that registering a model without provider raises ValueError."""
        with pytest.raises(
            ValueError,
            match="InvalidModelNoProvider must define `model_type` and `provider`",
        ):
            register_model(InvalidModelNoProvider)

    def test_register_model_overwrites_existing(self):
        """Test that re-registering a model overwrites the previous registration."""

        class OriginalModel(BaseModel):
            model_type = "overwrite_test"
            provider = "test_provider"

            def _initialize(self):
                pass

            def __call__(self):
                return "original"

        class NewModel(BaseModel):
            model_type = "overwrite_test"
            provider = "test_provider"

            def _initialize(self):
                pass

            def __call__(self):
                return "new"

        register_model(OriginalModel)
        assert model_registry["overwrite_test"]["test_provider"] == OriginalModel

        register_model(NewModel)
        assert model_registry["overwrite_test"]["test_provider"] == NewModel

    def test_register_as_decorator(self):
        """Test using register_model as a decorator."""

        @register_model
        class DecoratedModel(BaseModel):
            model_type = "decorated_type"
            provider = "decorated_provider"

            def _initialize(self):
                pass

            def __call__(self):
                return "decorated"

        assert "decorated_type" in model_registry
        assert "decorated_provider" in model_registry["decorated_type"]
        assert model_registry["decorated_type"]["decorated_provider"] == DecoratedModel

    def test_registry_structure(self):
        """Test the structure of model_registry."""
        assert isinstance(model_registry, dict)

        register_model(ValidModel)

        # Check nested dictionary structure
        assert isinstance(model_registry["test_type"], dict)
        assert callable(model_registry["test_type"]["test_provider"])
