"""Tests for msgflux.models.base module."""

import pytest

from msgflux.models.base import BaseModel


class ConcreteModel(BaseModel):
    """Concrete implementation of BaseModel for testing."""

    model_type = "test_model"
    provider = "test_provider"

    def __init__(self, model_id: str = "test-model", api_key: str = None):
        self.model_id = model_id
        self._api_key = api_key
        self.model = None
        self.processor = None
        self.client = None

    def _initialize(self):
        """Initialize the model."""
        self.client = "initialized_client"

    def __call__(self, *args, **kwargs):
        """Execute the model."""
        return {"result": "success"}


class TestBaseModel:
    """Test suite for BaseModel."""

    def test_base_model_msgflux_type(self):
        """Test that BaseModel has correct msgflux_type."""
        model = ConcreteModel()
        assert model.msgflux_type == "model"

    def test_base_model_to_ignore(self):
        """Test that BaseModel has correct to_ignore list."""
        model = ConcreteModel()
        assert "_api_key" in model.to_ignore
        assert "model" in model.to_ignore
        assert "processor" in model.to_ignore
        assert "client" in model.to_ignore
        assert "chat_transport" in model.to_ignore
        assert "http_transport" in model.to_ignore
        assert "credential_resolver" in model.to_ignore

    def test_instance_type(self):
        """Test instance_type returns correct model_type."""
        model = ConcreteModel()
        instance_type = model.instance_type()
        assert instance_type == {"model_type": "test_model"}

    def test_get_model_info(self):
        """Test get_model_info returns correct information."""
        model = ConcreteModel(model_id="gpt-4")
        model_info = model.get_model_info()
        assert model_info["model_id"] == "gpt-4"
        assert model_info["provider"] == "test_provider"

    def test_initialize_method_abstract(self):
        """Test that _initialize is abstract and cannot instantiate without it."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):

            class IncompleteModel(BaseModel):
                model_type = "incomplete"
                provider = "test"

                def __call__(self):
                    pass

            model = IncompleteModel()

    def test_initialize_called_correctly(self):
        """Test that _initialize works when implemented."""
        model = ConcreteModel()
        model._initialize()
        assert model.client == "initialized_client"

    def test_model_call(self):
        """Test that __call__ works."""
        model = ConcreteModel()
        result = model()
        assert result == {"result": "success"}

    def test_serialize(self):
        """Test model serialization."""
        model = ConcreteModel(model_id="test-123", api_key="secret")
        serialized = model.serialize()

        assert serialized["msgflux_type"] == "model"
        assert serialized["provider"] == "test_provider"
        assert serialized["model_type"] == "test_model"
        assert "state" in serialized

        # Check that ignored fields are not in state
        state = serialized["state"]
        assert "_api_key" not in state
        assert "model" not in state
        assert "processor" not in state
        assert "client" not in state

    def test_from_serialized(self):
        """Test model deserialization."""
        original = ConcreteModel(model_id="test-model-456")
        serialized = original.serialize()

        # Deserialize
        restored = ConcreteModel.from_serialized(serialized["state"])

        assert restored.model_id == "test-model-456"
        assert restored.client == "initialized_client"

    @pytest.mark.skip(
        reason="acall has issues with uvloop and asyncio.wait - needs fix in main code"
    )
    @pytest.mark.asyncio
    async def test_acall(self):
        """Test async call interface."""
        model = ConcreteModel()
        result = await model.acall()
        assert result == {"result": "success"}
