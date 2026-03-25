"""Tests for msgflux.nn.modules.predictor module."""

import pytest

from msgflux.core.dotdict import dotdict
from msgflux.core.message import Message
from msgflux.models.base import BaseModel
from msgflux.models.response import ModelResponse
from msgflux.nn.modules.predictor import Predictor


class MockModel(BaseModel):
    """Mock model for testing Predictor."""

    model_type = "text_classification"
    provider = "mock"

    def __init__(self):
        self.model_id = "mock-classifier"
        self._api_key = None
        self.model = None
        self.processor = None
        self.client = None

    def _initialize(self):
        pass

    def __call__(self, *, data, **kwargs):
        response = ModelResponse()
        response.set_response_type("text_classification")
        response.add({"label": "positive", "score": 0.95})
        return response

    async def acall(self, *, data, **kwargs):
        response = ModelResponse()
        response.set_response_type("text_classification")
        response.add({"label": "positive", "score": 0.95})
        return response


class TestPredictorInit:
    """Test Predictor initialization."""

    def test_basic_init(self):
        model = MockModel()
        predictor = Predictor(name="test", model=model)

        assert predictor.model == model
        assert predictor.config == {}
        assert predictor.task_inputs is None

    def test_init_with_message_fields(self):
        model = MockModel()
        predictor = Predictor(
            name="test",
            model=model,
            message_fields={"task_inputs": "content"},
        )

        assert predictor.task_inputs == "content"

    def test_init_with_config(self):
        model = MockModel()
        predictor = Predictor(
            name="test",
            model=model,
            config={"temperature": 0.7, "top_k": 50},
        )

        assert predictor.config["temperature"] == 0.7
        assert predictor.config["top_k"] == 50

    def test_init_with_response_mode(self):
        model = MockModel()
        predictor = Predictor(
            name="test",
            model=model,
            response_mode="prediction",
        )

        assert predictor.response_mode == "prediction"

    def test_init_with_templates(self):
        model = MockModel()
        predictor = Predictor(
            name="test",
            model=model,
            templates={"response": "Label: {{ label }}"},
        )

        assert predictor.templates["response"] == "Label: {{ label }}"

    def test_init_with_name(self):
        model = MockModel()
        predictor = Predictor(name="my_predictor", model=model)

        assert predictor.name == "my_predictor"
        assert predictor.get_module_name() == "my_predictor"

    def test_invalid_model_raises_type_error(self):
        with pytest.raises(TypeError):
            Predictor(name="test", model="not a model")

    def test_invalid_message_fields_key_raises_value_error(self):
        model = MockModel()
        with pytest.raises(ValueError, match="Invalid keys"):
            Predictor(
                name="test",
                model=model,
                message_fields={"invalid_key": "value"},
            )

    def test_invalid_config_type_raises_type_error(self):
        model = MockModel()
        with pytest.raises(TypeError):
            Predictor(name="test", model=model, config="not a dict")


class TestPredictorPrepareTask:
    """Test _prepare_task method."""

    def test_with_plain_data(self):
        model = MockModel()
        predictor = Predictor(name="test", model=model)

        inputs = predictor._prepare_task("plain data")

        assert inputs.data == "plain data"

    def test_with_message(self):
        model = MockModel()
        predictor = Predictor(
            name="test",
            model=model,
            message_fields={"task_inputs": "content"},
        )

        message = Message(content="test input")
        inputs = predictor._prepare_task(message)

        assert inputs.data == "test input"

    def test_with_list_data(self):
        model = MockModel()
        predictor = Predictor(name="test", model=model)

        inputs = predictor._prepare_task([1.0, 2.0, 3.0])

        assert inputs.data == [1.0, 2.0, 3.0]

    def test_with_model_preference_from_message(self):
        model = MockModel()
        predictor = Predictor(
            name="test",
            model=model,
            message_fields={"model_preference": "context.preferred_model"},
        )

        message = Message(content="test")
        message.context["preferred_model"] = "gpt-4"

        inputs = predictor._prepare_task(message)

        assert inputs.model_preference == "gpt-4"


class TestPredictorModelExecution:
    """Test model execution methods."""

    def test_prepare_model_execution(self):
        model = MockModel()
        predictor = Predictor(name="test", model=model, config={"temperature": 0.5})

        params = predictor._prepare_model_execution("test data")

        assert params.data == "test data"
        assert params.temperature == 0.5

    def test_prepare_model_execution_empty_config(self):
        model = MockModel()
        predictor = Predictor(name="test", model=model)

        params = predictor._prepare_model_execution("test data")

        assert params.data == "test data"

    def test_execute_model(self):
        model = MockModel()
        predictor = Predictor(name="test", model=model)

        response = predictor._execute_model("test input")

        assert isinstance(response, ModelResponse)
        assert response.response_type == "text_classification"

    @pytest.mark.asyncio
    async def test_aexecute_model(self):
        model = MockModel()
        predictor = Predictor(name="test", model=model)

        response = await predictor._aexecute_model("test input")

        assert isinstance(response, ModelResponse)
        assert response.response_type == "text_classification"


class TestPredictorForward:
    """Test forward and aforward methods."""

    def test_forward_plain_data(self):
        model = MockModel()
        predictor = Predictor(name="test", model=model)

        result = predictor("some text")

        assert result["label"] == "positive"
        assert result["score"] == 0.95

    def test_forward_with_response_mode(self):
        model = MockModel()
        predictor = Predictor(
            name="test",
            model=model,
            message_fields={"task_inputs": "content"},
            response_mode="prediction",
        )

        message = Message(content="test content")
        result = predictor(message)

        assert result is None
        assert message.prediction.label == "positive"

    def test_forward_with_templates(self):
        model = MockModel()
        predictor = Predictor(
            name="test",
            model=model,
            templates={"response": "{{ label }} ({{ score }})"},
        )

        result = predictor("test")

        assert result == "positive (0.95)"

    @pytest.mark.asyncio
    async def test_aforward_plain_data(self):
        model = MockModel()
        predictor = Predictor(name="test", model=model)

        result = await predictor.acall("some text")

        assert result["label"] == "positive"

    @pytest.mark.asyncio
    async def test_aforward_with_response_mode(self):
        model = MockModel()
        predictor = Predictor(
            name="test",
            model=model,
            message_fields={"task_inputs": "content"},
            response_mode="prediction",
        )

        message = Message(content="async test")
        result = await predictor.acall(message)

        assert result is None
        assert message.prediction.label == "positive"

    def test_inspect_model_execution_params(self):
        model = MockModel()
        predictor = Predictor(
            name="test",
            model=model,
            message_fields={"task_inputs": "content"},
            config={"temperature": 0.7},
        )

        message = Message(content="test")
        params = predictor.inspect_model_execution_params(message)

        assert params.data == "test"
        assert params.temperature == 0.7


class TestPredictorProcessResponse:
    """Test response processing."""

    def test_process_model_response(self):
        model = MockModel()
        predictor = Predictor(name="test", model=model)

        response = ModelResponse()
        response.set_response_type("text_classification")
        response.add({"label": "negative"})

        result = predictor._process_model_response(response, "message")

        assert result["label"] == "negative"

    def test_process_response_writes_to_message(self):
        model = MockModel()
        predictor = Predictor(
            name="test",
            model=model,
            message_fields={"task_inputs": "text"},
            response_mode="result",
        )

        response = ModelResponse()
        response.set_response_type("text_classification")
        response.add({"label": "spam"})

        message = Message()
        message.text = "buy now"

        raw = predictor._extract_raw_response(response)
        predictor._prepare_response(raw, message)

        assert message.result.label == "spam"
