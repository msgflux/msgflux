"""Tests for msgflux.models.gateway module."""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from msgflux.exceptions import ModelRouterError
from msgflux.models.base import BaseModel
from msgflux.models.gateway import ModelGateway
from msgflux.models.response import ModelResponse
from msgflux.models.compaction import ModelCompaction


class MockModel(BaseModel):
    """Mock model for testing gateway."""

    def __init__(
        self,
        model_id: str,
        model_type: str = "chat_completion",
        provider: str = "mock",
        *,
        should_fail: bool = False,
    ):
        self.model_id = model_id
        self.model_type = model_type
        self.provider = provider
        self.should_fail = should_fail
        self.call_count = 0
        self.reasoning_effort = None
        self.speed = None

    def _initialize(self):
        pass

    def __call__(self, **kwargs):
        self.call_count += 1
        if self.should_fail:
            raise RuntimeError(f"Mock failure for {self.model_id}")
        response = ModelResponse()
        response.add(f"Response from {self.model_id}")
        return response

    async def acall(self, **kwargs):
        self.call_count += 1
        if self.should_fail:
            raise RuntimeError(f"Mock async failure for {self.model_id}")
        response = ModelResponse()
        response.add(f"Async response from {self.model_id}")
        return response

    def set_reasoning_effort(self, reasoning_effort):
        self.reasoning_effort = reasoning_effort
        return self

    def set_speed(self, speed):
        self.speed = speed
        return self

    def serialize(self):
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "provider": self.provider,
        }


class CompactingMockModel(MockModel):
    def __init__(self, model_id: str, *, capacity: int | None, provider: str):
        super().__init__(model_id, provider=provider)
        self._capacity = capacity
        self.compaction_native_values = []

    @property
    def context_capacity(self):
        return self._capacity

    def compact_context(self, messages, *, system_prompt=None, native=True):
        self.compaction_native_values.append(native)
        return ModelCompaction(
            format="provider" if native else "messages",
            items=[{"role": "system", "content": self.model_id}],
            provider=self.provider,
            api_mode="responses",
            model_id=self.model_id,
        )

    async def acompact_context(self, messages, *, system_prompt=None, native=True):
        return self.compact_context(
            messages,
            system_prompt=system_prompt,
            native=native,
        )


def _deployment(
    name: str,
    model_id: str | None = None,
    model_type: str = "chat_completion",
    provider: str = "mock",
    *,
    should_fail: bool = False,
    time_constraints=None,
    description: str | None = None,
):
    """Helper to build a deployment dict."""
    if model_id is None:
        model_id = name
    entry = {
        "model_name": name,
        "model": MockModel(
            model_id=model_id,
            model_type=model_type,
            provider=provider,
            should_fail=should_fail,
        ),
    }
    if time_constraints is not None:
        entry["time_constraints"] = time_constraints
    if description is not None:
        entry["description"] = description
    return entry


class TestModelGatewayInitialization:
    """Test suite for ModelGateway initialization."""

    def test_gateway_initialization_basic(self):
        """Test basic ModelGateway initialization."""
        models = [
            _deployment("model-1"),
            _deployment("model-2"),
        ]

        gateway = ModelGateway(models=models)

        assert len(gateway.models) == 2
        assert gateway.model_type == "chat_completion"
        assert gateway.current_model_index == 0
        assert gateway.model_names == ["model-1", "model-2"]
        assert gateway.fallback is True
        assert gateway.model_descriptions == {
            "model-1": None,
            "model-2": None,
        }

    @patch("msgflux.models.gateway.Model.chat_completion")
    def test_gateway_accepts_chat_completion_shorthand(self, mock_factory):
        model = MockModel("gpt-5.6-luna", provider="openai")
        mock_factory.return_value = model

        gateway = ModelGateway(
            models=[
                {
                    "model_name": "fast",
                    "model": "openai/gpt-5.6-luna",
                    "description": "  Fast for repeatable tasks.  ",
                }
            ]
        )

        mock_factory.assert_called_once_with("openai/gpt-5.6-luna")
        assert gateway.models == [model]
        assert gateway.get_model_description("fast") == "Fast for repeatable tasks."

    @pytest.mark.parametrize("description", ["", "   ", 42])
    def test_gateway_rejects_invalid_description(self, description):
        deployment = _deployment("model-1")
        deployment["description"] = description

        with pytest.raises(TypeError, match=r"description.*non-empty string"):
            ModelGateway(models=[deployment])

    def test_gateway_rejects_non_boolean_fallback(self):
        with pytest.raises(TypeError, match="`fallback` must be a bool"):
            ModelGateway(models=[_deployment("model-1")], fallback="yes")

    def test_set_reasoning_effort_updates_every_fallback_model(self):
        gateway = ModelGateway(models=[_deployment("fast"), _deployment("strong")])

        result = gateway.set_reasoning_effort("high")

        assert result is gateway
        assert [model.reasoning_effort for model in gateway.models] == [
            "high",
            "high",
        ]

    def test_set_reasoning_effort_can_target_one_deployment(self):
        gateway = ModelGateway(models=[_deployment("fast"), _deployment("strong")])

        gateway.set_reasoning_effort("low", model_name="fast")

        assert gateway.models[0].reasoning_effort == "low"
        assert gateway.models[1].reasoning_effort is None

    def test_set_speed_updates_every_fallback_model(self):
        gateway = ModelGateway(models=[_deployment("fast"), _deployment("strong")])

        result = gateway.set_speed("fast")

        assert result is gateway
        assert [model.speed for model in gateway.models] == ["fast", "fast"]

    def test_set_speed_can_target_one_deployment(self):
        gateway = ModelGateway(models=[_deployment("fast"), _deployment("strong")])

        gateway.set_speed("ultrafast", model_name="strong")

        assert gateway.models[0].speed is None
        assert gateway.models[1].speed == "ultrafast"

    @pytest.mark.parametrize("speed", ["slow", "priority", "", 1, True])
    def test_set_speed_rejects_invalid_values(self, speed):
        gateway = ModelGateway(models=[_deployment("fast"), _deployment("strong")])

        with pytest.raises((TypeError, ValueError)):
            gateway.set_speed(speed)

        assert all(model.speed is None for model in gateway.models)

    @pytest.mark.parametrize("reasoning_effort", ["", "   ", 1, True])
    def test_set_reasoning_effort_rejects_invalid_values(self, reasoning_effort):
        gateway = ModelGateway(models=[_deployment("fast"), _deployment("strong")])

        with pytest.raises(TypeError, match="non-empty string or None"):
            gateway.set_reasoning_effort(reasoning_effort)

        assert all(model.reasoning_effort is None for model in gateway.models)

    def test_gateway_initialization_with_time_constraints(self):
        """Test ModelGateway with time constraints inside deployments."""
        models = [
            _deployment("model-1", time_constraints=[("22:00", "06:00")]),
            _deployment("model-2"),
        ]

        gateway = ModelGateway(models=models)

        assert gateway.raw_time_constraints == {"model-1": [("22:00", "06:00")]}
        assert "model-1" in gateway.parsed_time_constraints

    def test_gateway_empty_models_list(self):
        """Test ModelGateway raises error with empty models list."""
        with pytest.raises(TypeError, match="`models` must be a non-empty list"):
            ModelGateway(models=[])

    def test_gateway_invalid_models_type(self):
        """Test ModelGateway raises error with invalid models type."""
        with pytest.raises(TypeError, match="`models` must be a non-empty list"):
            ModelGateway(models="not-a-list")

    def test_gateway_non_dict_elements(self):
        """Test ModelGateway raises error when elements are not dicts."""
        with pytest.raises(TypeError, match="requires a list of dicts"):
            ModelGateway(models=[_deployment("model-1"), "not-a-dict"])

    def test_gateway_missing_model_name(self):
        """Test ModelGateway raises error when model_name is missing."""
        with pytest.raises(ValueError, match="missing required key `model_name`"):
            ModelGateway(models=[{"model": MockModel("m1")}, _deployment("model-2")])

    def test_gateway_missing_model(self):
        """Test ModelGateway raises error when model is missing."""
        with pytest.raises(ValueError, match="missing required key `model`"):
            ModelGateway(models=[{"model_name": "m1"}, _deployment("model-2")])

    def test_gateway_non_basemodel_instances(self):
        """Test ModelGateway raises error when model doesn't inherit BaseModel."""
        with pytest.raises(TypeError, match="does not inherit from `BaseModel`"):
            ModelGateway(
                models=[
                    {"model_name": "m1", "model": "not-a-model"},
                    _deployment("model-2"),
                ]
            )

    def test_gateway_single_model_warning(self):
        """Test ModelGateway with only one model (should warn but not fail)."""
        models = [_deployment("model-1")]

        gateway = ModelGateway(models=models)
        assert len(gateway.models) == 1

    def test_gateway_mixed_model_types(self):
        """Test ModelGateway raises error with different model types."""
        models = [
            _deployment("model-1", model_type="chat_completion"),
            _deployment("model-2", model_type="text_embedder"),
        ]

        with pytest.raises(TypeError, match="must be of the same `model_type`"):
            ModelGateway(models=models)

    def test_gateway_duplicate_model_names(self):
        """Test ModelGateway raises error with duplicate model names."""
        models = [
            _deployment("same-name", model_id="model-1"),
            _deployment("same-name", model_id="model-2"),
        ]

        with pytest.raises(ValueError, match="Duplicate model name"):
            ModelGateway(models=models)

    def test_gateway_model_without_model_type(self):
        """Test ModelGateway raises error when model lacks model_type."""
        model = MockModel("model-1")
        del model.model_type

        with pytest.raises(AttributeError, match="does not have a valid `model_type`"):
            ModelGateway(
                models=[
                    {"model_name": "m1", "model": model},
                    _deployment("model-2"),
                ]
            )

    def test_gateway_model_without_model_id(self):
        """Test ModelGateway raises error when model lacks model_id."""
        model = MockModel("model-1")
        del model.model_id

        with pytest.raises(AttributeError, match="does not have a valid `model_id`"):
            ModelGateway(
                models=[
                    {"model_name": "m1", "model": model},
                    _deployment("model-2"),
                ]
            )

    def test_gateway_model_without_provider(self):
        """Test ModelGateway raises error when model lacks provider."""
        model = MockModel("model-1")
        del model.provider

        with pytest.raises(AttributeError, match="does not have a valid `provider`"):
            ModelGateway(
                models=[
                    {"model_name": "m1", "model": model},
                    _deployment("model-2"),
                ]
            )

    def test_gateway_model_name_as_alias(self):
        """Test that model_name can be any arbitrary string alias."""
        models = [
            _deployment("weak", model_id="gpt-4.1-mini"),
            _deployment("strong", model_id="gpt-4.1"),
        ]

        gateway = ModelGateway(models=models)

        assert gateway.model_names == ["weak", "strong"]
        assert gateway.models[0].model_id == "gpt-4.1-mini"
        assert gateway.models[1].model_id == "gpt-4.1"

    def test_gateway_uses_smallest_known_context_capacity(self):
        gateway = ModelGateway(
            models=[
                {
                    "model_name": "large",
                    "model": CompactingMockModel(
                        "large", capacity=200_000, provider="openai"
                    ),
                },
                {
                    "model_name": "small",
                    "model": CompactingMockModel(
                        "small", capacity=100_000, provider="openrouter"
                    ),
                },
            ]
        )

        assert gateway.context_capacity == 100_000

    def test_gateway_context_capacity_is_unknown_if_one_model_is_unknown(self):
        gateway = ModelGateway(
            models=[
                {
                    "model_name": "known",
                    "model": CompactingMockModel(
                        "known", capacity=100_000, provider="openai"
                    ),
                },
                {
                    "model_name": "unknown",
                    "model": CompactingMockModel(
                        "unknown", capacity=None, provider="openrouter"
                    ),
                },
            ]
        )

        assert gateway.context_capacity is None

    def test_gateway_forces_portable_compaction_across_providers(self):
        first = CompactingMockModel("first", capacity=100_000, provider="openai")
        second = CompactingMockModel("second", capacity=100_000, provider="openrouter")
        gateway = ModelGateway(
            models=[
                {"model_name": "first", "model": first},
                {"model_name": "second", "model": second},
            ]
        )

        compacted = gateway.compact_context([])

        assert compacted.format == "messages"
        assert first.compaction_native_values == [False]
        assert second.compaction_native_values == []


class TestTimeConstraintParsing:
    """Test suite for time constraint parsing."""

    def test_parse_time_constraints_valid(self):
        """Test parsing valid time constraints."""
        models = [
            _deployment("model-1", time_constraints=[("09:00", "17:00")]),
            _deployment("model-2", time_constraints=[("22:00", "06:00")]),
        ]

        gateway = ModelGateway(models=models)

        assert "model-1" in gateway.parsed_time_constraints
        assert "model-2" in gateway.parsed_time_constraints
        assert len(gateway.parsed_time_constraints["model-1"]) == 1

    def test_parse_time_constraints_multiple_intervals(self):
        """Test parsing multiple time intervals for a model."""
        models = [
            _deployment(
                "model-1",
                time_constraints=[("09:00", "12:00"), ("13:00", "17:00")],
            ),
            _deployment("model-2"),
        ]

        gateway = ModelGateway(models=models)

        assert len(gateway.parsed_time_constraints["model-1"]) == 2

    def test_parse_time_constraints_invalid_format(self):
        """Test parsing time constraints with invalid time format."""
        models = [
            _deployment("model-1", time_constraints=[("25:00", "06:00")]),
            _deployment("model-2"),
        ]

        with pytest.raises(ValueError, match="Invalid format in time"):
            ModelGateway(models=models)

    def test_parse_time_constraints_not_list(self):
        """Test parsing time constraints when intervals is not a list."""
        models = [
            {
                "model_name": "model-1",
                "model": MockModel("m1"),
                "time_constraints": "not-a-list",
            },
            _deployment("model-2"),
        ]

        with pytest.raises(TypeError, match="must be a list of tuples"):
            ModelGateway(models=models)

    def test_parse_time_constraints_invalid_interval_type(self):
        """Test parsing time constraints with invalid interval type."""
        models = [
            {
                "model_name": "model-1",
                "model": MockModel("m1"),
                "time_constraints": ["not-a-tuple"],
            },
            _deployment("model-2"),
        ]

        with pytest.raises(TypeError, match="must be a tuple/list of two strings"):
            ModelGateway(models=models)

    def test_parse_time_constraints_non_string_times(self):
        """Test parsing time constraints with non-string times."""
        models = [
            {
                "model_name": "model-1",
                "model": MockModel("m1"),
                "time_constraints": [(9, 17)],
            },
            _deployment("model-2"),
        ]

        with pytest.raises(TypeError, match="must be strings"):
            ModelGateway(models=models)

    def test_no_time_constraints(self):
        """Test gateway without any time constraints."""
        models = [_deployment("model-1"), _deployment("model-2")]

        gateway = ModelGateway(models=models)

        assert gateway.raw_time_constraints is None
        assert gateway.parsed_time_constraints == {}


class TestTimeRestriction:
    """Test suite for time restriction checking."""

    def test_is_time_restricted_no_constraints(self):
        """Test model with no time constraints is not restricted."""
        models = [_deployment("model-1"), _deployment("model-2")]
        gateway = ModelGateway(models=models)

        assert not gateway._is_time_restricted("model-1")

    def test_is_time_restricted_within_range(self):
        """Test model is restricted when current time is within range."""
        with patch(
            "msgflux.models.gateway.utc_now",
            return_value=datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
        ):
            models = [
                _deployment("model-1", time_constraints=[("09:00", "17:00")]),
                _deployment("model-2"),
            ]

            gateway = ModelGateway(models=models)
            assert gateway._is_time_restricted("model-1")

    def test_is_time_restricted_outside_range(self):
        """Test model is not restricted when current time is outside range."""
        with patch(
            "msgflux.models.gateway.utc_now",
            return_value=datetime(2025, 1, 1, 8, 0, 0, tzinfo=timezone.utc),
        ):
            models = [
                _deployment("model-1", time_constraints=[("09:00", "17:00")]),
                _deployment("model-2"),
            ]

            gateway = ModelGateway(models=models)
            assert not gateway._is_time_restricted("model-1")

    def test_is_time_restricted_midnight_crossover(self):
        """Test time restriction crossing midnight (e.g., 22:00 to 06:00)."""
        # Mock current time to be 23:00 (restricted)
        with patch(
            "msgflux.models.gateway.utc_now",
            return_value=datetime(2025, 1, 1, 23, 0, 0, tzinfo=timezone.utc),
        ):
            models = [
                _deployment("model-1", time_constraints=[("22:00", "06:00")]),
                _deployment("model-2"),
            ]

            gateway = ModelGateway(models=models)
            assert gateway._is_time_restricted("model-1")

        # Mock current time to be 03:00 (also restricted)
        with patch(
            "msgflux.models.gateway.utc_now",
            return_value=datetime(2025, 1, 1, 3, 0, 0, tzinfo=timezone.utc),
        ):
            models = [
                _deployment("model-1", time_constraints=[("22:00", "06:00")]),
                _deployment("model-2"),
            ]

            gateway = ModelGateway(models=models)
            assert gateway._is_time_restricted("model-1")


class TestModelExecution:
    """Test suite for model execution."""

    def test_execute_model_basic(self):
        """Test basic model execution."""
        models = [_deployment("model-1"), _deployment("model-2")]
        gateway = ModelGateway(models=models)

        response = gateway()

        assert response is not None
        assert gateway.models[0].call_count == 1
        assert "model-1" in response.data

    def test_execute_model_with_fallback(self):
        """Test fallback when first model fails."""
        models = [
            _deployment("model-1", should_fail=True),
            _deployment("model-2"),
        ]
        gateway = ModelGateway(models=models)

        response = gateway()

        assert response is not None
        assert gateway.models[0].call_count == 1
        assert gateway.models[1].call_count == 1
        assert "model-2" in response.data

    def test_execute_model_all_fail(self):
        """Test error when all models fail."""
        models = [
            _deployment("model-1", should_fail=True),
            _deployment("model-2", should_fail=True),
        ]
        gateway = ModelGateway(models=models)

        with pytest.raises(ModelRouterError, match=r"All .* available models failed"):
            gateway()

    def test_execute_model_with_preference(self):
        """Test model preference is respected using model_name."""
        models = [_deployment("weak"), _deployment("strong")]
        gateway = ModelGateway(models=models)

        response = gateway(model_preference="strong")

        assert response is not None
        assert gateway.models[1].call_count == 1
        assert gateway.models[0].call_count == 0
        assert "strong" in response.data

    def test_execute_model_with_preference_fallback(self):
        """Test fallback when preferred model fails."""
        models = [
            _deployment("weak"),
            _deployment("strong", should_fail=True),
        ]
        gateway = ModelGateway(models=models)

        response = gateway(model_preference="strong")

        assert response is not None
        assert gateway.models[1].call_count == 1
        assert gateway.models[0].call_count == 1
        assert "weak" in response.data

    def test_execute_model_rejects_unknown_preference(self):
        gateway = ModelGateway(models=[_deployment("weak"), _deployment("strong")])

        with pytest.raises(ValueError, match="Unknown model `missing`"):
            gateway(model_preference="missing")

        assert all(model.call_count == 0 for model in gateway.models)

    def test_execute_model_without_fallback_does_not_transition(self):
        gateway = ModelGateway(
            models=[
                _deployment("weak"),
                _deployment("strong", should_fail=True),
            ],
            fallback=False,
        )

        with pytest.raises(ModelRouterError, match="All 1 available models failed"):
            gateway(model_preference="strong")

        assert gateway.models[1].call_count == 1
        assert gateway.models[0].call_count == 0

    def test_execute_model_without_fallback_uses_only_first_by_default(self):
        gateway = ModelGateway(
            models=[
                _deployment("first", should_fail=True),
                _deployment("second"),
            ],
            fallback=False,
        )

        with pytest.raises(ModelRouterError, match="All 1 available models failed"):
            gateway()

        assert gateway.models[0].call_count == 1
        assert gateway.models[1].call_count == 0

    def test_execute_model_with_kwargs(self):
        """Test passing kwargs to model."""
        models = [_deployment("model-1"), _deployment("model-2")]
        gateway = ModelGateway(models=models)

        response = gateway(temperature=0.7, max_tokens=100)

        assert response is not None
        assert gateway.models[0].call_count == 1

    def test_execute_model_time_restricted(self):
        """Test execution skips time-restricted models."""
        with patch(
            "msgflux.models.gateway.utc_now",
            return_value=datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
        ):
            models = [
                _deployment("model-1", time_constraints=[("09:00", "17:00")]),
                _deployment("model-2"),
            ]

            gateway = ModelGateway(models=models)
            response = gateway()

            # model-1 is restricted, so model-2 should be used
            assert gateway.models[0].call_count == 0
            assert gateway.models[1].call_count == 1
            assert "model-2" in response.data

    def test_execute_model_all_restricted(self):
        """Test error when all models are time-restricted."""
        with patch(
            "msgflux.models.gateway.utc_now",
            return_value=datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
        ):
            models = [
                _deployment("model-1", time_constraints=[("09:00", "17:00")]),
                _deployment("model-2", time_constraints=[("09:00", "17:00")]),
            ]

            gateway = ModelGateway(models=models)

            with pytest.raises(
                ModelRouterError,
                match="No model available due to time constraints",
            ):
                gateway()

    def test_execute_model_without_fallback_rejects_restricted_selection(self):
        with patch(
            "msgflux.models.gateway.utc_now",
            return_value=datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
        ):
            gateway = ModelGateway(
                models=[
                    _deployment(
                        "restricted",
                        time_constraints=[("09:00", "17:00")],
                    ),
                    _deployment("fallback"),
                ],
                fallback=False,
            )

            with pytest.raises(
                ModelRouterError,
                match="Selected model `restricted` is unavailable",
            ):
                gateway(model_preference="restricted")

            assert all(model.call_count == 0 for model in gateway.models)

    @pytest.mark.asyncio
    async def test_aexecute_model_basic(self):
        """Test basic async model execution."""
        models = [_deployment("model-1"), _deployment("model-2")]
        gateway = ModelGateway(models=models)

        response = await gateway.acall()

        assert response is not None
        assert gateway.models[0].call_count == 1
        assert "model-1" in response.data

    @pytest.mark.asyncio
    async def test_aexecute_model_with_fallback(self):
        """Test async fallback when first model fails."""
        models = [
            _deployment("model-1", should_fail=True),
            _deployment("model-2"),
        ]
        gateway = ModelGateway(models=models)

        response = await gateway.acall()

        assert response is not None
        assert gateway.models[0].call_count == 1
        assert gateway.models[1].call_count == 1
        assert "model-2" in response.data

    @pytest.mark.asyncio
    async def test_aexecute_model_without_fallback_does_not_transition(self):
        gateway = ModelGateway(
            models=[
                _deployment("weak"),
                _deployment("strong", should_fail=True),
            ],
            fallback=False,
        )

        with pytest.raises(ModelRouterError, match="All 1 available models failed"):
            await gateway.acall(model_preference="strong")

        assert gateway.models[1].call_count == 1
        assert gateway.models[0].call_count == 0

    @pytest.mark.asyncio
    async def test_aexecute_model_all_fail(self):
        """Test async error when all models fail."""
        models = [
            _deployment("model-1", should_fail=True),
            _deployment("model-2", should_fail=True),
        ]
        gateway = ModelGateway(models=models)

        with pytest.raises(ModelRouterError, match=r"All .* available models failed"):
            await gateway.acall()


class TestGatewaySerialization:
    """Test suite for gateway serialization."""

    def test_serialize_basic(self):
        """Test basic gateway serialization."""
        models = [_deployment("model-1"), _deployment("model-2")]
        gateway = ModelGateway(models=models)

        serialized = gateway.serialize()

        assert "msgflux_type" in serialized
        assert serialized["msgflux_type"] == "model_gateway"
        assert "state" in serialized
        assert "models" in serialized["state"]
        assert len(serialized["state"]["models"]) == 2
        assert serialized["state"]["models"][0]["model_name"] == "model-1"
        assert serialized["state"]["models"][1]["model_name"] == "model-2"

    def test_serialize_with_time_constraints(self):
        """Test serialization preserves time constraints in deployments."""
        models = [
            _deployment("model-1", time_constraints=[("09:00", "17:00")]),
            _deployment("model-2"),
        ]
        gateway = ModelGateway(models=models)

        serialized = gateway.serialize()

        deployments = serialized["state"]["models"]
        assert deployments[0]["time_constraints"] == [("09:00", "17:00")]
        assert "time_constraints" not in deployments[1]

    def test_serialize_preserves_descriptions_and_fallback_policy(self):
        gateway = ModelGateway(
            models=[
                _deployment("fast", description="Fast for repeatable tasks."),
                _deployment("deep", description="Deep reasoning for complex tasks."),
            ],
            fallback=False,
        )

        serialized = gateway.serialize()

        assert serialized["state"]["fallback"] is False
        assert serialized["state"]["models"][0]["description"] == (
            "Fast for repeatable tasks."
        )

    @patch("msgflux.models.model.Model.from_serialized")
    def test_from_serialized_basic(self, mock_from_serialized):
        """Test deserializing gateway from data."""
        mock_from_serialized.side_effect = [
            MockModel("model-1"),
            MockModel("model-2"),
        ]

        data = {
            "msgflux_type": "model_gateway",
            "state": {
                "models": [
                    {
                        "model_name": "weak",
                        "model": {"model_id": "model-1"},
                    },
                    {
                        "model_name": "strong",
                        "model": {"model_id": "model-2"},
                    },
                ],
            },
        }

        gateway = ModelGateway.from_serialized(data)

        assert len(gateway.models) == 2
        assert gateway.model_names == ["weak", "strong"]
        assert mock_from_serialized.call_count == 2

    @patch("msgflux.models.model.Model.from_serialized")
    def test_from_serialized_with_time_constraints(self, mock_from_serialized):
        """Test deserializing gateway with time constraints."""
        mock_from_serialized.side_effect = [
            MockModel("model-1"),
            MockModel("model-2"),
        ]

        data = {
            "msgflux_type": "model_gateway",
            "state": {
                "models": [
                    {
                        "model_name": "weak",
                        "model": {"model_id": "model-1"},
                        "time_constraints": [("09:00", "17:00")],
                    },
                    {
                        "model_name": "strong",
                        "model": {"model_id": "model-2"},
                    },
                ],
            },
        }

        gateway = ModelGateway.from_serialized(data)

        assert gateway.raw_time_constraints == {"weak": [("09:00", "17:00")]}

    @patch("msgflux.models.model.Model.from_serialized")
    def test_from_serialized_restores_descriptions_and_fallback(
        self, mock_from_serialized
    ):
        mock_from_serialized.side_effect = [
            MockModel("model-1"),
            MockModel("model-2"),
        ]
        data = {
            "msgflux_type": "model_gateway",
            "state": {
                "fallback": False,
                "models": [
                    {
                        "model_name": "fast",
                        "model": {"model_id": "model-1"},
                        "description": "Fast for repeatable tasks.",
                    },
                    {
                        "model_name": "deep",
                        "model": {"model_id": "model-2"},
                        "description": "Deep reasoning for complex tasks.",
                    },
                ],
            },
        }

        gateway = ModelGateway.from_serialized(data)

        assert gateway.fallback is False
        assert gateway.model_descriptions == {
            "fast": "Fast for repeatable tasks.",
            "deep": "Deep reasoning for complex tasks.",
        }

    def test_from_serialized_invalid_type(self):
        """Test error when deserializing with wrong msgflux_type."""
        data = {
            "msgflux_type": "wrong_type",
            "state": {},
        }

        with pytest.raises(ValueError, match="Incorrect msgflux type"):
            ModelGateway.from_serialized(data)

    def test_from_serialized_no_models(self):
        """Test error when deserializing without models."""
        data = {
            "msgflux_type": "model_gateway",
            "state": {
                "models": [],
            },
        }

        with pytest.raises(ValueError, match="does not contain models"):
            ModelGateway.from_serialized(data)
