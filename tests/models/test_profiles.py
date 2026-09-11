from msgflux.models.profiles.base import ModelCost


def test_model_cost_replaces_input_rate_for_cached_tokens():
    cost = ModelCost(
        input_per_million=10,
        output_per_million=20,
        cache_read_per_million=2,
    )

    total = cost.calculate(
        input_tokens=1_000,
        output_tokens=100,
        cached_tokens=800,
    )

    assert total == 0.0056


def test_model_cost_caps_cached_tokens_at_input_tokens():
    cost = ModelCost(
        input_per_million=10,
        output_per_million=20,
        cache_read_per_million=2,
    )

    total = cost.calculate(
        input_tokens=1_000,
        output_tokens=0,
        cached_tokens=2_000,
    )

    assert total == 0.002


def test_model_cost_preserves_free_cached_input_rate():
    cost = ModelCost(
        input_per_million=10,
        output_per_million=20,
        cache_read_per_million=0,
    )

    total = cost.calculate(
        input_tokens=1_000,
        output_tokens=0,
        cached_tokens=1_000,
    )

    assert total == 0
