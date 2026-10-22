"""Tests for programmatic chat provider registration."""

import pytest

from msgflux.models.registry import model_registry


@pytest.fixture(autouse=True)
def _cleanup_test_providers():
    yield
    providers = model_registry.get("chat_completion", {})
    for name in [name for name in providers if name.startswith("test-")]:
        del providers[name]


def test_register_minimal_provider(monkeypatch):
    import msgflux as mf

    monkeypatch.setenv("TEST_MINIMAL_API_KEY", "k")
    cls = mf.register_chat_provider(
        "test-minimal",
        base_url="https://api.example.com/v1",
        api_key_env="TEST_MINIMAL_API_KEY",
    )

    assert cls.provider == "test-minimal"
    assert cls.display_name == "test-minimal"
    assert cls.api_key_env == "TEST_MINIMAL_API_KEY"
    model = mf.Model.chat_completion("test-minimal/some-model")

    assert model.provider == "test-minimal"
    assert model.api_mode == "chat_completions"
    assert model.model_id == "some-model"


def test_register_both_modes_with_default(monkeypatch):
    import msgflux as mf

    monkeypatch.setenv("TEST_DUAL_API_KEY", "k")
    mf.register_chat_provider(
        "test-dual",
        base_url="https://api.example.com/v1",
        api_key_env="TEST_DUAL_API_KEY",
        display_name="Test Dual",
        api_modes=("chat_completions", "responses"),
        default_api_mode="responses",
    )
    model = mf.Model.chat_completion("test-dual/some-model")

    assert model.api_mode == "responses"
    assert model._get_api_key() == "k"
    assert model._get_base_url() == "https://api.example.com/v1"


def test_register_rejects_duplicates():
    import msgflux as mf

    mf.register_chat_provider(
        "test-dupe",
        base_url="https://api.example.com/v1",
        api_key_env="TEST_DUPE_API_KEY",
    )

    with pytest.raises(ValueError, match="already registered"):
        mf.register_chat_provider(
            "test-dupe",
            base_url="https://api.example.com/v1",
            api_key_env="TEST_DUPE_API_KEY",
        )


def test_register_overwrite_replaces():
    import msgflux as mf

    mf.register_chat_provider(
        "test-over",
        base_url="https://old.example.com/v1",
        api_key_env="TEST_OVER_API_KEY",
    )
    cls = mf.register_chat_provider(
        "test-over",
        base_url="https://new.example.com/v1",
        api_key_env="TEST_OVER_API_KEY",
        overwrite=True,
    )

    assert cls.base_url == "https://new.example.com/v1"


def test_register_validates_arguments():
    import msgflux as mf

    with pytest.raises(ValueError, match="`provider`"):
        mf.register_chat_provider("bad/name", base_url="https://x/v1", api_key_env="K")
    with pytest.raises(ValueError, match="`base_url`"):
        mf.register_chat_provider("test-bad", base_url="  ", api_key_env="K")
    with pytest.raises(ValueError, match="`api_modes`"):
        mf.register_chat_provider(
            "test-bad",
            base_url="https://x/v1",
            api_key_env="K",
            api_modes=("chat_completions", "nope"),
        )
    with pytest.raises(ValueError, match="`default_api_mode`"):
        mf.register_chat_provider(
            "test-bad",
            base_url="https://x/v1",
            api_key_env="K",
            api_modes=("chat_completions",),
            default_api_mode="responses",
        )
