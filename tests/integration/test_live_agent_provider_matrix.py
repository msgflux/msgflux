"""Opt-in live Agent regression matrix for OpenAI-compatible chat providers.

The base matrix is skipped unless ``MSGFLUX_LIVE_AGENT_PROVIDER_MATRIX=1`` is
set; the more expensive output stress case additionally requires
``MSGFLUX_LIVE_AGENT_STRESS=1``. Each selected provider uses
its own ``MSGFLUX_LIVE_<PROVIDER>_API_KEY``, ``..._MODEL`` and optional
``..._BASE_URL`` environment variables. Keys are only read by the runtime
credential resolver; this test never prints or persists their values.

For example, run one provider with its credentials/model configured, then:

    MSGFLUX_LIVE_AGENT_PROVIDER_MATRIX=1 \
      MSGFLUX_LIVE_AGENT_PROVIDERS=openrouter \
      uv run pytest -q tests/integration/test_live_agent_provider_matrix.py

The test makes real, billable requests: one tool-using streamed Agent run and
one follow-up run that reads the durable thread checkpoint. Keep provider
budgets and rate limits in mind. Hosted model IDs and endpoints may change;
override either with the matching ``MSGFLUX_LIVE_<PROVIDER>_*`` variable.
The stress opt-in adds a 512 KiB synthetic tool result, offload verification,
and SQLite checkpoint/event metrics recorded as pytest properties.
"""

from __future__ import annotations

import os
import asyncio
from pathlib import Path
from time import perf_counter

import msgspec
import pytest


_OPT_IN = "MSGFLUX_LIVE_AGENT_PROVIDER_MATRIX"
_PROVIDER_FILTER = "MSGFLUX_LIVE_AGENT_PROVIDERS"
_REQUIRE_ALL = "MSGFLUX_LIVE_AGENT_REQUIRE_ALL"
_STRESS_OPT_IN = "MSGFLUX_LIVE_AGENT_STRESS"


class ProviderConfig(msgspec.Struct, frozen=True):
    name: str
    api_key_env: str
    base_url: str | None
    model: str
    max_tokens: int = 384


_PROVIDER_DEFAULTS = {
    "openai": ("OPENAI_API_KEY", None, "gpt-4.1-mini"),
    "openrouter": (
        "OPENROUTER_API_KEY",
        "https://openrouter.ai/api/v1",
        "openai/gpt-4.1-mini",
    ),
    "groq": (
        "GROQ_API_KEY",
        "https://api.groq.com/openai/v1",
        "openai/gpt-oss-20b",
    ),
    "baseten": (
        "BASETEN_API_KEY",
        "https://inference.baseten.co/v1",
        "zai-org/GLM-5.2",
    ),
    "fireworks": (
        "FIREWORKS_API_KEY",
        "https://api.fireworks.ai/inference/v1",
        "accounts/fireworks/models/gpt-oss-120b",
    ),
    "nvidia": (
        "NVIDIA_API_KEY",
        "https://integrate.api.nvidia.com/v1",
        "openai/gpt-oss-20b",
    ),
}


def _provider_configs() -> tuple[list[ProviderConfig], list[str]]:
    """Return only fully configured providers; never read secrets into reports."""
    if os.getenv(_OPT_IN) != "1" and os.getenv(_STRESS_OPT_IN) != "1":
        return [], []

    explicit_filter = _PROVIDER_FILTER in os.environ
    require_all = os.getenv(_REQUIRE_ALL) == "1"
    requested = {
        name.strip().lower()
        for name in os.getenv(_PROVIDER_FILTER, ",".join(_PROVIDER_DEFAULTS)).split(",")
        if name.strip()
    }
    if require_all:
        requested = set(_PROVIDER_DEFAULTS)
    unknown = requested - _PROVIDER_DEFAULTS.keys()
    if unknown:
        raise pytest.UsageError(
            f"Unknown live provider name(s): {', '.join(sorted(unknown))}"
        )

    configs = []
    missing = []
    for name in sorted(requested):
        default_key_env, default_url, default_model = _PROVIDER_DEFAULTS[name]
        prefix = f"MSGFLUX_LIVE_{name.upper()}"
        key_env = os.getenv(f"{prefix}_KEY_ENV", default_key_env)
        model = os.getenv(f"{prefix}_MODEL", default_model)
        base_url = os.getenv(f"{prefix}_BASE_URL", default_url)
        try:
            max_tokens = int(
                os.getenv(f"{prefix}_MAX_TOKENS", "1024" if name == "nvidia" else "384")
            )
        except ValueError as exc:
            raise pytest.UsageError(
                f"{prefix}_MAX_TOKENS must be from 1 to 4096"
            ) from exc
        if not 1 <= max_tokens <= 4096:
            raise pytest.UsageError(f"{prefix}_MAX_TOKENS must be from 1 to 4096")
        if not os.getenv(key_env) or not model:
            missing.append(name)
            continue
        configs.append(ProviderConfig(name, key_env, base_url, model, max_tokens))
    if missing and (explicit_filter or require_all):
        raise pytest.UsageError(
            "Live provider configuration is missing API key or model for: "
            + ", ".join(missing)
        )
    return configs, missing


_CONFIGS, _MISSING_PROVIDERS = _provider_configs()
_LIVE_SKIP_REASON = (
    f"Set {_OPT_IN}=1 to enable the live provider matrix"
    if os.getenv(_OPT_IN) != "1"
    else "No requested providers have both an API key and model configured"
)
_STRESS_SKIP_REASON = (
    f"Set {_STRESS_OPT_IN}=1 to enable live provider stress tests"
    if os.getenv(_STRESS_OPT_IN) != "1"
    else "No requested providers have both an API key and model configured"
)
_PROVIDER_CASES = _CONFIGS + [
    pytest.param(
        None,
        id=f"{provider}-not-configured",
        marks=pytest.mark.skip(reason="API key or model is not configured"),
    )
    for provider in _MISSING_PROVIDERS
]
if not _PROVIDER_CASES:
    _PROVIDER_CASES = [pytest.param(None, id="live-provider-not-configured")]


def _make_model(config: ProviderConfig):
    """Use first-party clients where available, compatible transport otherwise."""
    from msgflux.models import Model

    kwargs = {
        "api_mode": "chat_completions",
        "api_key_env": config.api_key_env,
        "max_tokens": config.max_tokens,
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url
    # Prefer provider implementations and adaptations whenever registered.
    registered = Model.providers().get("chat_completion", [])
    if config.name in registered:
        return Model.chat_completion(f"{config.name}/{config.model}", **kwargs)

    # Exercise an unregistered OpenAI-compatible endpoint through the public
    # OpenAI adapter without implying first-party provider support.
    from msgflux.models.providers.openai import OpenAIChatCompletion

    return OpenAIChatCompletion(model_id=config.model, **kwargs)


def record_lookup(value: str) -> str:
    """Small deterministic side effect used to verify portable tool calling."""
    return f"checkpoint-token:{value}"


@pytest.mark.parametrize(
    "provider",
    _PROVIDER_CASES,
    ids=lambda item: item.name if item is not None else "live-provider-not-configured",
)
@pytest.mark.skipif(os.getenv(_OPT_IN) != "1", reason=_LIVE_SKIP_REASON)
@pytest.mark.skipif(bool(not _CONFIGS), reason=_LIVE_SKIP_REASON)
@pytest.mark.asyncio
async def test_live_agent_tool_stream_checkpoint_and_thread_replay(
    provider, tmp_path, record_property
):
    """Exercise Agent tool execution, streamed events, and checkpoint reuse."""
    from msgflux.data.stores import SQLiteCheckpointStore
    from msgflux.nn.modules.agent import Agent
    from msgflux.nn.extensions import ToolTurnLimitExtension
    from msgflux.runtime.context import ExecutionScope
    from msgflux.runtime.events import EventType

    checkpoint_path = str(tmp_path / "checkpoint.sqlite")

    def make_agent(store):
        return Agent(
            name=f"live_{provider.name}_matrix",
            model=_make_model(provider),
            checkpoint_store=store,
            tools=[record_lookup],
            extensions=[ToolTurnLimitExtension(2, warn_remaining=0)],
            config={"stream": True},
            system_prompt=(
                "For the requested lookup, call record_lookup exactly once with the "
                "requested value. Then give a concise answer quoting its exact return value."
            ),
        )

    store = SQLiteCheckpointStore(checkpoint_path)
    agent = make_agent(store)
    started = perf_counter()
    scope = ExecutionScope(
        namespace=agent.name,
        thread_id=f"live-{provider.name}",
        run_id="tool-run",
    )

    async with asyncio.timeout(180):
        events = [
            event
            async for event in agent.stream_events(
                "Look up the value 'alpha' using the tool now.", scope=scope
            )
        ]

    assert any(event.type == EventType.MODEL_REQUEST for event in events)
    assert any(event.type == EventType.MODEL_RESPONSE for event in events)
    assert any(event.type == EventType.TOOL_START for event in events)
    assert any(event.type == EventType.TOOL_END for event in events)
    assert any(event.type == EventType.MESSAGE_DELTA for event in events)
    assert events[-1].type == EventType.RUN_END

    saved = store.load_state(agent.name, scope.thread_id, scope.run_id)
    assert saved is not None
    assert saved["status"] == "completed"
    transcript = saved["messages"]["items"]
    assert any("checkpoint-token:alpha" in str(item) for item in transcript), (
        "checkpoint should contain the executed tool result"
    )

    store.close()
    store = SQLiteCheckpointStore(checkpoint_path)
    agent = make_agent(store)
    # A fresh Agent and store instance hydrate the last persisted thread state.
    replay_scope = ExecutionScope(
        namespace=agent.name,
        thread_id=scope.thread_id,
        run_id="replay-run",
    )
    async with asyncio.timeout(180):
        replay_events = [
            event
            async for event in agent.stream_events(
                "What exact checkpoint token did the lookup return?", scope=replay_scope
            )
        ]
    replay_state = store.load_state(agent.name, scope.thread_id, replay_scope.run_id)
    assert replay_state is not None
    assert replay_state["status"] == "completed"
    assert any(event.type == EventType.MESSAGE_END for event in replay_events)
    assert any(
        "checkpoint-token:alpha" in str(item)
        for item in replay_state["messages"]["items"]
    )
    store.close()
    properties = {
        "provider": provider.name,
        "model": provider.model,
        "api_mode": "chat_completions",
        "base_latency_seconds": perf_counter() - started,
        "base_event_count": len(events) + len(replay_events),
        "base_event_data_bytes": sum(
            _event_data_size(event) for event in (*events, *replay_events)
        ),
        "base_checkpoint_bytes": Path(checkpoint_path).stat().st_size,
    }
    for name, value in properties.items():
        record_property(name, value)


def _walk_tool_result_references(value):
    """Extract reference objects without retaining or printing payload bytes."""
    from msgflux.runtime.tool_results import get_tool_result_reference

    if isinstance(value, dict):
        reference = get_tool_result_reference(value)
        if reference is not None:
            yield reference
        for child in value.values():
            yield from _walk_tool_result_references(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_tool_result_references(child)


def _event_data_size(event) -> int:
    """Estimate serialized event data size without recording event contents."""
    import json

    return len(
        json.dumps(
            event.data, ensure_ascii=False, default=str, separators=(",", ":")
        ).encode("utf-8")
    )


@pytest.mark.parametrize(
    "provider",
    _PROVIDER_CASES,
    ids=lambda item: item.name if item is not None else "live-provider-not-configured",
)
@pytest.mark.skipif(os.getenv(_STRESS_OPT_IN) != "1", reason=_STRESS_SKIP_REASON)
@pytest.mark.skipif(bool(not _CONFIGS), reason=_LIVE_SKIP_REASON)
@pytest.mark.asyncio
async def test_live_agent_stress_large_offload_bounded_events_sqlite(
    provider, tmp_path, record_property
):
    """Stress streamed events, large tool output offload, and SQLite checkpoints."""
    from msgflux.data.stores import SQLiteCheckpointStore
    from msgflux.nn.extensions import ToolOutputOffloadExtension, ToolTurnLimitExtension
    from msgflux.nn.modules.agent import Agent
    from msgflux.runtime.context import ExecutionScope
    from msgflux.runtime.events import EventType
    from msgflux.runtime.tool_results import LocalToolResultStore

    payload_size = 512 * 1024
    max_events = 512
    checkpoint_path = Path(tmp_path) / "stress-checkpoint.sqlite"
    result_root = Path(tmp_path) / "large-tool-results"
    result_store = LocalToolResultStore(result_root, max_result_bytes=payload_size * 2)
    checkpoint_store = SQLiteCheckpointStore(str(checkpoint_path))

    def large_report(topic: str) -> str:
        """Return a large synthetic report for offload stress verification."""
        if topic != "alpha":
            raise ValueError("Expected the requested topic 'alpha'")
        return "report-marker:" + ("x" * (payload_size - len("report-marker:")))

    agent = Agent(
        name=f"live_{provider.name}_stress",
        model=_make_model(provider),
        checkpoint_store=checkpoint_store,
        tools=[large_report],
        extensions=[ToolTurnLimitExtension(2, warn_remaining=0)],
        config={"stream": True},
        system_prompt=(
            "Call large_report exactly once with topic='alpha'. "
            "Do not reproduce its contents. "
            "After the tool succeeds, reply only: large report processed."
        ),
    )
    offload = ToolOutputOffloadExtension(
        result_store, max_inline_bytes=256, preview_bytes=32
    )
    agent.tool_library.register_extension(offload.name, offload)
    scope = ExecutionScope(
        namespace=agent.name,
        thread_id=f"stress-{provider.name}",
        run_id="large-output-run",
    )

    started = perf_counter()
    try:
        async with asyncio.timeout(240):
            events = [
                event
                async for event in agent.stream_events(
                    "Run the large report tool for topic 'alpha' now.",
                    scope=scope,
                    event_buffer_limit=max_events,
                )
            ]
        elapsed_seconds = perf_counter() - started
        assert events[-1].type == EventType.RUN_END
        assert any(event.type == EventType.TOOL_START for event in events)
        assert any(event.type == EventType.TOOL_END for event in events)
        assert any(event.type == EventType.MESSAGE_DELTA for event in events)

        state = checkpoint_store.load_state(agent.name, scope.thread_id, scope.run_id)
        assert state is not None
        assert state["status"] == "completed"
        found_references = _walk_tool_result_references(state)
        references = {reference.result_id: reference for reference in found_references}
        assert len(references) == 1
        reference = next(iter(references.values()))
        assert reference.size_bytes == payload_size
        result_store.verify(reference)

        # Checkpoint byte count is measured after close to include committed SQLite data.
        checkpoint_store.close()
        checkpoint_bytes = checkpoint_path.stat().st_size
        properties = {
            "provider": provider.name,
            "model": provider.model,
            "api_mode": "chat_completions",
            "stress_latency_seconds": elapsed_seconds,
            "stress_event_count": len(events),
            "stress_event_data_bytes": sum(_event_data_size(event) for event in events),
            "stress_event_buffer_limit": max_events,
            "stress_checkpoint_bytes": checkpoint_bytes,
            "stress_offloaded_reference_count": len(references),
            "stress_offloaded_result_bytes": sum(
                ref.size_bytes for ref in references.values()
            ),
        }
        for name, value in properties.items():
            record_property(name, value)
    finally:
        checkpoint_store.close()


def test_model_factory_prefers_registered_provider(monkeypatch):
    from msgflux.models import Model

    selected = object()
    captured = {}

    monkeypatch.setattr(
        Model,
        "providers",
        classmethod(lambda cls: {"chat_completion": ["baseten"]}),
    )

    def create(path, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        return selected

    monkeypatch.setattr(
        Model,
        "chat_completion",
        classmethod(lambda cls, path, **kwargs: create(path, **kwargs)),
    )
    config = ProviderConfig(
        "baseten",
        "BASETEN_API_KEY",
        "https://inference.baseten.co/v1",
        "zai-org/GLM-5.2",
    )

    assert _make_model(config) is selected
    assert captured["path"] == "baseten/zai-org/GLM-5.2"
    assert captured["kwargs"]["api_key_env"] == "BASETEN_API_KEY"
    assert captured["kwargs"]["base_url"] == "https://inference.baseten.co/v1"
    assert captured["kwargs"]["api_mode"] == "chat_completions"


def test_model_factory_uses_openai_wire_adapter_when_provider_is_unregistered(
    monkeypatch,
):
    from msgflux.models import Model
    from msgflux.models.providers.openai import OpenAIChatCompletion

    monkeypatch.setattr(
        Model,
        "providers",
        classmethod(lambda cls: {"chat_completion": ["openai", "groq", "openrouter"]}),
    )
    config = ProviderConfig(
        "fireworks",
        "FIREWORKS_API_KEY",
        "https://api.fireworks.ai/inference/v1",
        "accounts/fireworks/models/gpt-oss-120b",
    )

    model = _make_model(config)

    assert isinstance(model, OpenAIChatCompletion)
    assert model.model_id == config.model
    assert model.sampling_params["base_url"] == config.base_url
    assert model.api_key_env == config.api_key_env


def test_provider_matrix_reports_unconfigured_defaults(monkeypatch):
    monkeypatch.setenv(_OPT_IN, "1")
    monkeypatch.delenv(_PROVIDER_FILTER, raising=False)
    monkeypatch.delenv(_REQUIRE_ALL, raising=False)
    for name, (key_env, _, _) in _PROVIDER_DEFAULTS.items():
        monkeypatch.delenv(key_env, raising=False)
        monkeypatch.delenv(f"MSGFLUX_LIVE_{name.upper()}_MODEL", raising=False)

    configs, missing = _provider_configs()

    assert configs == []
    assert missing == sorted(_PROVIDER_DEFAULTS)


def test_explicitly_selected_provider_without_credentials_fails(monkeypatch):
    monkeypatch.setenv(_OPT_IN, "1")
    monkeypatch.setenv(_PROVIDER_FILTER, "openrouter")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(pytest.UsageError, match="openrouter"):
        _provider_configs()


def test_require_all_rejects_partial_provider_configuration(monkeypatch):
    monkeypatch.setenv(_OPT_IN, "1")
    monkeypatch.setenv(_REQUIRE_ALL, "1")
    for name, (key_env, _, _) in _PROVIDER_DEFAULTS.items():
        monkeypatch.delenv(key_env, raising=False)

    with pytest.raises(pytest.UsageError, match=r"openai.*openrouter"):
        _provider_configs()
