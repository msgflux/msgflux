"""Manifest loading and local daemon source identity contracts."""

import asyncio
import hashlib
from pathlib import Path

import pytest
import pytest_asyncio

from msgflux.runtime.service import ServiceConflictError
from msgflux.runtime.service.local.discovery import (
    connect_local_service,
    restart_local_service,
)
from msgflux.runtime.service.local.files import process_lock, read_record
from msgflux.runtime.service.local.manifest import (
    initialize_service_config,
    load_service_manifest,
)
from msgflux.runtime.service.http.client import AgentServiceClient


@pytest_asyncio.fixture
async def clean_manifest_runtime(tmp_path):
    runtime = tmp_path / "runtime"
    yield runtime
    record = read_record(runtime) if runtime.exists() else None
    if record is not None:
        client = AgentServiceClient(record.url, token=record.token, timeout=2)
        try:
            try:
                await client.shutdown(expected_instance_id=record.instance_id)
            except Exception:
                # A previous assertion may have accepted shutdown while the
                # daemon is still releasing its listener and lifetime lock.
                for _ in range(200):
                    with process_lock(runtime / "daemon.lock") as lock:
                        if lock is not None:
                            break
                    await asyncio.sleep(0.025)
                else:
                    raise RuntimeError("manifest test daemon did not stop")
        finally:
            await client.aclose()
        for _ in range(200):
            if read_record(runtime) is None:
                break
            await asyncio.sleep(0.025)


def test_manifest_parses_factory_refs(tmp_path):
    path = tmp_path / "config.toml"
    content = (
        "[defaults]\nagent = 'worker'\n\n"
        "[agents.worker]\n"
        "factory = 'tests.test_local_service_runner:local_factory'\n"
    )
    path.write_text(content)

    manifest = load_service_manifest(path)

    assert manifest.path == path.resolve()
    assert manifest.digest == hashlib.sha256(content.encode()).hexdigest()
    assert tuple(manifest.agents) == ("worker",)
    assert (
        manifest.agents["worker"].factory
        == "tests.test_local_service_runner:local_factory"
    )
    assert manifest.defaults.agent_dir is None


def test_manifest_expands_agent_dir_without_touching_home(tmp_path, monkeypatch):
    fake_home = tmp_path / "fake-home"
    monkeypatch.setenv("HOME", str(fake_home))
    path = tmp_path / "config.toml"
    path.write_text(
        "[defaults]\nagent = 'worker'\nagent_dir = '~/agent-state/../state'\n"
        "[agents.worker]\nfactory = 'tests.test_local_service_runner:local_factory'\n"
    )

    manifest = load_service_manifest(path)

    assert manifest.defaults.agent_dir == str((fake_home / "state").resolve())


def test_manifest_normalizes_optional_agent_dir_for_custom_factories(tmp_path):
    path = tmp_path / "nested" / "config.toml"
    path.parent.mkdir()
    path.write_text(
        "[defaults]\nagent = 'worker'\nagent_dir = '../agent-state'\n"
        "[agents.worker]\nfactory = 'tests.test_local_service_runner:local_factory'\n"
    )

    manifest = load_service_manifest(path)

    assert manifest.defaults.agent_dir == str((tmp_path / "agent-state").resolve())


def test_manifest_parses_coding_agent_permissions(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        "[defaults]\nagent='explorer'\nmodel='provider/model-id'\n"
        "[agents.explorer]\nharness='coding'\npermissions=['read']\n"
    )

    manifest = load_service_manifest(path)

    assert manifest.agents["explorer"].permissions == ["read"]
    assert manifest.coding_catalog.agents["explorer"].permissions == ("read",)


@pytest.mark.parametrize(
    "content",
    [
        "[defaults]\nagent = 'worker'\nunknown = true\n",
        "[defaults]\nagent = 'worker'\nagent_dir = '   '\n[agents.worker]\n"
        "factory = 'tests.test_local_service_runner:local_factory'\n",
        "[defaults]\nagent = 'worker'\n[agents.worker]\nfactory = ''\n",
        "[defaults]\nagent = 'worker'\n[agents.worker]\nfactory = 'not-a-path'\n",
        "[defaults]\nagent = 'worker'\n[agents.\"\"]\nfactory = 'module:factory'\n",
        "[defaults]\nagent = 'worker'\n[agents]\n",
        "[defaults]\nagent = 'worker'\n[agents.worker]\nfactory = 'module:factory'\n\n"
        "[agents.worker]\nfactory = 'module:other'\n",
        "[defaults]\nagent = 'worker'\n[agents]\nworker = 'tests.test_local_service_runner:local_factory'\n",
    ],
)
def test_manifest_rejects_invalid_schema_or_factories(tmp_path, content):
    path = tmp_path / "config.toml"
    path.write_text(content)
    with pytest.raises((ValueError, TypeError)):
        load_service_manifest(path)


def test_runner_imports_factories_before_opening_service_store(tmp_path, monkeypatch):
    from msgflux.runtime.service.local import runner

    path = tmp_path / "config.toml"
    path.write_text(
        "[defaults]\nagent = 'worker'\n[agents.worker]\nfactory = 'missing.module:factory'\n"
    )

    def unexpected_store(*args, **kwargs):
        raise AssertionError("store must not be opened before factory imports")

    monkeypatch.setattr(runner, "SQLiteServiceStore", unexpected_store)
    with pytest.raises(ValueError, match="Invalid agent factory"):
        runner._make_manifest_service(path, tmp_path / "runtime", tmp_path)


@pytest.mark.asyncio
async def test_runner_rejects_custom_coding_runtime_before_creating_it(
    tmp_path, monkeypatch
):
    from msgflux.runtime.service.local.runner import serve_local_service

    parent = tmp_path / "shared-state"
    parent.mkdir(mode=0o755)
    runtime = parent / "runtime"
    config = tmp_path / "config.toml"
    config.write_text(
        "[defaults]\nagent='coding'\nmodel='provider/model'\n"
        "[agents.coding]\nharness='coding'\n"
    )
    monkeypatch.setenv("HOME", str(tmp_path / "fake-home"))

    with pytest.raises(ValueError, match="requires an explicit agent_dir") as caught:
        await serve_local_service(config=config, runtime_dir=runtime, cwd=Path.cwd())

    assert str(parent / "threads") in str(caught.value)
    assert parent.stat().st_mode & 0o777 == 0o755
    assert not runtime.exists()
    assert not (parent / "threads").exists()
    with pytest.raises(ValueError, match="requires an explicit agent_dir"):
        await restart_local_service(config=config, runtime_dir=runtime, cwd=tmp_path)
    assert not runtime.exists()


@pytest.mark.asyncio
async def test_discovery_rejects_custom_coding_runtime_before_preparing_it(
    tmp_path, monkeypatch
):
    parent = tmp_path / "shared-state"
    parent.mkdir(mode=0o755)
    parent.chmod(0o755)
    runtime = parent / "runtime"
    config = tmp_path / "config.toml"
    config.write_text(
        "[defaults]\nagent='coding'\nmodel='provider/model'\n"
        "[agents.coding]\nharness='coding'\n"
    )
    monkeypatch.setenv("HOME", str(tmp_path / "fake-home"))

    with pytest.raises(ValueError, match="requires an explicit agent_dir") as caught:
        await connect_local_service(config=config, runtime_dir=runtime, cwd=tmp_path)

    assert str(parent / "threads") in str(caught.value)
    assert parent.stat().st_mode & 0o777 == 0o755
    assert not runtime.exists()
    assert not (parent / "threads").exists()


def test_manifest_runner_passes_declared_default_agent(tmp_path):
    from msgflux.runtime.service.local.runner import _make_manifest_service

    path = tmp_path / "config.toml"
    path.write_text(
        "[defaults]\nagent = 'worker'\n[agents.worker]\n"
        "factory = 'tests.test_local_service_runner:local_factory'\n"
    )
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()

    service, config_path, _ = _make_manifest_service(path, runtime_dir, tmp_path)
    try:
        assert service.default_agent == "worker"
        assert config_path == str(path.resolve())
    finally:
        service.store.close()


@pytest.mark.parametrize(
    "body",
    [
        "[defaults]\nagent='missing'\n[agents.worker]\nfactory='m:f'\n",
        "[defaults]\nagent='worker'\n[agents]\n",
        "[defaults]\nagent='worker'\n[agents.worker]\nharness='other'\n",
        "[defaults]\nagent='worker'\n[agents.worker]\nharness='coding'\n"
        "subagents=['missing']\n",
        "[defaults]\nagent='worker'\n[agents.worker]\nharness='coding'\n"
        "[agents.worker.tools]\nactive=['agent']\n",
        "[defaults]\nagent='worker'\n[agents.worker]\nharness='coding'\n"
        "subagents=['child']\n[agents.child]\nharness='coding'\n"
        "subagents=['worker']\n",
        "[defaults]\nagent='worker'\n[agents.worker]\nharness='coding'\n"
        "[agents.worker.tools]\ndeferred=['agent']\n",
        "[defaults]\nagent='worker'\n[agents.worker]\n"
        "factory='module:factory'\npermissions=['read']\n",
    ],
)
def test_manifest_rejects_invalid_agent_configuration(tmp_path, body):
    path = tmp_path / "config.toml"
    path.write_text(body)
    with pytest.raises((ValueError, TypeError)):
        load_service_manifest(path)


def test_manifest_rejects_prompt_file_that_is_not_utf8(tmp_path):
    prompt = tmp_path / "prompt.md"
    prompt.write_bytes(b"\xff")
    config = tmp_path / "config.toml"
    config.write_text(
        "[defaults]\nagent='worker'\nmodel='openai-codex/gpt-6-luna'\n"
        "[agents.worker]\nharness='coding'\nprompt_file='prompt.md'\n"
    )
    with pytest.raises(ValueError, match="UTF-8"):
        load_service_manifest(config)


def test_initialize_config_is_secure_idempotent_and_never_overwrites(tmp_path):
    config = tmp_path / ".msgflux" / "config.toml"
    assert initialize_service_config(config) == config.resolve()
    assert config.stat().st_mode & 0o777 == 0o600
    template = config.read_text()
    assert 'model = "openai' not in template
    assert '# model = "provider/model-id"' in template
    assert "[agents.coding.tools]" not in template
    assert 'active = ["workspace", "agent"]' not in template
    assert '# permissions = ["read"]' in template
    config.write_text("existing user config\n")
    assert initialize_service_config(config) == config.resolve()
    assert config.read_text() == "existing user config\n"
    assert not list(config.parent.glob(".config-*"))


def test_initialize_config_can_set_default_model_and_preserves_existing_file(tmp_path):
    config = tmp_path / "config.toml"
    initialize_service_config(config, model='provider/model"quoted')
    assert 'model = "provider/model\\"quoted"' in config.read_text()
    config.write_text("user config\n")
    assert initialize_service_config(config, model="other/model") == config.resolve()
    assert config.read_text() == "user config\n"


@pytest.mark.parametrize("model", ["bad-model", "provider/", "provider/a b"])
def test_initialize_config_rejects_invalid_model_without_creating_files(
    tmp_path, model
):
    config = tmp_path / "new" / "config.toml"
    with pytest.raises(ValueError, match="Invalid --model"):
        initialize_service_config(config, model=model)
    assert not config.parent.exists()


def test_initialize_config_rejects_dangling_symlink_without_changing_it(tmp_path):
    config = tmp_path / "config.toml"
    target = tmp_path / "missing.toml"
    config.symlink_to(target)

    with pytest.raises(FileNotFoundError, match="dangling symlink"):
        initialize_service_config(config)

    assert config.is_symlink()
    assert config.readlink() == target
    assert not target.exists()


def test_initialize_config_race_preserves_published_config(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    config = tmp_path / "config.toml"

    def initialize():
        return initialize_service_config(config)

    with ThreadPoolExecutor(max_workers=8) as executor:
        published_paths = list(executor.map(lambda _: initialize(), range(16)))
    assert published_paths == [config.resolve()] * 16
    assert config.read_text().startswith('[defaults]\nagent = "coding"\n')
    assert not list(tmp_path.glob(".config-*"))


def test_runner_rejects_expanded_tool_overlap_before_opening_store(
    tmp_path, monkeypatch
):
    from msgflux.runtime.service.local import runner

    path = tmp_path / "config.toml"
    path.write_text(
        "[defaults]\nagent='worker'\nmodel='openai-codex/gpt-6-luna'\n"
        "agent_dir='agent-state'\n"
        "[agents.worker]\nharness='coding'\n"
        "[agents.worker.tools]\nactive=['workspace']\ndeferred=['bash']\n"
    )

    def unexpected_store(*args, **kwargs):
        raise AssertionError("store must not open for invalid expanded tools")

    monkeypatch.setattr(runner, "SQLiteServiceStore", unexpected_store)
    with pytest.raises(ValueError, match="both active and deferred"):
        runner._make_manifest_service(path, tmp_path / "runtime", tmp_path)


@pytest.mark.asyncio
async def test_manifest_sessions_are_lazy_isolated_and_survive_restart(
    tmp_path, clean_manifest_runtime
):
    module = tmp_path / "manifest_factory.py"
    module.write_text(
        """
import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.nn import Agent
from msgflux.runtime.service import AgentSession

def make_session(thread):
    marker = Path(__file__).with_name('factory_calls.log')
    with marker.open('a') as stream:
        stream.write(thread.thread_id + '\\n')
    db = hashlib.sha256(thread.thread_id.encode()).hexdigest() + '.sqlite3'
    checkpoints = SQLiteCheckpointStore(str(Path(__file__).with_name(db)))
    model = Mock()
    model.model_type = 'chat_completion'
    agent = Agent(name='manifest-worker', model=model, checkpoint_store=checkpoints)
    async def respond(**kwargs):
        prompt = kwargs['messages'].to_chatml()[-1]['content']
        response = ModelResponse()
        response.set_response_type('text_generation')
        response.add('manifest reply: ' + prompt)
        return response
    agent.generator.aforward = AsyncMock(side_effect=respond)
    return AgentSession(agent, on_close=checkpoints.close)
"""
    )
    config = tmp_path / "service.toml"
    config.write_text(
        "[defaults]\nagent = 'worker'\n[agents.worker]\nfactory = 'manifest_factory:make_session'\n"
    )
    runtime = clean_manifest_runtime
    client = await connect_local_service(
        config=config, runtime_dir=runtime, cwd=tmp_path, startup_timeout=15
    )
    try:
        marker = tmp_path / "factory_calls.log"
        assert not marker.exists()
        first = await client.open_thread("worker", thread_id="manifest-a")
        second = await client.open_thread("worker", thread_id="manifest-b")
        assert not marker.exists()
        async with client.watch(first.thread_id) as watcher:
            receipt = await client.prompt(
                first.thread_id, "alpha-marker", request_id="prompt-a"
            )
            assert (await asyncio.wait_for(anext(watcher), 3)).type == "run.start"
        assert receipt.status in {"accepted", "running", "completed"}
        receipt = await _wait_for_manifest_completion(
            client, first.thread_id, "prompt-a"
        )
        assert receipt.status == "completed"
        assert set(marker.read_text().splitlines()) == {first.thread_id}
        await client.prompt(second.thread_id, "beta-marker", request_id="prompt-b")
        assert (
            await _wait_for_manifest_completion(client, second.thread_id, "prompt-b")
        ).status == "completed"
        assert set(marker.read_text().splitlines()) == {
            first.thread_id,
            second.thread_id,
        }
        assert "alpha-marker" in str((await client.snapshot(first.thread_id)).messages)
        assert "beta-marker" not in str(
            (await client.snapshot(first.thread_id)).messages
        )
        assert "beta-marker" in str((await client.snapshot(second.thread_id)).messages)
    finally:
        await client.aclose()

    config.write_text(
        "[defaults]\nagent = 'worker'\n[agents.worker]\nfactory = 'missing.module:make_session'\n"
    )
    with pytest.raises(RuntimeError, match="exited during startup"):
        await restart_local_service(runtime_dir=runtime, restart_timeout=15)
    assert read_record(runtime) is None
    with process_lock(runtime / "daemon.lock") as lock:
        assert lock is not None

    config.write_text(
        "[defaults]\nagent = 'worker'\n[agents.worker]\nfactory = 'manifest_factory:make_session'\n"
    )
    recovered = await connect_local_service(
        config=config, runtime_dir=runtime, cwd=tmp_path, startup_timeout=15
    )
    try:
        assert "alpha-marker" in str((await recovered.snapshot("manifest-a")).messages)
        assert "beta-marker" not in str(
            (await recovered.snapshot("manifest-a")).messages
        )
        assert (
            await recovered.receipt("manifest-a", "prompt-a")
        ).run_id == receipt.run_id
        await recovered.prompt("manifest-a", "continued-a", request_id="prompt-c")
        assert (
            await _wait_for_manifest_completion(recovered, "manifest-a", "prompt-c")
        ).status == "completed"
        assert "continued-a" in str((await recovered.snapshot("manifest-a")).messages)
    finally:
        await recovered.aclose()


async def _wait_for_manifest_completion(client, thread_id, request_id):
    async def wait():
        while True:
            receipt = await client.receipt(thread_id, request_id)
            if receipt.status not in {"accepted", "running"}:
                return receipt
            await asyncio.sleep(0.02)

    return await asyncio.wait_for(wait(), 8)


@pytest.mark.asyncio
async def test_manifest_daemon_connect_identity_and_restart(
    tmp_path, clean_manifest_runtime
):
    module = tmp_path / "manifest_factory.py"
    module.write_text("def make_session(thread):\n    return None\n")
    config = tmp_path / "config.toml"
    config.write_text(
        "[defaults]\nagent = 'worker'\n[agents.worker]\nfactory = 'manifest_factory:make_session'\n"
    )
    runtime_dir = clean_manifest_runtime

    client = await connect_local_service(
        config=config, runtime_dir=runtime_dir, cwd=tmp_path, startup_timeout=15
    )
    try:
        initial_record = read_record(runtime_dir)
        assert initial_record is not None
        assert initial_record.factory is None
        assert initial_record.config_path == str(config.resolve())
        assert (
            initial_record.config_digest
            == hashlib.sha256(config.read_bytes()).hexdigest()
        )
        assert await client.agents() == ("worker",)
        with pytest.raises(ServiceConflictError):
            await connect_local_service(
                "manifest_factory:make_session",
                runtime_dir=runtime_dir,
                cwd=tmp_path,
                startup_timeout=3,
            )
        with pytest.raises(ServiceConflictError):
            await restart_local_service(
                "manifest_factory:make_session",
                runtime_dir=runtime_dir,
                cwd=tmp_path,
                restart_timeout=3,
            )

        other_config = tmp_path / "other.toml"
        other_config.write_text(config.read_text())
        with pytest.raises(ServiceConflictError):
            await connect_local_service(
                config=other_config,
                runtime_dir=runtime_dir,
                cwd=tmp_path,
                startup_timeout=3,
            )
        with pytest.raises(ServiceConflictError):
            await restart_local_service(
                config=other_config,
                runtime_dir=runtime_dir,
                cwd=tmp_path,
                restart_timeout=3,
            )

        config.write_text(config.read_text() + "\n# revision 2\n")
        with pytest.raises(ServiceConflictError):
            await connect_local_service(
                config=config, runtime_dir=runtime_dir, cwd=tmp_path, startup_timeout=3
            )
    finally:
        await client.aclose()

    replacement = await restart_local_service(
        runtime_dir=runtime_dir, restart_timeout=15
    )
    try:
        new_record = read_record(runtime_dir)
        assert new_record is not None
        assert new_record.instance_id != initial_record.instance_id
        assert new_record.config_path == initial_record.config_path
        assert (
            new_record.config_digest == hashlib.sha256(config.read_bytes()).hexdigest()
        )
        assert await replacement.agents() == ("worker",)
    finally:
        try:
            await replacement.shutdown(expected_instance_id=new_record.instance_id)
        finally:
            await replacement.aclose()


@pytest.mark.asyncio
async def test_connect_defaults_to_home_manifest(
    tmp_path, monkeypatch, clean_manifest_runtime
):
    module = tmp_path / "default_factory.py"
    module.write_text("def make_session(thread):\n    return None\n")
    config_dir = tmp_path / ".msgflux"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        "[defaults]\nagent = 'worker'\n[agents.worker]\nfactory = 'default_factory:make_session'\n"
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    client = await connect_local_service(
        runtime_dir=clean_manifest_runtime, cwd=tmp_path, startup_timeout=15
    )
    try:
        assert await client.agents() == ("worker",)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_connect_requires_model_from_initialized_default_manifest(
    tmp_path, monkeypatch, clean_manifest_runtime
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    config = tmp_path / ".msgflux" / "config.toml"
    with pytest.raises(ValueError, match=r"defaults\.model or agents\.coding\.model"):
        await connect_local_service(
            runtime_dir=clean_manifest_runtime, cwd=tmp_path, startup_timeout=15
        )
    assert config.exists()
    assert read_record(clean_manifest_runtime) is None
    config.write_text(
        config.read_text().replace(
            'agent = "coding"\n',
            'agent = "coding"\nmodel = "provider/model-id"\nagent_dir = "."\n',
            1,
        )
    )

    manifest = load_service_manifest(config)
    assert manifest.defaults.agent == "coding"
    assert manifest.agents["coding"].subagents == ["lite-coding"]
    client = await connect_local_service(
        runtime_dir=clean_manifest_runtime, cwd=tmp_path, startup_timeout=15
    )
    try:
        assert await client.agents() == ("coding", "lite-coding")
        record = read_record(clean_manifest_runtime)
        assert record is not None
        assert record.config_path == str(config.resolve())
    finally:
        await client.aclose()


@pytest.mark.parametrize(
    "option",
    [
        "permissions=['unknown']",
        "permissions=['read','read']",
        "permissions='read'",
        "read_only=true",
    ],
)
def test_manifest_rejects_invalid_or_obsolete_permissions(tmp_path, option):
    path = tmp_path / "config.toml"
    path.write_text(
        "[defaults]\nagent='coding'\nmodel='provider/model-id'\n"
        f"[agents.coding]\nharness='coding'\n{option}\n"
    )
    with pytest.raises(ValueError):
        load_service_manifest(path)


@pytest.mark.parametrize("option", ["", "permissions=[]"])
def test_manifest_distinguishes_default_and_empty_permissions(tmp_path, option):
    path = tmp_path / "config.toml"
    path.write_text(
        "[defaults]\nagent='coding'\nmodel='provider/model-id'\n"
        f"[agents.coding]\nharness='coding'\n{option}\n"
    )
    options = load_service_manifest(path).coding_catalog.agents["coding"]
    assert options.permissions == (() if option else ("read", "write", "execute"))
