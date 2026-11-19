"""Offline text search and compilation for msgFlux modules."""

import asyncio
import hashlib
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from contextlib import AsyncExitStack, ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal

import msgspec

from msgflux.core.examples import Example
from msgflux.nn import Agent, Module

from .evaluation import Score, normalize_score, predict
from .tool_target import get_tool_description, set_tool_description


@dataclass(frozen=True)
class TextResult:
    """Selected text and mean validation scores, including the initial candidate."""

    best_text: str
    baseline_score: float
    best_score: float
    metric_calls: int


@dataclass(frozen=True)
class ProgramResult:
    """The jointly selected instructions and their root program score."""

    best_texts: dict[str, str]
    baseline_score: float
    best_score: float
    metric_calls: int


class TextTarget(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Address one Agent instruction or local tool in a registered module tree."""

    module: str = ""
    tool: str | None = None

    def _agent(self, program: Module) -> Agent:
        agent = program.get_submodule(self.module)
        if not isinstance(agent, Agent):
            raise TypeError(f"Target module {self.module!r} must be an Agent")
        return agent

    def read(self, program: Module) -> str:
        agent = self._agent(program)
        if self.tool is None:
            return agent.get_task_instruction()
        return get_tool_description(agent, self.tool)

    def write(self, program: Module, text: str) -> None:
        agent = self._agent(program)
        if self.tool is None:
            agent.set_task_instruction(text)
        else:
            set_tool_description(agent, self.tool, text)


class OptimizationArtifact(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Versioned text update that can be applied without an optimization backend."""

    version: Literal[1]
    target: TextTarget
    source_sha256: str
    text: str

    def apply(self, program: Module) -> None:
        """Apply to a compatible program, checking the original target first."""
        source = self.target.read(program)
        if source == self.text:
            return
        if hashlib.sha256(source.encode()).hexdigest() != self.source_sha256:
            raise ValueError("Optimization target has changed since compilation")
        self.target.write(program, self.text)

    def save(self, path: str | Path) -> None:
        _save_artifact(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "OptimizationArtifact | ProgramArtifact":
        data = msgspec.json.decode(Path(path).read_bytes())
        schema = (
            ProgramArtifact
            if isinstance(data, dict) and data.get("version") == 2
            else cls
        )
        return msgspec.convert(data, type=schema)


class ProgramArtifact(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """One jointly selected combination of portable text updates."""

    version: Literal[2]
    updates: tuple[OptimizationArtifact, ...]

    def apply(self, program: Module) -> None:
        if not self.updates:
            raise ValueError("A program artifact must contain at least one target")
        _check_distinct(program, [update.target for update in self.updates])
        applied = []
        try:
            for update in self.updates:
                previous = update.target.read(program)
                update.apply(program)
                if previous != update.text:
                    applied.append((update.target, previous))
        except Exception:
            for target, previous in reversed(applied):
                target.write(program, previous)
            raise

    def save(self, path: str | Path) -> None:
        _save_artifact(self, path)


def _save_artifact(artifact, path):
    destination = Path(path)
    data = msgspec.json.format(msgspec.json.encode(artifact))
    file = NamedTemporaryFile(dir=destination.parent, delete=False)
    temporary = Path(file.name)
    try:
        with file:
            file.write(data)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _check_distinct(program, targets):
    locations = set()
    for target in targets:
        agent = target._agent(program)
        owner = (
            agent.system_prompt
            if target.tool is None
            else getattr(agent.tool_library, "registry", None)
        )
        key = (id(owner), target.tool)
        if key in locations:
            raise ValueError("Multiple targets refer to the same prompt or tool")
        locations.add(key)


@dataclass(frozen=True)
class CompilationResult:
    search: TextResult | ProgramResult
    artifact: OptimizationArtifact | ProgramArtifact
    _factory: Callable[[], Module]

    def build(self) -> Module:
        """Build a fresh program and apply the selected text."""
        program = self._factory()
        self.artifact.apply(program)
        return program

    def save(self, path: str | Path) -> None:
        self.artifact.save(path)


def _agents(program):
    pending = [("", program)]
    seen = set()
    while pending:
        path, module = pending.pop()
        if id(module) in seen:
            continue
        seen.add(id(module))
        if isinstance(module, Agent):
            yield path, module
        for name, child in reversed(module._modules.items()):
            if child is not None:
                pending.append((f"{path}.{name}" if path else name, child))


async def _close_agents(program: Module) -> None:
    async with AsyncExitStack() as stack:
        for _, agent in _agents(program):
            stack.push_async_callback(agent.aclose)


def _record_calls(calls):
    def record(_module, args, kwargs, output):
        calls.append({"args": args, "kwargs": kwargs, "output": output})

    return record


@contextmanager
def _trial(factory):
    program = factory()
    if not isinstance(program, Module):
        raise TypeError("The factory must return a new nn.Module")
    try:
        yield program
    finally:
        asyncio.run(_close_agents(program))


def _resolve_targets(program, target, targets):
    if target is None and targets is None and isinstance(program, Agent):
        target = TextTarget()
    if target is not None:
        components = {"instruction": target}
    elif targets is not None:
        components = dict(targets)
    else:
        components = {
            path: TextTarget(module=path)
            for path, agent in _agents(program)
            if agent.system_prompt.requires_grad
            and agent.get_task_instruction().strip()
        }
    if not components:
        raise ValueError("No trainable text targets in the program")
    if any(
        not isinstance(k, str) or not k.strip() or not isinstance(v, TextTarget)
        for k, v in components.items()
    ):
        raise ValueError("targets must map non-empty names to TextTarget instances")
    _check_distinct(program, components.values())
    return target, components


def _prepare_trial(trial, stack, components, seeds, texts, capture_calls):
    if texts.keys() != components.keys():
        raise ValueError("Candidate components differ from the program targets")
    _check_distinct(trial, components.values())
    traces = {}
    for name, item in components.items():
        if item.read(trial) != seeds[name]:
            raise ValueError("The factory produced a different source text")
        item.write(trial, texts[name])
        if capture_calls:
            calls = []
            handle = item._agent(trial).register_forward_hook(_record_calls(calls))
            stack.callback(handle.remove)
            traces[name] = calls
    return traces


def _reflection_context(example, prediction, score, feedback, traces):
    context = {
        "inputs": example.inputs,
        "prediction": prediction,
        "score": score,
        **feedback,
    }
    for name, calls in traces.items():
        key = f"{name}_specific_info"
        context[key] = {"calls": calls, **context.get(key, {})}
    return context


class Optimizer(ABC):
    """Implement _search to reuse dataset validation, compilation and persistence."""

    @abstractmethod
    def _search(self, seed, trainset, valset, evaluator) -> TextResult:
        """Search using evaluator(candidate, example) -> (finite score, feedback)."""

    def optimize_text(
        self,
        seed: str,
        *,
        trainset: Sequence,
        valset: Sequence,
        evaluator: Callable[..., Score],
    ) -> TextResult:
        if not isinstance(seed, str) or not seed.strip():
            raise ValueError("A non-empty text seed is required")
        if not trainset or not valset:
            raise ValueError("trainset and valset must both contain examples")

        def evaluate(candidate, example):
            return normalize_score(evaluator(candidate, example))

        return self._search(seed, list(trainset), list(valset), evaluate)

    def compile(
        self,
        factory: Callable[[], Module],
        *,
        trainset: Sequence[Example],
        valset: Sequence[Example],
        metric: Callable[[Example, Any], Score],
        target: TextTarget | None = None,
        targets: Mapping[str, TextTarget] | None = None,
        include_example_context: bool = True,
    ) -> CompilationResult:
        """Select instructions using a shared score on the entire program."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("Use asyncio.to_thread to compile from an async loop")
        if target is not None and targets is not None:
            raise ValueError("Specify either target or targets")
        if not trainset or not valset:
            raise ValueError("trainset and valset must both contain examples")
        with _trial(factory) as program:
            target, components = _resolve_targets(program, target, targets)
            seeds = {name: item.read(program) for name, item in components.items()}
            for name, item in components.items():
                item.write(program, seeds[name])

        def evaluate(candidate, example):
            texts = {"instruction": candidate} if target is not None else candidate
            with _trial(factory) as trial, ExitStack() as stack:
                traces = _prepare_trial(
                    trial,
                    stack,
                    components,
                    seeds,
                    texts,
                    target is None and include_example_context,
                )
                prediction = predict(trial, example)
                value, feedback = normalize_score(metric(example, prediction))
                if include_example_context:
                    feedback = _reflection_context(
                        example, prediction, value, feedback, traces
                    )
                return value, feedback

        if target is not None:
            result = self.optimize_text(
                seeds["instruction"],
                trainset=trainset,
                valset=valset,
                evaluator=evaluate,
            )
            best = {"instruction": result.best_text}
        else:
            result = self._search_program(seeds, list(trainset), list(valset), evaluate)
            best = result.best_texts
            if best.keys() != seeds.keys():
                raise ValueError("Optimizer returned incompatible program components")
        updates = tuple(
            OptimizationArtifact(
                1, item, hashlib.sha256(seeds[name].encode()).hexdigest(), best[name]
            )
            for name, item in components.items()
        )
        artifact = updates[0] if target is not None else ProgramArtifact(2, updates)
        return CompilationResult(result, artifact, factory)

    def _search_program(self, seeds, trainset, valset, evaluator) -> ProgramResult:
        raise NotImplementedError(
            f"{type(self).__name__} does not support joint program search"
        )
