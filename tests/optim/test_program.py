"""Joint program compilation: targets, artifacts and component feedback."""

import hashlib
import json

import msgspec
import pytest

from msgflux import Example, nn
from msgflux.models.response import ModelResponse
from msgflux.optim import (
    GEPA,
    OptimizationArtifact,
    Optimizer,
    ProgramArtifact,
    ProgramResult,
    TextResult,
    TextTarget,
)

ROUTE_SEED = "Route each ticket to a queue."
ROUTE_TUNED = "Route invoice tickets to the billing queue."
REPLY_SEED = "Answer the ticket."
REPLY_TUNED = "Answer with a strict reply."
AUDIT_SEED = "Audit the ticket."

TICKET = Example("Missing invoice", "queue-billing:tone-strict")
TARGETS = {
    "routing": TextTarget(module="router"),
    "response": TextTarget(module="reply"),
}


def text(value):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(value)
    return response


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _update(module, source, replacement):
    return OptimizationArtifact(
        1, TextTarget(module=module), digest(source), replacement
    )


class QueueModel:
    """Deterministic router: only a billing prompt routes invoices to billing."""

    model_type = "chat_completion"

    def __call__(self, *, system_prompt, **_kwargs):
        return text("queue-billing" if "billing" in system_prompt else "queue-general")


class ToneModel:
    """Deterministic writer: only a strict prompt produces a strict reply."""

    model_type = "chat_completion"

    def __call__(self, *, system_prompt, **_kwargs):
        return text("tone-strict" if "strict" in system_prompt else "tone-loose")


class AuditModel:
    """Stand-in for the frozen agent, which must never be a target."""

    model_type = "chat_completion"

    def __call__(self, **_kwargs):
        return text("audited")


class TicketProgram(nn.Module):
    """Root program: router, writer, a frozen auditor and a blank prompt."""

    def __init__(self):
        super().__init__()
        self.router = nn.Agent(
            name="router", model=QueueModel(), system_prompt=ROUTE_SEED
        )
        self.reply = nn.Agent(name="reply", model=ToneModel(), system_prompt=REPLY_SEED)
        self.auditor = nn.Agent(
            name="auditor", model=AuditModel(), system_prompt=AUDIT_SEED
        )
        self.auditor.system_prompt.requires_grad_(requires_grad=False)
        self.blank = nn.Agent(name="blank", model=AuditModel(), system_prompt="   ")

    def forward(self, ticket):
        self.auditor(ticket)
        return f"{self.router(ticket)}:{self.reply(ticket)}"


class RoutedProgram(nn.Module):
    """Root program that skips the writer unless the router selects billing."""

    def __init__(self):
        super().__init__()
        self.router = nn.Agent(
            name="router", model=QueueModel(), system_prompt=ROUTE_SEED
        )
        self.reply = nn.Agent(name="reply", model=ToneModel(), system_prompt=REPLY_SEED)

    def forward(self, ticket):
        queue = self.router(ticket)
        if queue != "queue-billing":
            return f"{queue}:skipped"
        return f"{queue}:{self.reply(ticket)}"


class NestedProgram(nn.Module):
    """Root program whose agents live inside a ModuleDict."""

    def __init__(self):
        super().__init__()
        self.agents = nn.ModuleDict(
            {
                "router": nn.Agent(
                    name="router", model=QueueModel(), system_prompt=ROUTE_SEED
                ),
                "reply": nn.Agent(
                    name="reply", model=ToneModel(), system_prompt=REPLY_SEED
                ),
                "auditor": nn.Agent(
                    name="auditor", model=AuditModel(), system_prompt=AUDIT_SEED
                ),
            }
        )
        self.agents["auditor"].system_prompt.requires_grad_(requires_grad=False)

    def forward(self, ticket):
        return f"{self.agents['router'](ticket)}:{self.agents['reply'](ticket)}"


def exact(example, prediction):
    """Binary program score: only one prompt being right is still a miss."""
    return float(prediction == example.labels), {
        "feedback": f"Expected {example.labels}, got {prediction}."
    }


class Reflector:
    """Deterministic reflection model that rewrites the prompt it is shown."""

    model_type = "chat_completion"

    def __init__(self, rewrites):
        self.rewrites = dict(rewrites)
        self.requests = []

    def __call__(self, *, messages):
        self.requests.append(messages[-1]["content"])
        sections = self.requests[-1].split("```")
        assert len(sections) >= 3, self.requests[-1]
        return text(self.rewrites.get(sections[1].strip(), sections[1].strip()))


class RecordingOptimizer(Optimizer):
    """Structural backend: records the joint seed mapping without searching."""

    def __init__(self):
        self.seeds = []
        self.feedback = []

    def _search(self, seed, trainset, valset, evaluator):
        raise AssertionError("A joint compile must use _search_program")

    def _search_program(self, seeds, trainset, valset, evaluator):
        self.seeds.append(dict(seeds))
        _, feedback = evaluator(dict(seeds), valset[0])
        self.feedback.append(feedback)
        return ProgramResult(dict(seeds), 0.0, 0.0, 1)


def test_auto_discovery_targets_every_trainable_instruction():
    optimizer = RecordingOptimizer()

    result = optimizer.compile(
        TicketProgram, trainset=[TICKET], valset=[TICKET], metric=exact
    )

    # Discovery skips the frozen prompt, the blank prompt and non-agent modules.
    assert optimizer.seeds == [{"router": ROUTE_SEED, "reply": REPLY_SEED}]
    assert isinstance(result.search, ProgramResult)
    assert result.search.best_texts == {"router": ROUTE_SEED, "reply": REPLY_SEED}

    # A joint compilation always persists one version 2 combination artifact.
    assert isinstance(result.artifact, ProgramArtifact)
    assert result.artifact.version == 2
    assert [update.target for update in result.artifact.updates] == [
        TextTarget(module="router"),
        TextTarget(module="reply"),
    ]
    assert [update.source_sha256 for update in result.artifact.updates] == [
        digest(ROUTE_SEED),
        digest(REPLY_SEED),
    ]

    built = result.build()
    assert built("Missing invoice") == "queue-general:tone-loose"
    assert built.auditor.get_task_instruction() == AUDIT_SEED
    assert built.blank.get_task_instruction() == "   "


def test_explicit_names_keep_the_requested_subset():
    optimizer = RecordingOptimizer()

    result = optimizer.compile(
        TicketProgram, targets=TARGETS, trainset=[TICKET], valset=[TICKET], metric=exact
    )

    assert optimizer.seeds == [{"routing": ROUTE_SEED, "response": REPLY_SEED}]
    # Names label the components; the artifact keeps addressing the modules.
    assert [update.target.module for update in result.artifact.updates] == [
        "router",
        "reply",
    ]

    subset = RecordingOptimizer()
    result = subset.compile(
        TicketProgram,
        targets={"routing": TextTarget(module="router")},
        trainset=[TICKET],
        valset=[TICKET],
        metric=exact,
    )
    assert subset.seeds == [{"routing": ROUTE_SEED}]
    assert isinstance(result.artifact, ProgramArtifact)
    assert result.build().reply.get_task_instruction() == REPLY_SEED


def test_frozen_agent_is_never_discovered_or_written():
    with pytest.raises(ValueError, match="frozen"):
        RecordingOptimizer().compile(
            TicketProgram,
            target=TextTarget(module="auditor"),
            trainset=[TICKET],
            valset=[TICKET],
            metric=exact,
        )

    artifact = _update("auditor", AUDIT_SEED, "Audit every ticket.")
    program = TicketProgram()
    with pytest.raises(ValueError, match="frozen"):
        artifact.apply(program)
    assert program.auditor.get_task_instruction() == AUDIT_SEED


def test_global_score_and_component_feedback_reach_the_evaluator():
    optimizer = RecordingOptimizer()

    def metric(example, prediction):
        assert prediction == "queue-general:skipped"
        return 0.0, {
            "feedback": "The invoice never reached billing.",
            "routing_specific_info": {"queue": "wrong"},
        }

    optimizer.compile(
        RoutedProgram,
        targets=TARGETS,
        trainset=[TICKET],
        valset=[TICKET],
        metric=metric,
    )

    (feedback,) = optimizer.feedback
    # The metric scores the root program's output; its feedback is forwarded.
    assert feedback["inputs"] == TICKET.inputs
    assert feedback["prediction"] == "queue-general:skipped"
    assert feedback["score"] == 0.0
    assert feedback["feedback"] == "The invoice never reached billing."

    # Each component gets its own traced calls plus its own diagnostics.
    assert feedback["routing_specific_info"] == {
        "calls": [{"args": (TICKET.inputs,), "kwargs": {}, "output": "queue-general"}],
        "queue": "wrong",
    }
    # The skipped branch is reported with an empty call list, not omitted.
    assert feedback["response_specific_info"] == {"calls": []}


def test_include_example_context_disables_the_traced_payload():
    optimizer = RecordingOptimizer()

    optimizer.compile(
        RoutedProgram,
        targets=TARGETS,
        trainset=[TICKET],
        valset=[TICKET],
        metric=lambda *_: (0.0, {"feedback": "unused"}),
        include_example_context=False,
    )

    assert optimizer.feedback == [{"feedback": "unused"}]


def test_component_feedback_reaches_the_reflector_per_component():
    pytest.importorskip("gepa")

    def metric(example, prediction):
        return float(prediction == example.labels), {
            "shared_note": "SHARED-NOTE",
            "routing_specific_info": {"feedback": "ROUTING-NOTE"},
            "response_specific_info": {"feedback": "RESPONSE-NOTE"},
        }

    # RoutedProgram skips the writer while the router says "queue-general".
    reflector = Reflector({ROUTE_SEED: ROUTE_TUNED, REPLY_SEED: REPLY_TUNED})
    GEPA(reflector, 12).compile(
        RoutedProgram,
        targets=TARGETS,
        trainset=[TICKET],
        valset=[TICKET],
        metric=metric,
    )

    routing = [p for p in reflector.requests if ROUTE_SEED in p]
    response = [p for p in reflector.requests if REPLY_SEED in p]
    assert routing and response
    for prompt in routing:
        assert "ROUTING-NOTE" in prompt
        assert "RESPONSE-NOTE" not in prompt
        assert "SHARED-NOTE" in prompt
        # The router's own traced call is the one rendered for this component.
        assert "queue-general" in prompt
    for prompt in response:
        assert "RESPONSE-NOTE" in prompt
        assert "ROUTING-NOTE" not in prompt
        assert "SHARED-NOTE" in prompt
        # The skipped branch contributes no traced output to this component.
        assert "#### output" not in prompt


def test_only_a_simultaneous_change_improves_the_binary_metric():
    pytest.importorskip("gepa")
    rewrites = {ROUTE_SEED: ROUTE_TUNED, REPLY_SEED: REPLY_TUNED}
    compile_kwargs = {"trainset": [TICKET], "valset": [TICKET], "metric": exact}

    # One proposal per iteration: neither edit alone reaches the exact match.
    sequential = Reflector(rewrites)
    one_at_a_time = GEPA(sequential, 16).compile(
        TicketProgram, targets=TARGETS, **compile_kwargs
    )
    assert len(sequential.requests) > len(TARGETS)
    assert one_at_a_time.search.baseline_score == 0.0
    assert one_at_a_time.search.best_score == 0.0
    assert one_at_a_time.search.best_texts == {
        "routing": ROUTE_SEED,
        "response": REPLY_SEED,
    }
    assert one_at_a_time.build()("Missing invoice") == "queue-general:tone-loose"

    # module_selector="all" proposes both edits for the same candidate.
    joint = Reflector(rewrites)
    together = GEPA(joint, 16, module_selector="all").compile(
        TicketProgram, targets=TARGETS, **compile_kwargs
    )
    assert together.search.baseline_score == 0.0
    assert together.search.best_score == 1.0
    assert together.search.metric_calls <= 16
    assert together.search.best_texts == {
        "routing": ROUTE_TUNED,
        "response": REPLY_TUNED,
    }
    assert together.build()("Missing invoice") == "queue-billing:tone-strict"
    assert together.artifact.version == 2
    assert [update.text for update in together.artifact.updates] == [
        ROUTE_TUNED,
        REPLY_TUNED,
    ]


def test_module_selector_accepts_only_the_documented_values():
    assert GEPA(str, 4).module_selector == "round_robin"
    assert GEPA(str, 4, module_selector="all").module_selector == "all"
    with pytest.raises(ValueError, match="module_selector"):
        GEPA(str, 4, module_selector="ALL")


def test_moduledict_paths_are_addressable_and_discovered():
    program = NestedProgram()
    target = TextTarget(module="agents.router")

    assert program.get_submodule("agents.router") is program.agents["router"]
    assert target.read(program) == ROUTE_SEED
    target.write(program, ROUTE_TUNED)
    assert program.agents["router"].get_task_instruction() == ROUTE_TUNED

    optimizer = RecordingOptimizer()
    result = optimizer.compile(
        NestedProgram, trainset=[TICKET], valset=[TICKET], metric=exact
    )
    assert optimizer.seeds == [
        {"agents.router": ROUTE_SEED, "agents.reply": REPLY_SEED}
    ]
    assert [update.target.module for update in result.artifact.updates] == [
        "agents.router",
        "agents.reply",
    ]
    assert result.build().agents["auditor"].get_task_instruction() == AUDIT_SEED


def test_v2_artifact_roundtrip_applies_both_prompts(tmp_path):
    artifact = ProgramArtifact(
        2,
        (
            _update("router", ROUTE_SEED, ROUTE_TUNED),
            _update("reply", REPLY_SEED, REPLY_TUNED),
        ),
    )
    path = tmp_path / "program.optim.json"
    artifact.save(path)

    stored = json.loads(path.read_text())
    assert stored["version"] == 2
    assert [update["target"]["module"] for update in stored["updates"]] == [
        "router",
        "reply",
    ]

    # One loader entry point serves both formats.
    restored = OptimizationArtifact.load(path)
    assert isinstance(restored, ProgramArtifact)
    assert restored == artifact

    program = TicketProgram()
    restored.apply(program)
    restored.apply(program)  # Applying twice is supported.
    assert program.router.get_task_instruction() == ROUTE_TUNED
    assert program.reply.get_task_instruction() == REPLY_TUNED
    assert program("Missing invoice") == "queue-billing:tone-strict"
    assert TicketProgram().router.get_task_instruction() == ROUTE_SEED

    legacy = tmp_path / "single.optim.json"
    _update("router", ROUTE_SEED, ROUTE_TUNED).save(legacy)
    single = OptimizationArtifact.load(legacy)
    assert isinstance(single, OptimizationArtifact)
    assert single.version == 1


@pytest.mark.parametrize(
    "second",
    [
        _update("reply", "A different draft.", REPLY_TUNED),
        _update("ghost", "Anything.", REPLY_TUNED),
    ],
    ids=["changed-source", "missing-target"],
)
def test_v2_artifact_rolls_back_when_a_later_target_is_incompatible(second):
    artifact = ProgramArtifact(2, (_update("router", ROUTE_SEED, ROUTE_TUNED), second))
    program = TicketProgram()

    with pytest.raises((ValueError, AttributeError)):
        artifact.apply(program)

    # The first update is reverted, so the program is unchanged end to end.
    assert program.router.get_task_instruction() == ROUTE_SEED
    assert program.reply.get_task_instruction() == REPLY_SEED
    assert program("Missing invoice") == "queue-general:tone-loose"


def test_artifact_rejects_explicit_aliases_and_duplicate_targets(tmp_path):
    program = TicketProgram()

    # Only the documented field names are accepted by the target schema.
    with pytest.raises(TypeError):
        TextTarget(agent="router")
    with pytest.raises(TypeError):
        TextTarget(module="router", instruction=ROUTE_TUNED)

    duplicate = ProgramArtifact(
        2,
        (
            _update("router", ROUTE_SEED, ROUTE_TUNED),
            _update("router", ROUTE_SEED, REPLY_TUNED),
        ),
    )
    with pytest.raises(ValueError, match="Multiple targets"):
        duplicate.apply(program)
    assert program.router.get_task_instruction() == ROUTE_SEED

    path = tmp_path / "aliased.json"
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "updates": [
                    {
                        "version": 1,
                        "target": {"agent": "router"},
                        "source_sha256": digest(ROUTE_SEED),
                        "text": ROUTE_TUNED,
                    }
                ],
            }
        )
    )
    with pytest.raises(msgspec.ValidationError, match="agent"):
        OptimizationArtifact.load(path)


def test_empty_program_artifact_is_rejected():
    with pytest.raises(ValueError, match="at least one target"):
        ProgramArtifact(2, ()).apply(TicketProgram())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"target": TextTarget(), "targets": {"routing": TextTarget()}},
            "either target or targets",
        ),
        (
            {
                "targets": {
                    "routing": TextTarget(module="router"),
                    "again": TextTarget(module="router"),
                }
            },
            "Multiple targets",
        ),
        ({"targets": {" ": TextTarget(module="router")}}, "non-empty names"),
        ({"targets": {"routing": "router"}}, "non-empty names"),
    ],
)
def test_invalid_target_requests_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RecordingOptimizer().compile(
            TicketProgram, trainset=[TICKET], valset=[TICKET], metric=exact, **kwargs
        )


def test_backend_implementing_only_single_target_search_cannot_compile_jointly():
    class TextOnly(Optimizer):
        def _search(self, seed, trainset, valset, evaluator):
            return TextResult(seed, 0.0, 0.0, 1)

    with pytest.raises(NotImplementedError, match="joint program search"):
        TextOnly().compile(
            TicketProgram,
            targets={"routing": TextTarget(module="router")},
            trainset=[TICKET],
            valset=[TICKET],
            metric=exact,
        )

    # A single target still works and persists the version 1 format.
    result = TextOnly().compile(
        TicketProgram,
        target=TextTarget(module="router"),
        trainset=[TICKET],
        valset=[TICKET],
        metric=exact,
    )
    assert isinstance(result.search, TextResult)
    assert isinstance(result.artifact, OptimizationArtifact)
    assert result.artifact.version == 1
    assert result.artifact.target == TextTarget(module="router")
    assert result.build().router.get_task_instruction() == ROUTE_SEED


def test_joint_compilation_validates_the_selected_components():
    class WrongComponents(RecordingOptimizer):
        def _search_program(self, seeds, trainset, valset, evaluator):
            return ProgramResult({"other": ROUTE_TUNED}, 0.0, 0.0, 1)

    with pytest.raises(ValueError, match="incompatible program components"):
        WrongComponents().compile(
            TicketProgram, trainset=[TICKET], valset=[TICKET], metric=exact
        )

    class ProbeWrongCandidate(RecordingOptimizer):
        def _search_program(self, seeds, trainset, valset, evaluator):
            evaluator({"other": ROUTE_TUNED}, valset[0])
            return ProgramResult(dict(seeds), 0.0, 0.0, 1)

    with pytest.raises(ValueError, match="Candidate components differ"):
        ProbeWrongCandidate().compile(
            TicketProgram, trainset=[TICKET], valset=[TICKET], metric=exact
        )
