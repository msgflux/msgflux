import json
import subprocess
import sys

import msgspec
import pytest

from msgflux import Example, nn
from msgflux.models.response import ModelResponse
from msgflux.optim import (
    OptimizationArtifact,
    Optimizer,
    TextResult,
    TextTarget,
    evaluate,
)


class EchoModel:
    model_type = "chat_completion"

    def __call__(self, *, system_prompt, **_kwargs):
        response = ModelResponse()
        response.set_response_type("text_generation")
        response.add(system_prompt)
        return response


class ChooseText(Optimizer):
    def _search(self, seed, trainset, valset, evaluator):
        candidate = trainset[0].labels
        baseline, _ = evaluator(seed, valset[0])
        improved, _ = evaluator(candidate, valset[0])
        return TextResult(candidate, baseline, improved, 2)


class Workflow(nn.Module):
    def __init__(self, instruction="Before"):
        super().__init__()
        self.router = nn.Agent(
            name="router", model=EchoModel(), system_prompt=instruction
        )

    def forward(self, question):
        return self.router(question)


def test_compilation_persists_only_the_target_and_scores_the_root(tmp_path):
    examples = [Example({"question": "Hello"}, "After")]
    result = ChooseText().compile(
        Workflow,
        trainset=examples,
        valset=examples,
        metric=lambda example, prediction: prediction == example.labels,
        target=TextTarget(module="router"),
    )
    assert result.search == TextResult("After", 0, 1, 2)
    path = tmp_path / "compiled.json"
    result.save(path)
    restored = OptimizationArtifact.load(path)
    program = Workflow()
    restored.apply(program)
    restored.apply(program)
    assert (
        program.router.system_prompt.data
        == result.build().router.system_prompt.data
        == "After"
    )
    assert Workflow().router.system_prompt.data == "Before"
    changed = Workflow("Changed")
    with pytest.raises(ValueError, match="changed"):
        restored.apply(changed)
    assert changed.router.system_prompt.data == "Changed"
    data = json.loads(path.read_text())
    data["version"] = 99
    path.write_text(json.dumps(data))
    with pytest.raises(msgspec.ValidationError):
        OptimizationArtifact.load(path)


def test_evaluate_retains_diagnostics_and_validates_score():
    rows = [Example({"question": "hello"}, "Before"), Example("world", "Other")]
    report = evaluate(
        Workflow(), rows, lambda ex, value: (ex.labels == value, {"wanted": ex.labels})
    )
    assert report.mean_score == 0.5
    assert report.rows[1].feedback == {"wanted": "Other"}
    assert report.rows[0].prediction == "Before"
    with pytest.raises(ValueError, match="finite"):
        evaluate(str, [Example("one", "one")], lambda *_: float("nan"))


def test_deployment_imports_without_gepa():
    script = """
import sys
sys.modules['gepa'] = None
from msgflux.optim import OptimizationArtifact, evaluate
"""
    subprocess.run([sys.executable, "-c", script], check=True)  # noqa: S603


def test_documented_queue_metric_requires_one_label():
    from examples.optimize_triage import queue_accuracy

    example = Example("Invoice is missing", "billing")
    assert queue_accuracy(example, "Billing")[0] == 1
    assert queue_accuracy(example, "billing or access")[0] == 0


def test_signature_state_snapshot_retains_editable_text():
    def build():
        return nn.Agent(
            name="triage",
            model=EchoModel(),
            system_prompt="Seed",
            signature="ticket: str -> queue: str",
        )

    original = build()
    original.set_task_instruction("Revised")
    state = original.state_dict()
    restored = build()
    restored.load_state_dict(state)
    assert restored.get_task_instruction() == "Revised"
    assert restored.get_system_prompt() == original.get_system_prompt()
    restored.set_task_instruction("Final")
    assert "Write an encoded JSON" in restored.get_system_prompt()

    legacy = build()
    legacy.load_state_dict({"system_prompt": state["system_prompt"]})
    assert legacy.get_system_prompt() == original.get_system_prompt()
    with pytest.raises(ValueError, match="composed"):
        legacy.set_task_instruction("Ambiguous")


def test_compilation_evaluates_the_description_seen_by_the_model():
    def lookup(ticket: str) -> str:
        """Read a ticket."""
        return ticket

    class CatalogReader(EchoModel):
        def __call__(self, *, tool_catalog, **kwargs):
            entry = tool_catalog.tool_entries()[0]
            return super().__call__(system_prompt=entry.description)

    def build():
        return nn.Agent(name="support", model=CatalogReader(), tools=[lookup])

    example = Example("Where is my invoice?", "Retrieve an invoice by ticket.")
    result = ChooseText().compile(
        build,
        target=TextTarget(tool="lookup"),
        trainset=[example],
        valset=[example],
        metric=lambda ex, prediction: ex.labels == prediction,
    )
    assert result.search.best_score == 1
    original, compiled = build(), result.build()
    (original_schema,) = original.tool_library.get_tool_json_schemas()
    (schema,) = compiled.tool_library.get_tool_json_schemas()
    assert schema["function"]["description"] == example.labels
    assert original_schema["function"]["description"] == "Read a ticket."
    assert schema["function"]["parameters"] == original_schema["function"]["parameters"]
