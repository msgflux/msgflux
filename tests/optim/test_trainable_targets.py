"""Compile and restore parameters through ordinary program and tool registrations."""

from msgflux import nn
from msgflux.core.examples import Example
from msgflux.optim import (
    OptimizationArtifact,
    Optimizer,
    ProgramResult,
    TextResult,
    TextTarget,
)
from tests.optim.test_program import AuditModel


class SelectText(Optimizer):
    def _search(self, seed, trainset, valset, evaluator):
        value, _ = evaluator(seed + " edited", valset[0])
        return TextResult(seed + " edited", 0, value, 1)

    def _search_program(self, seeds, trainset, valset, evaluator):
        selected = {name: text + " edited" for name, text in seeds.items()}
        value, _ = evaluator(selected, valset[0])
        return ProgramResult(selected, 0, value, 1)


def fit(factory, **kwargs):
    examples = [Example("question", "audited")]
    return SelectText().compile(
        factory,
        trainset=examples,
        valset=examples,
        metric=lambda e, p: float(p == e.labels),
        **kwargs,
    )


def build_main():
    worker = nn.Agent(
        name="worker",
        model=AuditModel(),
        system_prompt="Delegate",
        description="Delegate a task",
    )
    return nn.Agent(
        name="main", model=AuditModel(), system_prompt="Route", tools=[worker]
    )


def test_agent_as_tool_stays_out_of_auto_search_and_allows_explicit_selection():
    result = fit(build_main)
    main = result.build()
    worker_path = "tool_library.library.worker.impl"
    worker = main.get_submodule(worker_path)
    assert result.artifact.version == 1
    assert main.get_task_instruction() == "Route edited"
    assert worker.get_task_instruction() == "Delegate"
    assert worker.system_prompt.requires_grad
    explicit = fit(build_main, target=TextTarget(module=worker_path)).build()
    assert explicit.get_task_instruction() == "Route"
    assert (
        explicit.get_submodule(worker_path).get_task_instruction() == "Delegate edited"
    )


def test_joint_discovery_and_restore_with_shared_agent_and_equal_valued_parameters(
    tmp_path,
):
    class Program(nn.Module):
        def __init__(self):
            super().__init__()
            self.main = build_main()
            self.worker = self.main.get_submodule("tool_library.library.worker.impl")
            self.left = nn.Parameter("Read", "instruction")
            self.right = nn.Parameter("Read", "instruction")
            self.alias = self.left

        def forward(self, text):
            return self.worker(text)

    program = Program()
    names = dict(program.named_parameters())
    assert names["left"] is not names["right"]
    assert "alias" not in names
    assert (
        dict(program.named_parameters(remove_duplicate=False))["alias"] is names["left"]
    )
    fit_result = fit(Program)
    assert set(fit_result.search.best_texts) == {"left", "right", "main", "worker"}
    file = tmp_path / "joint.json"
    fit_result.save(file)
    OptimizationArtifact.load(file).apply(program)
    assert (
        program.left.data == program.right.data == program.alias.data == "Read edited"
    )
    assert program.worker.get_task_instruction() == "Delegate edited"
    assert program.main.get_task_instruction() == "Route edited"


def test_tool_opt_in_compiles_catalog_and_instruction_together():
    def factory():
        main = build_main()
        main.tool_library.description_parameter("worker").requires_grad_(
            requires_grad=True
        )
        return main

    result = fit(factory)
    assert set(result.search.best_texts) == {"instruction", "tools.worker.description"}
    main = result.build()
    assert (
        main.tool_library.get_tool_definition("worker").description
        == "Delegate a task edited"
    )
    assert (
        main.get_submodule("tool_library.library.worker.impl").get_task_instruction()
        == "Delegate"
    )


def test_state_restores_matching_parent_and_child_registration_names():
    def lookup(key: str) -> str:
        """Return the matching record."""
        return key

    def build():
        child = nn.Agent(name="child", model=AuditModel(), tools=[lookup])
        return nn.Agent(name="parent", model=AuditModel(), tools=[child])

    worker = "tool_library.library.child.impl"
    target = TextTarget(module=worker, tool="lookup")
    trained = build()
    target.write(trained, "Look up an invoice by its reference.")
    restored = build()
    restored.load_state_dict(trained.state_dict())
    assert target.read(restored) == target.read(trained)


def test_usage_guidance_reaches_model_at_runtime_and_survives_restoration(tmp_path):
    from msgflux.optim import Evaluate
    from tests.optim.test_optimizer import EchoModel

    def lookup(reference: str) -> str:
        """Return a receipt."""
        return reference

    lookup.tool_config = {"usage_guidance": "Check the invoice."}

    def build():
        agent = nn.Agent(
            name="desk", model=EchoModel(), system_prompt="Route", tools=[lookup]
        )
        library = agent.tool_library
        library.description_parameter("lookup").requires_grad_(requires_grad=True)
        guidance = library.usage_guidance_parameter("lookup")
        assert not guidance.requires_grad
        guidance.requires_grad_(requires_grad=True)
        return agent

    example = Example("Invoice?", "Check the invoice. edited")

    def metric(row, response):
        return float(row.labels in response)

    fit_result = SelectText().compile(
        build, trainset=[example], valset=[example], metric=metric
    )
    assert set(fit_result.search.best_texts) == {
        "instruction",
        "tools.lookup.description",
        "tools.lookup.usage_guidance",
    }
    path = tmp_path / "tool-fields.json"
    fit_result.save(path)
    restored = build()
    OptimizationArtifact.load(path).apply(restored)
    comparison = Evaluate(dataset=[example], metric=metric).compare(build(), restored)
    assert comparison.baseline_mean_score == 0 and comparison.candidate_mean_score == 1
    (schema,) = restored.tool_library.get_tool_json_schemas()
    assert schema["function"]["description"] == "Return a receipt. edited"
    assert example.labels not in str(schema)
    assert restored.get_task_instruction() == "Route edited"
    fresh = build()
    fresh.load_state_dict(restored.state_dict())
    explicit = TextTarget(tool="lookup", field="usage_guidance")
    assert explicit.read(fresh) == example.labels
    assert example.labels in fresh.get_system_prompt()
