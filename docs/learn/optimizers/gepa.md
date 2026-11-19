# Optimizers and evaluation

Use `GEPA.optimize_text()` to search for a better string and `GEPA.compile()` to evaluate changes inside an `nn.Agent` or a custom `nn.Module`. Both accept a metric that you define. A Signature is optional. From a checkout containing the optimizer, install the search backend on Python 3.11–3.14:

```bash
uv sync --extra optimizers
```

For a release that includes this feature, install the published extra with `pip install 'msgflux[optimizers]'`.

## Optimize free text

This example searches for a shorter help message that describes the actions a reader needs. It uses the real GEPA engine and a deterministic reflection callable so you can reproduce the result offline.

```python
from msgflux.optim import GEPA

def propose(messages):
    return "```\nOpen Settings to cancel or renew.\n```"

def help_score(candidate, required_action):
    score = float(required_action in candidate and len(candidate) <= 50)
    return score, {"feedback": f"Mention {required_action!r} in at most 50 characters."}

optimizer = GEPA(reflection_model=propose, max_metric_calls=12)
result = optimizer.optimize_text(
    "Manage your account here.",
    trainset=["cancel", "renew"],
    valset=["cancel", "renew"],
    evaluator=help_score,
)
print(result.best_text)
```

Here both sets use the same two requirements to demonstrate a deterministic contract. When estimating quality on unseen inputs, use separate examples for search, selection and final evaluation. Write `result.best_text` to a UTF-8 text file to use it in any application.

## Compile and deploy an Agent

Run the following example from a repository checkout with `uv run --extra optimizers python examples/optimize_triage.py`. It uses a deterministic chat model, runs inference through the regular Agent entry point, saves an artifact and evaluates a newly constructed agent on held-out tickets. Expected validation accuracy is 0.0 before optimization and 1.0 after; holdout accuracy is 1.0 on two examples.

```python
--8<-- "examples/optimize_triage.py"
```

In an application, pass your normal msgFlux chat completion model as `reflection_model` and configure the task model in `build_agent()`. For example, after configuring an OpenAI account or loading your project environment, use `mf.Model.chat_completion("openai/gpt-6-luna")`. Live model proposals and validation scores vary. A reflection callable can also accept the engine's prompt and return the proposed response text.

By default, compilation gives the reflection model the training input, the root program's prediction, the numeric score and the metric's feedback. This lets it inspect the request that produced an error. Structured predictions retain their output fields. Set `include_example_context=False` when the metric should control exactly what the reflection model sees. With either setting, the metric can supply its own redacted values for `inputs` or `prediction`. Standalone `optimize_text` uses the diagnostic payload returned by its evaluator.

Each trial builds a fresh program, applies the candidate, executes the root module and calls `metric(example, prediction)`. A string in `Example.inputs` is passed positionally; a mapping is expanded into keyword arguments. The metric receives the consumed response and has access to `Example.labels`. Supply a factory that creates independent agents, conversation state and tools suitable for repeated execution. Configure external clients in the factory or inject clients owned by your application. From an async application run the synchronous compilation with `await asyncio.to_thread(optimizer.compile, ...)`.

### Choose a target

| Program | Argument to compile |
|---|---|
| Root agent with a system prompt or Signature instructions | Default, `TextTarget()` |
| Child agent registered as `self.router` in an `nn.Module` | `target=TextTarget(module="router")` |
| Local tool named `lookup` on that child agent | `target=TextTarget(module="router", tool="lookup")` |

Import `TextTarget` from `msgflux.optim`. Use `target=` for a single text or `targets=` for a jointly scored collection. For an agent with both an explicit system prompt and a Signature, the explicit instruction is the editable component. The generated output contract stays in the composed prompt. Use `agent.get_task_instruction()` to inspect the editable text and `agent.set_task_instruction(text)` to replace it manually. Replacements preserve the sequence of template expressions and format placeholders and honor a frozen `agent.system_prompt`. A Signature expressed only as input/output fields needs an explicit initial `system_prompt` for optimization.

For tools, the initial target supports directly registered portable local tools. Choose a metric that observes whether the program selected the right tool and handled its output.

### Save and restore

`result.build()` constructs a fresh program with the selected text. For deployment, use `result.save(path)` and apply `OptimizationArtifact.load(path)` to the program constructed by your application, as in the example. The JSON stores a format version, target address, source text digest and replacement text. Loading requires the base msgFlux package. A changed source text or unresolved target raises an error; applying the artifact twice to the same optimized text is supported. For an agent instruction, a new `state_dict()` snapshot also contains the editable `task_instruction` and composed `system_prompt`; loading it into a compatible agent restores both. Older snapshots with only the composed prompt remain usable for inference. When a legacy snapshot changes a composed Signature or schema prompt, rebuild from its source configuration to edit it. For ordinary module parameter and buffer snapshots, see [Module core concepts](../nn/module/core-concepts.md).

## Define and inspect a metric

Return a finite numeric score (higher is better) or a pair of score and diagnostic dictionary. The diagnostic can explain a missing field, an incorrect tool selection or a discrepancy with a reference answer. GEPA can pass that feedback to the reflection model. Keep private records out of feedback when using an external model.

`result.search` contains the initial and selected validation means and the evaluation count. `max_metric_calls` budgets evaluations of examples. The budget must cover the initial validation set. GEPA starts a proposal when the remaining budget can cover scoring the old and new minibatches and validating an accepted candidate. A smaller budget can return just the baseline. A multi-turn agent can make several model or tool calls during one evaluation. Run `evaluate(program, examples, metric)` on your held-out set to inspect `mean_score` and individual `rows` with example, prediction, score and feedback. Evaluate the original and compiled programs with the same inputs and inference settings. The standalone evaluator calls the supplied program for each example. Pass a callable that creates a fresh instance per call when a custom module carries state across inputs.

Set `workers` on `GEPA` to evaluate independent examples concurrently. Keep the default of one worker for models, tools or metrics that require serial access. Set `run_dir` to a dedicated local directory to enable GEPA checkpoints. Resume using the same program, input order, models, evaluator and dataset; use a fresh directory when changing the experiment. The resumed engine reports cumulative search evaluations and runs the initial validation batch again before loading the search state. Reuse a prediction cache for costly evaluations.

### Run a live evaluation

From the repository root, configure `OPENAI_API_KEY` in the environment or in a local `.env` file. Run:

```bash
uv run --extra optimizers python examples/optimize_query_planner.py
```

This example reuses the `QueryPlanner` Signature from `examples/intent-router.py` and runs GPT-6 Luna for task inference and reflection. It searches on three questions, selects on two and compares both agents on three separate questions. Its exact-plan metric checks the sequence of intents, required step fields and the rule that a document lookup follows a search. The script prints predictions and diagnostics for each held-out question. The dataset, metric and search budget are fixed. Model outputs can vary between runs. The six holdout calls are additional to the ten-evaluation search budget.

For a comparison with a published DSPy result on a fixed 68-example test set, see [Reproducing the DSPy Facility Support example](facility-support.md).

## Optimize an agent inside a program

A program is an ordinary `nn.Module`. Pass its class as the factory when it needs no constructor arguments, or supply a function that constructs it with its dependencies. The compiler builds the whole program for each evaluation. `TextTarget(module="router")` selects the editable instruction and the metric scores the value returned by the program's `forward`.

This example runs from a repository checkout using the deterministic task and reflection models from the triage example. The router decides when the billing specialist runs. The specialist's instruction is frozen. After compilation, an invoice reaches the specialist and the final output is `billing`.

```python
import msgflux as mf
from msgflux import nn
from msgflux.optim import GEPA, TextTarget
from examples.optimize_triage import TriageModel, build_agent, queue_accuracy, reflection


class SupportProgram(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = build_agent()
        self.billing = nn.Agent(
            name="billing",
            model=TriageModel(),
            system_prompt="Classify invoice requests as billing.",
        )
        self.billing.system_prompt.requires_grad_(requires_grad=False)

    def forward(self, ticket):
        queue = self.router(ticket)
        if queue == "billing":
            return self.billing(ticket)
        return queue


result = GEPA(reflection, max_metric_calls=8).compile(
    SupportProgram,
    target=TextTarget(module="router"),
    trainset=[mf.Example("Missing invoice", "billing")],
    valset=[mf.Example("Wrong invoice date", "billing")],
    metric=queue_accuracy,
)
assert result.build()("Copy of invoice") == "billing"
```

### Search several prompts together

Pass a program factory without a target to discover registered agents with a trainable, nonempty task instruction. A root Agent defaults to its own prompt. Use a mapping to select a subset, give components meaningful names, or include a local tool description:

```python
targets = {
    "routing": TextTarget(module="router"),
    "response": TextTarget(module="writer"),
}
result = optimizer.compile(
    build_program, targets=targets, trainset=train, valset=validation, metric=metric
)
winning_instructions = result.search.best_texts
result.save("program.optim.json")
```

Here `build_program` must return a module with `router` and `writer` agents. Every candidate contains both instructions. GEPA selects a parent, proposes a change to one named component at a time in round-robin order, then uses the root program's score for acceptance and validation. Use `GEPA(reflection_model, module_selector="all")` to propose changes to every target in the same candidate before measuring the combined effect. This also allows a pair of edits to improve a program when either edit on its own would leave the score unchanged. One program evaluation can call several agents. The reflector receives the root input and prediction and the selected agent's calls (positional inputs, keyword inputs and outputs). If a branch skips an agent, its call list is empty. A metric may add component-specific diagnostics with a mapping such as `{"routing_specific_info": {"feedback": "The invoice went to support"}}`. Set `include_example_context=False` to provide the entire reflection payload from the metric.

The version 2 artifact stores the selected updates together. Use the same `OptimizationArtifact.load(path).apply(program)` entry point for a single instruction or a joint artifact. A deployment must have compatible targets for the complete combination. The version 1 format continues to load.

Run `uv run --extra optimizers python examples/optimize_support_program.py` for a complete offline example. It optimizes both agents in a ticket-routing program, saves a joint artifact and checks the output of a newly constructed program after restoring it.

## Implement another optimizer

Subclass `msgflux.optim.Optimizer` and implement `_search(seed, trainset, valset, evaluator)`. It receives the initial string and a normalized evaluator returning a score and diagnostics. Return a `TextResult` with the selected string, baseline and selected validation means, and the number of metric calls. The shared base provides standalone text optimization, module compilation, target resolution and artifact persistence. Implement `_search_program(seeds, trainset, valset, evaluator)` to support joint search. It receives and returns named instruction mappings through `ProgramResult`. A backend can be an optional dependency imported at search time.

### Tracking experiments

A tracker can consume the backend-independent report today. For example, run the following after the triage evaluation to export summary metrics and per-example scores as JSON:

```python
import json
from examples.optimize_triage import main

report = main()
print(json.dumps({
    "mean_score": report.mean_score,
    "scores": [row.score for row in report.rows],
}))
```

Use `Evaluate.compare` to get paired baseline and candidate results. An optional MLflow integration is planned for recording datasets, metrics and selected artifacts. [Evaluating programs](evaluation.md)
