# Optimizers and evaluation

Use `GEPA.optimize_text()` to search for a better string and `GEPA.compile()` to evaluate changes inside an `nn.Agent` or a custom `nn.Module`. Both accept a metric that you define. A Signature is optional. From a checkout containing the optimizer, install the search backend on Python 3.11–3.14:

```bash
uv sync --extra optimizers
```

For a release that includes this feature, install the published extra with `pip install 'msgflux[optimizers]'`.

## Optimize free text

Load `OPENAI_API_KEY` from the environment or a local `.env`. Here the reflector rewrites a help message. The evaluator checks whether the text mentions the invoice and receipt. Both splits deliberately contain the same two requirements because the metric describes a fixed copywriting contract.

```python
import msgflux as mf
from msgflux.optim import GEPA

mf.load_dotenv()
reflect = mf.Model.chat_completion("openai/gpt-6-luna")

def help_score(text, needed):
    return float(needed in text.lower()), {
        "feedback": f"Tell users to enter an invoice number to get a receipt. Include the word {needed}."
    }

result = GEPA(reflect, max_metric_calls=8).optimize_text(
    "Find a document.", trainset=["invoice", "receipt"],
    valset=["invoice", "receipt"], evaluator=help_score,
)
print(result.best_text)
```

In one run Luna produced “Ask the user to enter their invoice number to get a receipt.” The validation score rose from 0 to 1 in eight metric calls. You can write `result.best_text` to a UTF-8 file and use it anywhere that accepts text.

## Compile and deploy an Agent

Run `uv run --extra optimizers python examples/optimize_triage.py` with `OPENAI_API_KEY` configured. This example uses a live task model and reflector, saves the selected instruction and evaluates a newly constructed agent on held-out tickets.

```python
--8<-- "examples/optimize_triage.py"
```

In an application, pass your normal msgFlux chat completion model as `reflection_model` and configure the task model in `build_agent()`. For example, after configuring an OpenAI account or loading your project environment, use `mf.Model.chat_completion("openai/gpt-6-luna")`. Live model proposals and validation scores vary. A reflection callable can also accept the engine's prompt and return the proposed response text.

By default, compilation gives the reflection model the training input, the root program's prediction, the numeric score and the metric's feedback. This lets it inspect the request that produced an error. Structured predictions retain their output fields. Set `include_example_context=False` when the metric should control exactly what the reflection model sees. With either setting, the metric can supply its own redacted values for `inputs` or `prediction`. Standalone `optimize_text` uses the diagnostic payload returned by its evaluator.

Each trial builds a fresh program, applies the candidate, executes the root module and calls `metric(example, prediction)`. A string in `Example.inputs` is passed positionally; a mapping is expanded into keyword arguments. The metric receives the consumed response and has access to `Example.labels`. Supply a factory that creates independent agents, conversation state and tools suitable for repeated execution. Configure external clients in the factory or inject clients owned by your application. From an async application run the synchronous compilation with `await asyncio.to_thread(optimizer.compile, ...)`.

### Choose a target

| Program | Argument to compile |
|---|---|
| Root agent instruction | `target=TextTarget()` |
| Child agent registered as `self.router` in an `nn.Module` | `target=TextTarget(module="router")` |
| Text parameter named `style` on a custom writer module | `target=TextTarget(module="writer", parameter="style")` |
| Runtime instructions for using the local tool `lookup` | `target=TextTarget(tool="lookup", field="usage_guidance")` |
| Local tool named `lookup` on that child agent | `target=TextTarget(module="router", tool="lookup")` |

Import `TextTarget` from `msgflux.optim`. Use `target=` for a single text or `targets=` for a jointly scored collection. For an agent with both an explicit system prompt and a Signature, the explicit instruction is the editable component. The generated output contract stays in the composed prompt. Use `agent.get_task_instruction()` to inspect the editable text and `agent.set_task_instruction(text)` to replace it manually. Replacements preserve the sequence of template expressions and format placeholders and honor a frozen `agent.system_prompt`. A Signature expressed only as input/output fields needs an explicit initial `system_prompt` for optimization.

### Optimize a tool description

Without an explicit target, the compiler discovers registered, nonempty text parameters with `requires_grad=True`. System prompts start trainable. Each directly exposed portable local tool has a registry-owned description parameter that starts frozen. Enable it inside the factory so every trial sees the same configuration:

```python
agent.tool_library.description_parameter("lookup").requires_grad_(requires_grad=True)
```

Use `target=TextTarget(tool="lookup")` for an explicit, description-only search, including when the automatic discovery flag is off. Supply `module="router"` to address the same entry on a child agent. MCP, provider-native, deferred and bucket-managed entries use descriptions supplied by their own runtime.

1. **Read the starting text.** `compile` constructs the program. `TextTarget.read` calls `get_tool_description` on the agent's registry-owned parameter. For a local function this description can originate from its docstring.
2. **Try a candidate in the running agent.** For each example, the compiler builds a fresh program. `TextTarget.write` calls `set_tool_description`, which updates the text parameter. The catalog shown to the task model carries the new text. The entry retains its executor and argument schema.
3. **Score the outcome.** The metric receives the example and the root program's prediction. Include requests that should use the tool and requests that should go elsewhere. If the requirement is to verify a specific call, have the program expose that observation to the metric. Answer accuracy alone measures the final answer.
4. **Deploy the selected description.** `fit.build()` applies the selected text to a fresh program. To deploy in another process, save the optimization artifact, reconstruct the compatible program and call `OptimizationArtifact.load(path).apply(program)`. Use `target.read(program)` to inspect the deployed description.

Run `uv run --extra optimizers python examples/optimize_tool_description.py` from a checkout. The script jointly optimizes the agent instruction and lookup description, saves both and scores the restored agent against the exact archived receipt code. To reproduce the provider configuration of the recorded run, use an OpenCode Go account named `work` and the OpenAI key for Luna:

```bash
uv run --extra optimizers python examples/optimize_tool_description.py \
  --task-model opencode-go/deepseek-v4.1-flash --account work
```

In the recorded run, baseline and selected accuracy were 0 and 1 on both validation and holdout, with six search evaluations. Each of those splits contains one invoice. The fixed examples exercise dispatch, compilation and restoration. Live model output varies across runs.

```python
--8<-- "examples/optimize_tool_description.py"
```

To optimize the router instruction and its lookup description together, pass both addresses to one search. Each candidate is scored on the output of `build_program()`'s root module:

```python
targets = {
    "routing": TextTarget(module="router"),
    "receipt_lookup": TextTarget(module="router", tool="lookup"),
}
fit = optimizer.compile(
    build_program, targets=targets, trainset=train, valset=validation, metric=task_accuracy
)
```

### Optimize runtime usage guidance

A tool description appears in the callable schema. The runtime inserts its `usage_guidance` into the agent's system prompt when the tool is available. These are separate text parameters with independent starting values and training flags. Give guidance an initial instruction through the tool configuration or enable the tool's declared defaults, then opt it into automatic discovery inside the factory:

```python
@mf.tool_config(usage_guidance="Use lookup when the request includes an invoice reference.")
def lookup(reference: str) -> str:
    """Look up a record."""
    return "rcpt-" + reference[::-1] + "-archived"

def build_desk():
    desk = nn.Agent(name="desk", model=task_model, system_prompt="Help with receipts.", tools=[lookup])
    desk.tool_library.usage_guidance_parameter("lookup").requires_grad_(requires_grad=True)
    return desk
```

This factory exposes the desk instruction and the runtime guidance to joint search. To search just the guidance, pass `target=TextTarget(tool="lookup", field="usage_guidance")`. The default field for `TextTarget(tool="lookup")` remains `description`. Both fields use the registry’s state dict and can be restored together. Inspect the effective prompt with `desk.get_system_prompt()` and the callable schema with `desk.tool_library.get_tool_json_schemas()`.

For a runnable version that trains all three texts and logs the artifact and holdout scores locally:

```bash
MLFLOW_TRACKING_URI=sqlite:///receipts.db uv run --extra optimizers --extra mlflow \
  python examples/optimize_tool_description.py --train-usage-guidance --mlflow
```

The example uses `openai/gpt-6-luna` by default. To use OpenCode Go for the task model, append `--task-model opencode-go/deepseek-v4.1-flash --account work`. In a run with that configuration, GEPA selected changes to all three texts in six metric calls. Exact-match accuracy went from 0 to 1 on the single held-out receipt. [Runtime tool guidance and defaults](../nn/agent/tools/config.md#usage_guidance).

### Save and restore

`result.build()` constructs a fresh program with the selected text. For deployment, use `result.save(path)` and apply `OptimizationArtifact.load(path)` to the program constructed by your application, as in the example. The JSON stores a format version, target address, source text digest and replacement text. Loading requires the base msgFlux package. A changed source text or unresolved target raises an error; applying the artifact twice to the same optimized text is supported. For an agent instruction, a new `state_dict()` snapshot also contains the editable `task_instruction` and composed `system_prompt`; loading it into a compatible agent restores both. Older snapshots with only the composed prompt remain usable for inference. When a legacy snapshot changes a composed Signature or schema prompt, rebuild from its source configuration to edit it. Registry-owned description values are also part of the agent state dict. The factory configures trainability. A snapshot created before descriptions became parameters retains the factory’s tool text when loaded. For ordinary module parameter and buffer snapshots, see [Module core concepts](../nn/module/core-concepts.md).

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

Run `uv run --extra optimizers python examples/gepa_single_agent_and_program.py` with an OpenAI key. Its first factory constructs a single router. Its second factory builds a two-agent program and evaluates the queue and priority together. Both use Luna for inference and reflection.

When a worker appears only in `tools=[worker]`, its internal prompt stays outside automatic discovery. This rule applies whether the root is a single agent or a custom module. To train the worker jointly, expose it on the program through a named attribute, such as `self.billing`. This preserves the same instance for delegation and training. Use `target=TextTarget(module="billing")` to select just that worker. [Agent-as-tool factory example](../nn/agent/tools/index.md#choosing-which-agent-prompts-to-optimize).

### Search several prompts together

Pass a program factory without a target to discover its trainable text parameters. Aliases of the same parameter are selected once. Two separate parameters with identical starting text remain separate targets. Use a mapping to select a subset, give components meaningful names, or include a local tool description:

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

## Implement another optimizer

Subclass `msgflux.optim.Optimizer` and implement `_search(seed, trainset, valset, evaluator)`. It receives the initial string and a normalized evaluator returning a score and diagnostics. Return a `TextResult` with the selected string, baseline and selected validation means, and the number of metric calls. The shared base provides standalone text optimization, module compilation, target resolution and artifact persistence. Implement `_search_program(seeds, trainset, valset, evaluator)` to support joint search. It receives and returns named instruction mappings through `ProgramResult`. A backend can be an optional dependency imported at search time.

### Tracking experiments

Use `Evaluate.compare` to get paired baseline and candidate results. To record the search budget used, selected text artifact, model identities and holdout metrics, call `log_optimization` inside an active MLflow run. [Tracking optimization with MLflow](mlflow.md).
