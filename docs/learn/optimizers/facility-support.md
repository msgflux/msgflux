# Reproducing the DSPy Facility Support example

The DSPy notebook classifies the urgency, sentiment and service categories of 200 facility-support requests. Its saved outputs report a test score of **75.44%** with the original instructions and **87.01%** with GEPA's selected instructions. Each test request is worth one point, divided equally among correct urgency, correct sentiment and accuracy across ten category flags. [Original notebook at the tested DSPy revision](https://github.com/stanfordnlp/dspy/blob/2c07ebbee3a66f6e04b71e141c6c198f1b31a129/docs/docs/tutorials/gepa_facilitysupportanalyzer/index.ipynb)

The msgFlux example reads the initial instructions and the three published winning prompts directly from the local notebook. It repeats the notebook's shuffle with Python seed 0 and its 66/66/68 train, validation and test split. It evaluates three conditions on the **same 68 test requests**:

| Condition | Instructions |
|---|---|
| Baseline | The source docstrings in the notebook |
| Published prompt transfer | The three final GEPA prompts recorded in the notebook output |
| Native GEPA fit | Three separate searches using the msgFlux compiler, starting from the original instructions |

The task model and reflection model both default to `openai/gpt-6-luna`. The original tutorial used GPT-4.1 Nano for inference and GPT-5 for reflection. OpenAI returned HTTP 404 for both the historical Nano alias and its dated snapshot during reproduction. The msgFlux program uses three typed agent calls with reasoning and output fields. Its native fit searches each field independently, uses the first eight training cases and first four validation cases, and sets a budget of 18 metric calls per field. DSPy used a jointly scored program with predictor-specific feedback and its light budget. The transfer condition tests how the published instructions behave on a different model and runtime. The native condition measures what this smaller training run learned.

## Observed results

Each percentage below is computed from the 68 test cases. Category accuracy counts agreement on both positive and negative labels.

| Instruction set | Urgency | Sentiment | Categories | Overall |
|---|---:|---:|---:|---:|
| Original | 70.59% | 63.24% | 92.94% | 75.59% |
| Published DSPy prompts | 95.59% | 76.47% | 93.09% | 88.38% |
| Native GEPA fit | 70.59% | 63.24% | 92.94% | 75.59% |

The native search retained the initial instruction for each field in this run. Identical prompts therefore share their holdout predictions with the baseline. The result shows a transfer benefit from the published prompts. The reduced native fit did not reproduce that improvement. [Download the scores, model names, budget usage and prompt fingerprints](facility-support-run.json).

## Run locally

Start in the msgFlux repository and configure `OPENAI_API_KEY` in the environment or a local `.env`. Set `DSPY` to the local DSPy checkout. The script reads the notebook as data, and the dataset download is pinned and checksum-verified. The report contains the model names, per-example test scores, search evaluation counts and instructions. The model service can return different predictions on repeated runs.

```bash
DSPY=../dspy
NOTEBOOK="$DSPY/docs/docs/tutorials/gepa_facilitysupportanalyzer/index.ipynb"
curl --fail --location \
  https://raw.githubusercontent.com/meta-llama/llama-prompt-ops/f6ce9977392c10aae51b9832c8558cf0001e68f2/use-cases/facility-support-analyzer/dataset.json \
  --output /tmp/facility-support.json
uv run --extra optimizers python examples/reproduce_facility_support.py \
  "$NOTEBOOK" /tmp/facility-support.json --report /tmp/facility-report.json
```

Set `--workers` to control holdout concurrency. Training runs serially. Up to three task-model calls are required per holdout case per condition. Identical instructions for a field reuse their test scores across conditions within a run. The report records the notebook SHA-256 so another run can verify its input revision.

## Full-budget native run and DSPy control

The local notebook records a budget of approximately 1,643 full-program evaluations. GEPA samples training minibatches and stops by evaluation budget; its 39 iterations in the saved run are an observed count. The full experiment uses all 66 training and 66 validation examples and reserves all 68 test examples for final scoring. Both implementations use DeepSeek V4.1 Flash on OpenCode Go for prediction and GPT-6 Luna for reflection.

Run the native command once per field. Each invocation has an independent checkpoint and prediction cache. The task model uses the `work` account's OpenCode Go credentials; reflection uses the OpenAI credentials configured for the process. These runs exercise `GEPA.compile`, the normal Agent entry point, module targeting and artifact persistence. Each field has a budget of 1,643 evaluations. DSPy's equal nominal budget counts executions of its entire **three-predictor** program. The search algorithms allocate that work differently, and model caching affects the number of network requests.

```bash
for field in urgency sentiment categories; do
  uv run --extra optimizers python examples/facility_full_native.py \
    "$field" "$NOTEBOOK" /tmp/facility-support.json ./facility-run
done
DSPY_CACHEDIR=./facility-run/reference-cache \
  uv run --with-editable "$DSPY" --with litellm --extra optimizers \
  python examples/facility_full_reference.py \
  "$NOTEBOOK" /tmp/facility-support.json ./facility-run/reference
```

The DSPy control executes the pinned notebook's program, score and feedback functions. It keeps round-robin predictor updates, Pareto selection and disabled merges. Per-example predictions allow both scorers to be checked on identical outputs, independently of model sampling.

After both searches and holdout evaluations finish, run the offline comparison against the same files. It recomputes both frameworks' scores for each holdout prediction and checks each saved msgFlux instruction against the validation scores of all its accepted candidates. The resulting `comparison.json` includes scores for individual test requests. Its `paired_metrics` section uses the named-metric evaluator to report the mean change, improvement and regression counts, ties and per-request deltas for overall score, urgency, sentiment and categories. Deltas follow the holdout order. In the joint run, sentiment improved on 22 requests and regressed on three. Urgency and categories tied on all 68.

```bash
uv run --with-editable "$DSPY" --with litellm --extra optimizers \
  python examples/facility_full_report.py \
  "$NOTEBOOK" /tmp/facility-support.json ./facility-run
```

Use a fresh output directory when changing the dataset, task model, seed instructions or runtime settings. Reuse a directory only to resume the same experiment.

To reproduce the metric-only condition, add `--feedback-only` to the native command and use a separate output directory. The context-enabled condition adds the input ticket, the agent's structured prediction (including its reasoning field), and the score to the same feedback. Compare both outputs against one DSPy control with `facility_full_report.py --reference-dir ./facility-run/reference`. DSPy also uses feedback specific to each predictor, including which categories were missed or added, and scores the joint program. The native model uses provider-default sampling settings. The DSPy control requests temperature 1 and a 6,000-token task limit.

### Joint search on the complete program

This variant registers three agents on one module and passes their named prompts to a single GEPA search. Each proposal is scored by running the root program and averaging the three field scores exactly as in the notebook. The reflector receives the selected field's call and expected/predicted feedback. A budget of 1,643 here counts whole-program examples.

```bash
uv run --extra optimizers python examples/facility_joint.py \
  "$NOTEBOOK" /tmp/facility-support.json ./facility-joint
uv run --with-editable "$DSPY" --with litellm --extra optimizers \
  python examples/facility_full_report.py \
  "$NOTEBOOK" /tmp/facility-support.json ./facility-run \
  --reference-dir ./facility-run/reference --joint-dir ./facility-joint
```

The joint run writes one version 2 artifact containing the three instructions. The report checks those instructions against the joint validation candidates and the cached predictions for each holdout request.

### Full-budget results on the 68 held-out requests

| Search | Baseline | After GEPA | Gain |
|---|---:|---:|---:|
| msgFlux, separate agents with metric feedback only | 79.75% | 85.74% | +5.98 pp |
| msgFlux, separate agents with input and prediction context | 79.75% | 88.77% | +9.02 pp |
| msgFlux, joint program with component call traces | 79.75% | 89.07% | +9.31 pp |
| DSPy, joint program | 80.59% | 88.38% | +7.79 pp |

Gains use unrounded paired scores. The msgFlux joint search sampled all 66 training requests and validated 21 candidates over 33 iterations and 1,584 whole-program evaluations. The winning combination changed the sentiment prompt and retained the original urgency and category prompts. On the test set, it improved 22 requests, tied on 43 and regressed on three relative to its baseline. DSPy recorded 1,707 evaluations with the same nominal budget of 1,643 because the engines stop at different iteration boundaries.

The offline report confirmed that the saved msgFlux combination attained the highest candidate score on validation and reproduced the cached holdout predictions. Both metrics agreed on all 408 holdout rows in the context-enabled comparison (three methods, before and after, on 68 cases). The 0.69 percentage point difference between the two joint searches comes from a single split and model run. [Download per-example scores, candidate counts and input checksums](facility-support-full-run.json).
