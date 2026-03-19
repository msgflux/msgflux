# Inline

## ✦₊⁺ Overview

`Inline` is a lightweight domain-specific language (DSL) for composing
**message-centric workflows** from small, focused functions.

You describe the pipeline as a string — sequences, branches, parallel stages,
and loops — and `Inline` takes care of executing it. Every step receives a
shared `dotdict` message, reads what it needs, and writes its results back.

---

## 1. **How It Works**

Every module in an `Inline` pipeline receives the shared `dotdict` as its only
argument and writes results directly onto it. **Return values are ignored.**

Modules should treat the message as the **single source of truth**. This design
ensures that pipelines behave consistently in sequential, parallel, and async
execution — the caller always reads results from `msg`, never from a return
value.

```python
# Correct — write to msg, return nothing
def enrich(msg):
    msg.score = len(msg.text)

# Return value is silently discarded — still works, but misleading
def tag(msg):
    msg.tag = "urgent"
    return msg  # has no effect
```

Because every step mutates the same object, later steps automatically see
everything written by earlier ones — no need to thread outputs through function
arguments.

### Compiled execution

The pipeline expression is parsed once at construction time into an internal
execution tree (AST). This means syntax errors are caught immediately when you
create the `Inline` object, not at runtime, and the same parsed structure is
reused across every call without re-parsing the string.

---

## 2. **Constructor**

```python
Inline(expression, modules, *, max_iterations=1000)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `expression` | `str` | Pipeline string. Parsed once at construction. Syntax errors surface immediately. |
| `modules` | `Mapping[str, Callable]` | Name-to-callable mapping. Values may be plain functions, async functions, or `nn.Module` instances. |
| `max_iterations` | `int` | Maximum number of iterations any `@{…}` while loop may run before a `RuntimeError` is raised. Default: `1000`. |

```python
import msgflux as mf

def prep(msg):
    msg.ready = True

def feat_a(msg):
    msg.score_a = 1

def feat_b(msg):
    msg.score_b = 2

def combine(msg):
    msg.total = msg.score_a + msg.score_b

flux = mf.Inline(
    "prep -> [feat_a, feat_b] -> combine",
    {"prep": prep, "feat_a": feat_a, "feat_b": feat_b, "combine": combine},
)
```

---

## 3. **Execution**

| Method | When to use |
|--------|-------------|
| `flux(msg)` | Synchronous |
| `await flux.acall(msg)` | Asynchronous |

Both return the same `msg` object after all steps have run.

---

## 4. **Syntax Overview**

| Pattern | Description | Example |
|---------|-------------|---------|
| `->` | Sequential | `"a -> b -> c"` |
| `[…]` | Parallel | `"[b, c]"` |
| `{cond ? a}` | Conditional (if) | `"{score > 0.9 ? accept}"` |
| `{cond ? a, b}` | Conditional (if/else) | `"{is_vip == true ? vip, standard}"` |
| `{c1 ? a, c2 ? b, c}` | Multi-branch (if/elif/else) | `"{score > 0.9 ? premium, score > 0.5 ? standard, fallback}"` |
| `@{field op value}: …;` | While loop | `"@{retries < 3}: fetch;"` |

`field` represents a **path read from the current message** at runtime
(e.g. `score`, `user.age`, `output.confidence`).

Fields are resolved against the `dotdict` message passed to the pipeline:

```
{confidence > 0.8}
{user.profile.age >= 18}
```

Conditions are not free-form Python — they are always a comparison between a
message field and a literal value.

---

## 5. **Sequential Execution**

Modules run in the order listed. Each step sees the message exactly as the
previous step left it, so earlier results are always available to later stages.

???+ example

    ```python
    import msgflux as mf

    def load(msg):
        msg.raw = "hello world"

    def tokenize(msg):
        msg.tokens = msg.raw.split()

    def count(msg):
        msg.n_tokens = len(msg.tokens)

    flux = mf.Inline(
        "load -> tokenize -> count",
        {"load": load, "tokenize": tokenize, "count": count},
    )

    msg = mf.dotdict()
    msg = flux(msg)

    print(msg.tokens)    # ["hello", "world"]
    print(msg.n_tokens)  # 2
    ```

---

## 6. **Parallel Execution**

Modules inside `[…]` run concurrently in a thread pool. All of them receive
**the same message object**, so they execute as a group before the pipeline
moves on.

### Design rule

Parallel modules must write to **disjoint message paths**. Writing to the same
key from two concurrent modules is a data race.

```python
# Safe — each module owns its own key
def fetch_weather(msg):
    msg.weather = "sunny"

def fetch_news(msg):
    msg.news = ["headline_1"]

# Unsafe — both mutate the same list
def feat_a(msg):
    msg.results.append("a")

def feat_b(msg):
    msg.results.append("b")
```

???+ example

    ```python
    import msgflux as mf

    def fetch_weather(msg):
        msg.weather = "sunny"

    def fetch_news(msg):
        msg.news = ["headline_1"]

    def summarize(msg):
        msg.summary = f"{msg.weather} | {msg.news[0]}"

    flux = mf.Inline(
        "[fetch_weather, fetch_news] -> summarize",
        {
            "fetch_weather": fetch_weather,
            "fetch_news": fetch_news,
            "summarize": summarize,
        },
    )

    msg = mf.dotdict()
    msg = flux(msg)

    print(msg.summary)  # "sunny | headline_1"
    ```

!!! warning "Race Conditions"
    Parallel modules share the same `dotdict`. Writing to the **same key** from
    two modules simultaneously produces unpredictable results. Design parallel
    stages so each one owns a distinct subtree of the message.

---

## 7. **Conditionals**

A conditional evaluates conditions against the current message at runtime.
Conditions are checked in order — the **first match wins**. The last item
without `?` is the **default** branch (fallback).

### Syntax

```
{condition ? module}                              # if
{condition ? module_a, module_b}                  # if / else
{cond1 ? module_a, cond2 ? module_b, module_c}   # if / elif / else
```

Branches are separated by `,`. Each branch with `?` is a condition–module
pair. The last item without `?` is the default — it runs when no condition
matches. If no default is provided and nothing matches, no module runs.

Use `_` as an explicit wildcard when you want to make the default visually
clear:

```
{cond1 ? module_a, _ ? module_b}
```

???+ example "If / else"

    ```python
    import msgflux as mf

    def flag_urgent(msg):
        msg.label = "urgent"

    def flag_normal(msg):
        msg.label = "normal"

    flux = mf.Inline(
        "{priority > 7 ? flag_urgent, flag_normal}",
        {"flag_urgent": flag_urgent, "flag_normal": flag_normal},
    )

    msg = mf.dotdict(priority=9)
    msg = flux(msg)
    print(msg.label)  # "urgent"
    ```

???+ example "Multi-branch (if / elif / else)"

    ```python
    import msgflux as mf

    def route_premium(msg):
        msg.tier = "premium"

    def route_standard(msg):
        msg.tier = "standard"

    def route_free(msg):
        msg.tier = "free"

    flux = mf.Inline(
        "{score > 90 ? route_premium, score > 50 ? route_standard, route_free}",
        {
            "route_premium": route_premium,
            "route_standard": route_standard,
            "route_free": route_free,
        },
    )

    msg = mf.dotdict(score=75)
    msg = flux(msg)
    print(msg.tier)  # "standard"
    ```

### Key Paths

Conditions access nested message fields using dot notation resolved via
`message.get(key_path)`.

```
user.profile.age
request.headers.authorization
output.agent
```

If the key path does not exist in the message, its value is treated as `None`,
so `is None` checks work naturally for absent keys.

### None Checks

Use `is None` / `is not None` to branch on the presence or absence of a field.

???+ example

    ```python
    import msgflux as mf

    def transcribe(msg):
        msg.text = f"[transcript of {msg.user.audio}]"

    def ask_name(msg):
        msg.prompt = "What is your name?"

    def greet(msg):
        msg.greeting = f"Hello, {msg.user.name}!"

    # Runs transcribe only when audio is present
    flux1 = mf.Inline(
        "{user.audio is not None ? transcribe}",
        {"transcribe": transcribe},
    )

    msg1 = mf.dotdict(user=mf.dotdict(audio="clip.wav"))
    msg1 = flux1(msg1)
    print(msg1.text)  # "[transcript of clip.wav]"

    # Routes based on whether name is set
    flux2 = mf.Inline(
        "{user.name is None ? ask_name, greet}",
        {"ask_name": ask_name, "greet": greet},
    )

    msg2 = mf.dotdict(user=mf.dotdict(name=None))
    msg2 = flux2(msg2)
    print(msg2.prompt)  # "What is your name?"
    ```

### Comparison Operators

| Operator | Description |
|----------|-------------|
| `==` | Equal |
| `!=` | Not equal |
| `>` | Greater than |
| `<` | Less than |
| `>=` | Greater or equal |
| `<=` | Less or equal |
| `is None` | Value is `None` or key is absent |
| `is not None` | Value is present and not `None` |

### Type Coercion

The comparison value in the DSL is always a string literal. Before comparing,
`Inline` tries to coerce it to the type of the actual message value. This lets
you write natural conditions even when the message holds strings.

???+ example

    ```python
    import msgflux as mf

    def accept(msg):
        msg.accepted = True

    def reject(msg):
        msg.accepted = False

    def enable(msg):
        msg.action = "enabled"

    def disable(msg):
        msg.action = "disabled"

    # String "10" is coerced to int before comparison with literal 5
    flux1 = mf.Inline(
        "{count > 5 ? accept, reject}",
        {"accept": accept, "reject": reject},
    )

    msg1 = mf.dotdict(count="10")  # count is a string
    msg1 = flux1(msg1)
    print(msg1.accepted)  # True  ("10" coerced to 10, 10 > 5)

    # String "true" is coerced to bool
    flux2 = mf.Inline(
        "{enabled == true ? enable, disable}",
        {"enable": enable, "disable": disable},
    )

    msg2 = mf.dotdict(enabled="true")  # enabled is a string
    msg2 = flux2(msg2)
    print(msg2.action)  # "enabled"
    ```

If coercion fails, values are compared as strings.

### Logical Operators

Conditions can be combined with logical operators to express multi-part rules.

| Operator | Description |
|----------|-------------|
| `!` | NOT (highest precedence) |
| `&` | AND |
| `\|\|` | OR (lowest precedence) |

Precedence order (highest to lowest): `!`, `&`, `||`. Use parentheses to make
intent explicit when combining operators.

???+ example

    ```python
    import msgflux as mf

    def allow(msg):
        msg.access = "allowed"

    def deny(msg):
        msg.access = "denied"

    def grant(msg):
        msg.access = "granted"

    def block(msg):
        msg.access = "blocked"

    # AND with NOT: active AND NOT banned
    # Reads as: (active == true) AND (NOT (banned == true))
    flux1 = mf.Inline(
        "{active == true & !banned == true ? allow, deny}",
        {"allow": allow, "deny": deny},
    )

    msg1 = mf.dotdict(active=True, banned=False)
    msg1 = flux1(msg1)
    print(msg1.access)  # "allowed"

    # OR with AND grouping: parentheses make evaluation order explicit
    flux2 = mf.Inline(
        "{(plan == 'premium' & credits > 0) || trial == true ? grant, block}",
        {"grant": grant, "block": block},
    )

    msg2 = mf.dotdict(plan="premium", credits=5, trial=False)
    msg2 = flux2(msg2)
    print(msg2.access)  # "granted"
    ```

---

## 8. **While Loops**

Syntax: `@{condition}: pipeline;`

The body runs repeatedly as long as the condition holds. The condition is
re-evaluated from the current message before every iteration, so modules inside
the loop can write values that cause it to stop.

???+ example

    ```python
    import msgflux as mf

    def init(msg):
        msg.counter = 0
        msg.total = 0

    def step(msg):
        msg.counter += 1
        msg.total += msg.counter  # accumulates: 1 + 2 + 3 + 4 + 5

    def done(msg):
        msg.finished = True

    flux = mf.Inline(
        "init -> @{counter < 5}: step; -> done",
        {"init": init, "step": step, "done": done},
    )

    msg = mf.dotdict()
    msg = flux(msg)

    print(msg.counter)   # 5
    print(msg.total)     # 15
    print(msg.finished)  # True
    ```

### `max_iterations`

`max_iterations` caps how many times a while body may run. When the limit is
reached, `Inline` raises a `RuntimeError` instead of looping forever. This
makes infinite-loop bugs loud and immediate rather than causing the process to
hang silently.

???+ example

    ```python
    import msgflux as mf

    def poll(msg):
        msg.attempts = msg.get("attempts", 0) + 1

    # active stays True forever — max_iterations=5 stops it after 5 runs.
    flux = mf.Inline(
        "@{active == true}: poll;",
        {"poll": poll},
        max_iterations=5,
    )

    msg = mf.dotdict(active=True)
    try:
        flux(msg)
    except RuntimeError as e:
        print(e)
        # While loop exceeded maximum iterations (5).
        # Possible infinite loop detected. Condition: active == true
        print(msg.attempts)  # 5
    ```

!!! tip
    Set `max_iterations` to a small, intentional value when the loop represents
    a **retry budget**. This makes the limit explicit and surfaces bugs early
    instead of hanging for 1 000 iterations.

---

## 9. **Error Handling**

If a module raises an exception, `Inline` wraps it in a `RuntimeError` that
includes the module name, making it easy to trace the failure back to the
right step.

In parallel stages, `Inline` waits for all branches to complete before
reporting errors. If multiple branches fail, all failures are reported together.

```
RuntimeError: Parallel execution failed for:
  `feat_a`: ValueError('invalid input'),
  `feat_b`: TimeoutError('connection timed out')
```

For sequential steps, the error is raised immediately and execution stops.

```
RuntimeError: Execution failed for `enrich`: KeyError('text')
```

---

## 10. **Async Execution**

Use `await flux.acall(msg)` when any module in the pipeline is an async
function or an `nn.Module` with an `acall` method. The snippet below runs
natively in Jupyter and IPython, where top-level `await` is supported.

Sync modules inside an async pipeline are called directly — no thread-pool
overhead. Parallel stages run each branch as a separate coroutine via
`ascatter_gather`.

???+ example

    ```python
    import asyncio
    import msgflux as mf

    async def fetch(msg):
        await asyncio.sleep(0)  # simulate I/O
        msg.data = "fetched"

    async def process(msg):
        msg.result = msg.data.upper()

    flux = mf.Inline("fetch -> process", {"fetch": fetch, "process": process})

    msg = mf.dotdict()
    msg = await flux.acall(msg)

    print(msg.result)  # "FETCHED"
    ```

---

## 11. **Complex Workflow**

All constructs compose freely. The example below combines every DSL feature
in a single pipeline: a conditional transcription step, two parallel analysis
modules, a refinement loop, a routing branch, and a final cleanup step.

???+ example

    ```python
    import msgflux as mf

    def ingest(msg):
        msg.text = msg.get("raw_input", "incoming message")

    def transcribe(msg):
        msg.text = f"[transcript of {msg.audio}]"

    def extract_entities(msg):
        msg.entities = ["Alice", "Bob"]

    def analyze_sentiment(msg):
        msg.sentiment = "positive"
        msg.confidence = 0.5  # starts below 0.8, triggers the refinement loop

    def refine(msg):
        msg.confidence = min(msg.confidence + 0.25, 1.0)

    def priority_handler(msg):
        msg.queue = "priority"

    def standard_handler(msg):
        msg.queue = "standard"

    def bulk_handler(msg):
        msg.queue = "bulk"

    def finalize(msg):
        msg.done = True

    workflow = """
        ingest
        -> {audio is not None ? transcribe}
        -> [extract_entities, analyze_sentiment]
        -> @{confidence < 0.8}: refine;
        -> {is_urgent == true ? priority_handler,
            confidence > 0.9 ? standard_handler,
            bulk_handler}
        -> finalize
    """

    flux = mf.Inline(
        workflow,
        {
            "ingest": ingest,
            "transcribe": transcribe,
            "extract_entities": extract_entities,
            "analyze_sentiment": analyze_sentiment,
            "refine": refine,
            "priority_handler": priority_handler,
            "standard_handler": standard_handler,
            "bulk_handler": bulk_handler,
            "finalize": finalize,
        },
    )

    msg = mf.dotdict(raw_input="important call", audio="call.wav", is_urgent=True)
    msg = flux(msg)

    print(msg.text)        # "[transcript of call.wav]"
    print(msg.entities)    # ["Alice", "Bob"]
    print(msg.confidence)  # 1.0  (0.5 → 0.75 → 1.0, two refine iterations)
    print(msg.queue)       # "priority"
    print(msg.done)        # True
    ```

Execution order:

1. `ingest` — always runs
2. `transcribe` — runs because `audio` is not `None`
3. `extract_entities` and `analyze_sentiment` — run in parallel
4. `refine` — loops until `confidence >= 0.8` (capped at 1000 iterations)
5. `priority_handler` — runs because `is_urgent == true` (first match wins)
6. `finalize` — always runs

Field names used in conditions (e.g. `audio`, `confidence`, `is_urgent`)
always refer to values stored in the message passed to the pipeline.

---

## 12. **DSL Grammar**

```
pipeline     ::= step ("->" step)*

step         ::= module
               | parallel
               | conditional
               | while_loop

module       ::= IDENTIFIER

parallel     ::= "[" module ("," module)* "]"

conditional  ::= "{" cond_branch ("," cond_branch)* ("," default)? "}"

cond_branch  ::= condition "?" module

default      ::= module

while_loop   ::= "@{" condition "}:" pipeline ";"

condition    ::= logical_expr

logical_expr ::= logical_or

logical_or   ::= logical_and ("||" logical_and)*

logical_and  ::= logical_not  ("&"  logical_not)*

logical_not  ::= "!" logical_not | primary

primary      ::= "(" logical_expr ")" | comparison

comparison   ::= key_path operator value
               | key_path "is" "None"
               | key_path "is" "not" "None"

key_path     ::= IDENTIFIER ("." IDENTIFIER)*

operator     ::= "==" | "!=" | "<" | ">" | "<=" | ">="
```
