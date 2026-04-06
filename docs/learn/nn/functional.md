# Functional

## ✦₊⁺ Overview

The `msgflux.nn.functional` module provides concurrent execution primitives inspired by MPI scatter-gather patterns and PyTorch's functional API. While `nn.Module` encapsulates stateful components, `nn.functional` offers stateless operations that coordinate how those components run in parallel.

In practice, this means you can take any callable — a plain function, an `nn.Agent`, or any `nn.Module` — and run multiple instances concurrently with a single function call. The module handles thread management, error collection, and telemetry automatically, so you focus on **what** to run rather than **how** to parallelize it.

### Key Features

- **Concurrent Execution**: Thread pools and async event loops for parallel processing
- **Gather Patterns**: Map, scatter, and broadcast primitives for different use cases
- **Error Handling**: Typed `TaskError` results for failed tasks

### Pattern Comparison

The three gather patterns cover the most common parallel execution shapes:

```
MAP GATHER                SCATTER GATHER            BROADCAST GATHER
────────────────          ────────────────          ────────────────
input1 ──┐    ┌──> r1    input1 ──> f1 ──┐──> r1           ┌──> f1 ──> r1
input2 ──┼──> f ──> r2   input2 ──> f2 ──┼──> r2   input ──├──> f2 ──> r2
input3 ──┘    └──> r3    input3 ──> f3 ──┘──> r3           └──> f3 ──> r3

Same function             Different functions       Multiple functions
Multiple inputs           Paired inputs/funcs       Same input
```

All core functions have async counterparts prefixed with `a`. Use the sync versions for scripts and CLI tools; use the async versions inside `async def` contexts like web frameworks and pipelines:

| Sync | Async | When to Use |
|------|-------|-------------|
| `map_gather` | `amap_gather` | Same function, multiple inputs |
| `scatter_gather` | `ascatter_gather` | Different functions, paired inputs |
| `bcast_gather` | `abcast_gather` | Multiple functions, same input |
| `wait_for` | — | Execute a callable with optional timeout |
| `wait_for_event` | `await_for_event` | Block until an `asyncio.Event` is set |
| `spawn` | `aspawn` | Fire-and-forget task dispatch |

---

## 1. **Quick Start**

The fastest way to use the functional API is through `map_gather`. Pass a function and a list of argument tuples — each tuple becomes a concurrent call. The results come back in order, as a tuple, regardless of which call finishes first:

???+ example "Parallel Execution"

    ```python
    import msgflux.nn.functional as F

    def process(x):
        return x * 2

    # Run process(1), process(2), process(3) in parallel
    results = F.map_gather(process, args_list=[(1,), (2,), (3,)])
    print(results)  # (2, 4, 6)
    ```

For async code, every gather function has an `a`-prefixed counterpart that returns an awaitable. The API is identical — just `await` the result:

???+ example "Async"

    ```python
    import msgflux.nn.functional as F

    async def async_square(x):
        return x * x

    results = await F.amap_gather(async_square, args_list=[(2,), (3,), (4,)])
    print(results)  # (4, 9, 16)
    ```

---

## 2. **Gather Functions**

### `map_gather`

`map_gather` applies **the same function** to **multiple inputs** concurrently. This is the most common pattern — any time you need to process a batch of items through the same logic, `map_gather` turns sequential iteration into parallel execution.

Under the hood, sync calls are dispatched to a thread pool, while async calls use `asyncio.gather`. Results are always returned in the same order as the input, so you can zip them back together.

???+ example

    === "Basic"

        ```python
        import msgflux.nn.functional as F

        def square(x):
            return x * x

        results = F.map_gather(square, args_list=[(2,), (3,), (4,)])
        print(results)  # (4, 9, 16)
        ```

    === "With Multiple Arguments"

        Each tuple in `args_list` is unpacked as positional arguments to the function:

        ```python
        import msgflux.nn.functional as F

        def add(x, y):
            return x + y

        results = F.map_gather(add, args_list=[(1, 2), (3, 4), (5, 6)])
        print(results)  # (3, 7, 11)
        ```

    === "With kwargs"

        Use `kwargs_list` when you need to pass keyword arguments. Each dict in the list is paired with the corresponding tuple in `args_list`:

        ```python
        import msgflux.nn.functional as F

        def multiply(x, y=2):
            return x * y

        results = F.map_gather(
            multiply,
            args_list=[(1,), (3,), (5,)],
            kwargs_list=[{"y": 3}, {"y": 4}, {"y": 5}]
        )
        print(results)  # (3, 12, 25)
        ```

    === "With Timeout"

        Set a global `timeout` (in seconds) to prevent any single task from blocking indefinitely. If a task exceeds the timeout, it returns a `TaskError` instead of raising:

        ```python
        import msgflux.nn.functional as F
        import time

        def slow_task(x):
            time.sleep(0.5)
            return x * x

        results = F.map_gather(
            slow_task,
            args_list=[(2,), (3,), (4,)],
            timeout=1.0
        )
        ```

**Async version:** `amap_gather`

Since Agents are callable (`agent(input)` calls `forward`), you can pass them directly to `map_gather`. This is especially powerful for batch inference — classifying, translating, or summarizing multiple inputs concurrently through the same agent:

???+ example "With Agents"

    === "Sync — classify multiple inputs"

        ```python
        import msgflux as mf
        import msgflux.nn as nn
        import msgflux.nn.functional as F

        class Classifier(nn.Agent):
            model = mf.Model.chat_completion("openai/gpt-4.1-mini")
            instructions = "Classify the sentiment as positive, negative, or neutral. Reply with one word."

        agent = Classifier()

        reviews = [
            "This product is amazing!",
            "Terrible experience, never again.",
            "It's okay, nothing special.",
        ]

        results = F.map_gather(agent, args_list=[(r,) for r in reviews])
        for review, result in zip(reviews, results):
            print(f"{review[:30]}... → {result}")
        # This product is amazing!...   → positive
        # Terrible experience, never ... → negative
        # It's okay, nothing special.... → neutral
        ```

    === "Async"

        ```python
        results = await F.amap_gather(
            agent,
            args_list=[(r,) for r in reviews],
        )
        ```

---

### `scatter_gather`

`scatter_gather` pairs **different functions** with **corresponding inputs** and runs them all concurrently. The i-th function receives the i-th input. This is the right pattern when each task requires a different handler — for example, routing a summarization job to one agent and a translation job to another, all in one parallel call.

The function list and `args_list` must have the same length. Each function-input pair runs independently, and results come back in order.

???+ example

    === "Basic"

        ```python
        import msgflux.nn.functional as F

        def double(x): return x * 2
        def triple(x): return x * 3
        def square(x): return x ** 2

        results = F.scatter_gather(
            [double, triple, square],
            args_list=[(5,), (5,), (5,)]
        )
        print(results)  # (10, 15, 25)
        ```

    === "Different Inputs"

        Each function gets its own distinct input:

        ```python
        import msgflux.nn.functional as F

        def double(x): return x * 2
        def triple(x): return x * 3
        def square(x): return x ** 2

        results = F.scatter_gather(
            [double, triple, square],
            args_list=[(2,), (3,), (4,)]
        )
        print(results)  # (4, 9, 16)
        ```

    === "With kwargs Only"

        You can omit `args_list` and pass only `kwargs_list`. Each dict maps to the corresponding function:

        ```python
        import msgflux.nn.functional as F

        def greet(name="World"):
            return f"Hello, {name}"

        def farewell(person):
            return f"Goodbye, {person}"

        results = F.scatter_gather(
            [greet, greet, farewell],
            kwargs_list=[{}, {"name": "Earth"}, {"person": "Commander"}]
        )
        print(results)  # ("Hello, World", "Hello, Earth", "Goodbye, Commander")
        ```

**Async version:** `ascatter_gather`

With Agents, `scatter_gather` lets you run completely different AI tasks in parallel — each agent specializes in something different, and each receives its own input:

???+ example "With Agents"

    === "Sync — different agents, different inputs"

        ```python
        import msgflux as mf
        import msgflux.nn as nn
        import msgflux.nn.functional as F

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        class Summarizer(nn.Agent):
            model = model
            instructions = "Summarize the text in one sentence."

        class Translator(nn.Agent):
            model = model
            instructions = "Translate the text to Portuguese."

        summarizer = Summarizer()
        translator = Translator()

        results = F.scatter_gather(
            [summarizer, translator],
            args_list=[
                ("Explain how neural networks learn from data...",),
                ("The weather is beautiful today.",),
            ],
        )

        summary, translation = results
        ```

    === "Async"

        ```python
        results = await F.ascatter_gather(
            [summarizer, translator],
            args_list=[
                ("Explain how neural networks learn from data...",),
                ("The weather is beautiful today.",),
            ],
        )
        ```

---

### `bcast_gather`

`bcast_gather` sends **the same arguments** to **multiple functions** and gathers the results. This is the fan-out pattern — useful when you have a single piece of data that needs to be processed by several independent handlers simultaneously.

A common use case with Agents is analyzing one document from multiple perspectives at once: summarize it, translate it, and extract keywords — all in parallel, all from the same input. The total time is roughly the slowest handler, not the sum.

???+ example

    === "Basic"

        ```python
        import msgflux.nn.functional as F

        def square(x):
            return x * x

        def cube(x):
            return x * x * x

        def double(x):
            return x * 2

        results = F.bcast_gather([square, cube, double], 5)
        print(results)  # (25, 125, 10)
        ```

    === "Error Handling"

        When a task fails, the gather functions **do not raise** — instead, the failed position contains a `TaskError` object with the original exception. This lets the other tasks complete successfully and gives you fine-grained control over error handling:

        ```python
        import msgflux.nn.functional as F
        from msgflux import TaskError

        def square(x):
            return x * x

        def fail(x):
            raise ValueError("Intentional error")

        def cube(x):
            return x * x * x

        # Failed tasks return a TaskError instance — no exception is raised
        results = F.bcast_gather([square, fail, cube], 2)
        print(results)  # (4, TaskError(index=1, ...), 8)

        for i, result in enumerate(results):
            if isinstance(result, TaskError):
                print(f"Task {i} failed: {result.exception}")
            else:
                print(f"Task {i} result: {result}")
        ```

    === "With kwargs"

        Keyword arguments are broadcast to all functions, just like positional arguments:

        ```python
        import msgflux.nn.functional as F

        def fetch_user(user_id):
            return {"id": user_id, "name": f"User {user_id}"}

        def fetch_posts(user_id):
            return [f"Post {i}" for i in range(3)]

        def fetch_comments(user_id):
            return [f"Comment {i}" for i in range(5)]

        user, posts, comments = F.bcast_gather(
            [fetch_user, fetch_posts, fetch_comments],
            user_id=123
        )
        ```

**Async version:** `abcast_gather`

???+ example "With Agents"

    === "Sync — same input, multiple agents"

        Three agents process the same text in parallel. Each agent has a different specialization but receives the same input:

        ```python
        import msgflux as mf
        import msgflux.nn as nn
        import msgflux.nn.functional as F

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        class Summarizer(nn.Agent):
            model = model
            instructions = "Summarize the text in one sentence."

        class Translator(nn.Agent):
            model = model
            instructions = "Translate the text to Portuguese."

        class KeywordExtractor(nn.Agent):
            model = model
            instructions = "Extract 3 keywords from the text, comma-separated."

        text = "Quantum computing uses qubits that can exist in superposition..."

        summary, translation, keywords = F.bcast_gather(
            [Summarizer(), Translator(), KeywordExtractor()], text
        )
        ```

    === "Async"

        ```python
        summary, translation, keywords = await F.abcast_gather(
            [summarizer, translator, extractor], text
        )
        ```

---

### Using `dotdict` Messages

The gather functions work directly with `msgflux.dotdict` objects, which enables a message-passing style where concurrent functions mutate a shared or individual message. This is useful for building enrichment pipelines — each function adds a different field to the message.

With `scatter_gather`, each function gets its own `dotdict`, so there's no risk of conflicts:

???+ example "scatter_gather with messages"

    ```python
    import msgflux as mf
    import msgflux.nn.functional as F

    def process_user(msg):
        msg.type = "user"
        msg.processed = True

    def process_admin(msg):
        msg.type = "admin"
        msg.permissions = ["read", "write", "delete"]

    def process_guest(msg):
        msg.type = "guest"
        msg.permissions = ["read"]

    msg1 = mf.dotdict({"id": 1, "name": "Alice"})
    msg2 = mf.dotdict({"id": 2, "name": "Bob"})
    msg3 = mf.dotdict({"id": 3, "name": "Charlie"})

    F.scatter_gather(
        [process_user, process_admin, process_guest],
        args_list=[(msg1,), (msg2,), (msg3,)]
    )

    print(msg1.type)  # user
    print(msg2.type)  # admin
    print(msg3.type)  # guest
    ```

With `bcast_gather`, all functions share the **same** `dotdict` object. This is powerful for enrichment — multiple functions add different fields to the same message in parallel. The key rule is that each function must write to **disjoint paths** to avoid race conditions:

???+ example "bcast_gather with a shared message"

    ```python
    import msgflux as mf
    import msgflux.nn.functional as F
    from datetime import datetime
    from msgflux import TaskError

    def add_timestamp(msg):
        msg.timestamp = datetime.now().isoformat()

    def add_metadata(msg):
        msg.set("metadata.version", "1.0")
        msg.set("metadata.source", "api")

    def validate(msg):
        msg.validated = True

    message = mf.dotdict({"data": "important"})
    results = F.bcast_gather([add_timestamp, add_metadata, validate], message)

    if any(isinstance(r, TaskError) for r in results):
        raise RuntimeError("One of the parallel steps failed")

    print(message.timestamp)         # 2024-01-15T10:30:00.123456
    print(message.metadata.version)  # 1.0
    print(message.validated)         # True
    ```

!!! warning "Race Conditions"
    Parallel modules share the same `dotdict`. Write to **disjoint paths** — modifying the same key from two concurrent functions produces unpredictable results.

---

## 3. **Utility Functions**

Beyond the gather patterns, the functional module provides lower-level utilities for bridging sync/async boundaries and dispatching fire-and-forget tasks.

### `wait_for`

`wait_for` executes a callable and blocks until the result is ready, with an optional timeout. It handles both sync and async functions transparently — if you pass an `async def`, it spins up an event loop automatically. This is useful when you need to call an async function from sync code without managing the event loop yourself:

???+ example

    === "Sync Function"

        ```python
        import msgflux.nn.functional as F

        def slow_computation(x):
            import time
            time.sleep(0.1)
            return x * x

        result = F.wait_for(slow_computation, 5)
        print(result)  # 25
        ```

    === "Async Function"

        ```python
        import msgflux.nn.functional as F

        async def async_task(x):
            return x * 2

        # Runs async function in sync context
        result = F.wait_for(async_task, 3)
        print(result)  # 6
        ```

    === "With Timeout"

        ```python
        import msgflux.nn.functional as F

        result = F.wait_for(slow_computation, 10, timeout=0.5)
        ```

---

### `wait_for_event`

`wait_for_event` bridges `asyncio.Event` into synchronous code. It blocks the calling thread until the event is set, without busy-waiting. This is useful in mixed sync/async systems where a background async task needs to signal completion to a synchronous caller:

???+ example

    ```python
    import msgflux.nn.functional as F
    import asyncio
    import threading
    import time

    event = asyncio.Event()

    def set_event_later():
        time.sleep(0.1)
        loop = asyncio.get_event_loop()
        loop.call_soon_threadsafe(event.set)

    thread = threading.Thread(target=set_event_later)
    thread.start()

    F.wait_for_event(event)  # Blocks until event is set
    print("Event was set!")
    ```

**Async version:** `await_for_event`

---

### `spawn`

`spawn` dispatches a task without waiting for a result — fire-and-forget. The function runs in a background thread (sync) or as a detached coroutine (async). Errors are logged but never raised to the caller, making `spawn` safe for non-critical side effects like logging, cache warming, and notifications:

???+ example

    === "Sync Function"

        ```python
        import msgflux.nn.functional as F

        def log_event(event_type, user_id):
            print(f"Logging: {event_type} for user {user_id}")

        # Returns immediately
        F.spawn(log_event, "login", 12345)
        print("Main thread continues...")
        ```

    === "Async Function"

        ```python
        import msgflux.nn.functional as F

        async def async_log(message):
            import asyncio
            await asyncio.sleep(1)
            print(f"[Async] {message}")

        F.spawn(async_log, "Hello from spawn")
        ```

    === "Error Handling"

        ```python
        import msgflux.nn.functional as F

        def failing_task():
            raise ValueError("This task failed!")

        # Error is logged, not raised
        F.spawn(failing_task)
        ```

!!! tip "Use Cases"
    Spawn is ideal for logging, cache updates, notifications, and non-critical side effects.

**Async version:** `aspawn`
