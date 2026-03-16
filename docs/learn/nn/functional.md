# nn.functional

## ✦₊⁺ Overview

The `msgflux.nn.functional` module provides concurrent execution primitives inspired by MPI scatter-gather patterns and PyTorch's functional API.

### Key Features

- **Concurrent Execution**: Thread pools and async event loops for parallel processing
- **Gather Patterns**: Map, scatter, and broadcast primitives for different use cases
- **Message Passing**: `dotdict` works directly with the generic gather helpers
- **Zero Overhead**: No performance penalty for single-task execution
- **Error Handling**: Typed `TaskError` results for failed tasks — no silent `None`

### Pattern Comparison

```
MAP GATHER              SCATTER GATHER          BROADCAST GATHER
──────────────          ──────────────          ────────────────
input1 ──┐              input1 ──> f1 ──┐               ┌──> f1 ──> r1
input2 ──┼──> f ──>     input2 ──> f2 ──┼──>    input ──├──> f2 ──> r2
input3 ──┘              input3 ──> f3 ──┘               └──> f3 ──> r3

Same function           Different functions     Multiple functions
Multiple inputs         Paired inputs/funcs     Same input
```

| Pattern | When to Use |
|---------|-------------|
| `map_gather` | Apply the same function to multiple inputs |
| `scatter_gather` | Route different inputs to different functions |
| `bcast_gather` | Fan-out one input to multiple functions |

All core functions have async counterparts prefixed with `a`:

| Sync | Async |
|------|-------|
| `map_gather` | `amap_gather` |
| `scatter_gather` | `ascatter_gather` |
| `wait_for` | — |
| `wait_for_event` | `await_for_event` |
| `fire_and_forget` | `afire_and_forget` |

---

## 1. **Quick Start**

???+ example "Parallel Execution"

    ```python
    import msgflux.nn.functional as F

    def process(x):
        return x * 2

    # Run process(1), process(2), process(3) in parallel
    results = F.map_gather(process, args_list=[(1,), (2,), (3,)])
    print(results)  # (2, 4, 6)
    ```

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

Apply the same function to multiple inputs concurrently.

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

        ```python
        import msgflux.nn.functional as F

        def add(x, y):
            return x + y

        results = F.map_gather(add, args_list=[(1, 2), (3, 4), (5, 6)])
        print(results)  # (3, 7, 11)
        ```

    === "With kwargs"

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

---

### `scatter_gather`

Distribute different functions across corresponding inputs.

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

---

### `bcast_gather`

Broadcast the same arguments to multiple functions.

???+ example

    === "Basic"

        ```python
        import msgflux.nn.functional as F

        def square(x): return x * x
        def cube(x): return x * x * x
        def double(x): return x * 2

        results = F.bcast_gather([square, cube, double], 5)
        print(results)  # (25, 125, 10)
        ```

    === "Error Handling"

        ```python
        import msgflux.nn.functional as F
        from msgflux import TaskError

        def square(x): return x * x
        def fail(x): raise ValueError("Intentional error")
        def cube(x): return x * x * x

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

---

### Using `dotdict` Messages

The gather helpers work directly with `msgflux.dotdict` objects.

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

### `wait_for`

Execute a callable and wait for the result with optional timeout.

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

Wait for an `asyncio.Event` in synchronous code.

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

### `fire_and_forget`

Dispatch a task without waiting for a result.

???+ example

    === "Sync Function"

        ```python
        import msgflux.nn.functional as F

        def log_event(event_type, user_id):
            print(f"Logging: {event_type} for user {user_id}")

        # Returns immediately
        F.fire_and_forget(log_event, "login", 12345)
        print("Main thread continues...")
        ```

    === "Async Function"

        ```python
        import msgflux.nn.functional as F

        async def async_log(message):
            import asyncio
            await asyncio.sleep(1)
            print(f"[Async] {message}")

        F.fire_and_forget(async_log, "Hello from fire_and_forget")
        ```

    === "Error Handling"

        ```python
        import msgflux.nn.functional as F

        def failing_task():
            raise ValueError("This task failed!")

        # Error is logged, not raised
        F.fire_and_forget(failing_task)
        ```

!!! tip "Use Cases"
    Fire-and-forget is ideal for logging, cache updates, notifications, and non-critical side effects.

**Async version:** `afire_and_forget`
