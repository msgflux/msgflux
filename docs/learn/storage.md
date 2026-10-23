# Storage Integration

## ✦₊⁺ Overview

msgFlux does not ship a general-purpose database abstraction. The application
owns its storage integration: it chooses the client or database, configures
connections, defines schemas and migrations, manages transactions, and decides
how data is backed up or retained.

This page describes the v1 public API removal and shows the supported pattern:
call your own store from plain Python code, then expose that code as a tool or
use it directly in your application.

---

## 1. **What Changed in v1**

The generic database integration API was removed:

- `mf.DB` is no longer exported.
- `msgflux.data.dbs` and its provider registry were removed.
- The DB-backed `mf.response_cache` decorator and `msgflux.cache` module were
  removed. Model response caching remains available through `enable_cache`.
- Available DB types, response helpers, and provider implementations
  (`cachetools`, `diskcache`, and `faiss`) were removed with that package.
- `nn.Searcher` no longer accepts a removed `VectorDB` object. Pass one of the
  supported retriever types instead.

There is no compatibility alias or deprecation shim. Code that imports
`mf.DB` or `msgflux.data.dbs` fails immediately and should initialize its own
storage client instead.

!!! note "Runtime durability is separate"
    The removal applies to the generic `data.dbs` integration layer, not to
    runtime persistence. Checkpoints, task journals, approvals, the agent
    inbox, and service cache ownership remain supported through
    `CheckpointStore`, `Store`, the task stores, and their SQLite
    implementations. See [Agent Resources](nn/agent/resources.md) and
    [Background Tasks](nn/agent/tools/background-tasks.md).

---

## 2. **Application-Owned Storage**

The following example uses Python's standard-library SQLite client. The
application owns the database file, schema, and connection lifecycle. msgFlux
only sees a normal Python function.

```python
# app/storage.py
import sqlite3
from contextlib import closing


def find_order(order_id: str) -> str:
    """Look up an order status in the application's database.

    Args:
        order_id: Application order identifier.

    Returns:
        The order status, or a not-found message.
    """
    with closing(sqlite3.connect("orders.sqlite3")) as connection:
        row = connection.execute(
            "SELECT status FROM orders WHERE id = ?",
            (order_id,),
        ).fetchone()

    return row[0] if row else f"Order {order_id} not found."
```

This function calls the user-managed store directly. No msgFlux database
provider, registry entry, or vendor-specific dependency is required.

The application can expose the same function as a tool:

```python
import msgflux as mf
import msgflux.nn as nn

from app.storage import find_order


class SupportAssistant(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    system_prompt = "Use the order lookup tool when the user asks about an order."
    tools = [find_order]


assistant = SupportAssistant()
```

The model decides when to call `find_order`; msgFlux validates and executes the
callable. The application remains responsible for the database connection,
query parameters, error handling, access control, and data retention.

If your code previously created a DB object through `mf.DB`, replace it with the
client or repository your application already uses:

```python
# Removed: provider registry and framework-owned configuration
# cache = mf.DB.kv("cachetools", ttl=300)

# Supported: initialize and own the dependency in application code
from cachetools import TTLCache

cache = TTLCache(maxsize=1_000, ttl=300)
```

The second form is ordinary application code. msgFlux does not need to know
which storage library or vendor you choose.

---

## 3. **Custom Retrievers**

`nn.Searcher` works with the supported retriever families. If an application
wants semantic search over its own store, it can implement the retriever
interface in application code and pass that object to `Searcher`:

```python
from msgflux.core.dotdict import dotdict
from msgflux.data.retrievers.types import SemanticRetriever
from msgflux.nn.modules.searcher import Searcher


class MyStoreRetriever(SemanticRetriever):
    def __init__(self, repository):
        self.repository = repository

    def __call__(self, queries, top_k=4, **kwargs):
        if isinstance(queries, str):
            queries = [queries]

        data = []
        for query in queries:
            results = [
                dotdict({"data": item, "score": score})
                for item, score in self.repository.search(query, limit=top_k)
            ]
            data.append(dotdict({"results": results}))

        return dotdict({"response_type": "semantic_search", "data": data})


searcher = Searcher(retriever=MyStoreRetriever(my_repository))
```

The adapter belongs to the application. It keeps the existing `Searcher`
pipeline while leaving connection management and search semantics under the
application's control.

---

## 4. **Guidance**

- Choose and configure storage outside msgFlux.
- Keep credentials in the application's normal secret-management path; do not
  put them in model prompts or tool arguments.
- Validate and parameterize queries before accessing the store.
- Expose narrow operations as tools rather than a generic "run SQL" tool.
- Define transaction, retry, migration, and backup behavior in the application.
- Use msgFlux runtime stores when you need durable checkpoints, approvals,
  inbox delivery, task records, or service cache behavior.
