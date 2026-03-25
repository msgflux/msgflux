# nn.Searcher

The `nn.Searcher` module provides a unified interface for information retrieval using lexical search (BM25) or web search (Wikipedia).

All code examples use the recommended import pattern:

```python
import msgflux as mf
import msgflux.nn as nn
```

## Quick Start

### AutoParams Initialization (Recommended)

This is the preferred and recommended way to define searchers in msgFlux. It promotes reusability and clear intent.

```python
import msgflux as mf
import msgflux.nn as nn

# Create the BM25 backend
bm25 = mf.Retriever.lexical("bm25")
bm25.add([
    "Python is a programming language created by Guido van Rossum",
    "JavaScript runs in the browser and on the server with Node.js",
    "Rust is fast and memory safe, created by Mozilla",
    "Go was designed at Google by Robert Griesemer",
])

class DocSearcher(nn.Searcher):
    """Searcher for technical documentation."""
    retriever = bm25
    config    = {"top_k": 3, "return_score": True}

# All parameters are already defined — just instantiate
doc_searcher = DocSearcher()

# Search
results = doc_searcher("Python programming")
print(results)
# [{'results': [{'data': 'Python is a...', 'score': 2.35}, ...], 'query': 'Python programming'}]
```

### Traditional Initialization

For quick scripts or one-off usage:

```python
searcher = nn.Searcher(
    retriever=bm25,
    config={"top_k": 3}
)
```

---

## Output Format

Searcher always returns a list with one entry per query. Each entry is a dict with `results` and `query`:

```python
[
    {
        "query": "Python programming",
        "results": [
            {"data": "Python is a programming language...", "score": 2.35},
            {"data": "JavaScript runs in the browser...", "score": 0.0},
        ]
    }
]
```

- `data`: The retrieved document content.
- `score`: Relevance score (present when `return_score=True`, otherwise `None`).
- `query`: The original query string.

---

## Single Search vs Multi Search

=== "Single"

    Pass a string to search for a single query:

    ```python
    searcher = nn.Searcher(retriever=bm25, config={"top_k": 2, "return_score": True})

    result = searcher("Python programming")
    # [{'results': [...], 'query': 'Python programming'}]
    ```

=== "Multi"

    Pass a list of strings to search multiple queries at once:

    ```python
    results = searcher(["Python programming", "Rust memory safe"])
    # [
    #     {'results': [...], 'query': 'Python programming'},
    #     {'results': [...], 'query': 'Rust memory safe'}
    # ]
    ```

---

## Response Templates

Use Jinja2 templates to format search results into readable strings. When a `response` template is set, the return type changes from `list` to `str`.

=== "Scored"

    ```python
    class ScoredSearcher(nn.Searcher):
        """Searcher with formatted output including scores."""
        retriever = bm25
        config    = {"top_k": 2, "return_score": True}
        templates = {
            "response": """Query: {{ query }}
    {% for r in results %}  {{ loop.index }}. {{ r.data }} (score: {{ r.score }})
    {% endfor %}"""
        }

    searcher = ScoredSearcher()
    result = searcher("Python programming")
    # "Query: Python programming
    #   1. Python is a programming language... (score: 2.35)
    #   2. JavaScript runs in the browser... (score: 0.0)"
    ```

=== "Context-Only"

    Extract just the document content for RAG pipelines:

    ```python
    class ContextSearcher(nn.Searcher):
        """Extracts raw document content for RAG pipelines."""
        retriever = bm25
        config    = {"top_k": 3}
        templates = {
            "response": "{% for r in results %}{{ r.data }}\n{% endfor %}"
        }

    searcher = ContextSearcher()
    context = searcher("Python")
    # "Python is a programming language created by Guido van Rossum\n..."
    ```

=== "Multi-Query"

    Each result is formatted individually and joined with double newlines:

    ```python
    result = searcher(["Python", "Rust"])
    # "Query: Python
    #   1. Python is a programming language... (score: 1.17)
    #   2. JavaScript runs in the browser... (score: 0.0)
    #
    # Query: Rust
    #   1. Rust is fast and memory safe... (score: 1.23)
    #   2. Python is a programming language... (score: 0.0)"
    ```

---

## Configuration Options

The `config` dict controls retrieval behavior:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `top_k` | int | 4 | Maximum results per query |
| `threshold` | float | 0.0 | Minimum score to include a result |
| `return_score` | bool | False | Include relevance scores in output |
| `dict_key` | str | - | Key to extract from `List[Dict]` inputs |

```python
class PrecisionSearcher(nn.Searcher):
    """Searcher with strict filtering."""
    retriever = bm25
    config    = {
        "top_k": 5,
        "threshold": 0.5,
        "return_score": True,
    }

searcher = PrecisionSearcher()
```

---

## Message Fields and Response Mode

### Message Field Mapping

Use `Message` objects for structured input/output. This decouples your searcher from the specific data structure.

```python
class StructuredSearcher(nn.Searcher):
    """Searcher that reads queries from a Message field."""
    retriever      = bm25
    message_fields = {"task_inputs": "query.user"}
    config         = {"top_k": 3}

searcher = StructuredSearcher()

msg = mf.Message()
msg.set("query.user", "What is dependency injection?")

result = searcher(msg)
```

### Response Mode

Controls where results are written:

```python
# Default (None): returns results directly
searcher = nn.Searcher(retriever=bm25)
result = searcher("query")  # Returns list

# Path mode: writes to message field, returns None
class PipelineSearcher(nn.Searcher):
    """Writes results into message.context for downstream modules."""
    retriever      = bm25
    message_fields = {"task_inputs": "question"}
    response_mode  = "context"
    config         = {"top_k": 3}

searcher = PipelineSearcher()

msg = mf.Message()
msg.question = "Python programming"
searcher(msg)  # Returns None
print(msg.context)  # Results are here
```

---

## Creating Searcher Hierarchies

Build specialized searchers through inheritance to share configuration.

```python
# Base searcher for all documentation
class BaseDocSearcher(nn.Searcher):
    """Base searcher for documentation with common config."""
    retriever = bm25

# High precision searcher for critical docs
class CriticalDocSearcher(BaseDocSearcher):
    """High precision searcher for critical documentation."""
    config = {"top_k": 3, "threshold": 2.0}  # Strict

# Broad searcher for general exploration
class ExploratorySearcher(BaseDocSearcher):
    """Broad searcher for exploratory searches."""
    config = {"top_k": 10, "threshold": 0.0}  # Permissive

critical    = CriticalDocSearcher()
exploratory = ExploratorySearcher()
```

---

## Integration with Agents

A Searcher with a docstring (used as `description`) and default `annotations` can be plugged directly as a tool into an Agent — no wrapper function needed.

```python
bm25 = mf.Retriever.lexical("bm25")
bm25.add(documents)

class KBSearcher(nn.Searcher):
    """Search the company knowledge base for relevant documents."""
    retriever = bm25
    config    = {"top_k": 5}
    templates = {"response": "{% for r in results %}{{ r.data }}\n{% endfor %}"}

class SupportAgent(nn.Agent):
    """Customer support agent with access to knowledge base."""
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    tools = [KBSearcher]

support_agent = SupportAgent()
response = support_agent("How do I reset my password?")
```

The docstring becomes the tool description, the class name becomes the tool name, and the default annotations (`query: str -> str`) define the tool schema — all via AutoParams.
