# nn.Searcher

## ✦₊⁺ Overview

The `nn.Searcher` module provides a unified interface for information retrieval using lexical search (BM25), fuzzy search (RapidFuzz), or web search (Wikipedia).

---

## 1. **Quick Start**

!!! info "Initialization styles"

    === "Declarative (recommended)"

        ```python
        import msgflux as mf
        import msgflux.nn as nn

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

        searcher = DocSearcher()
        results = searcher("Python programming")
        # [{'results': [{'data': 'Python is a...', 'score': 2.35}, ...], 'query': 'Python programming'}]
        ```

    === "Fuzzy"

        ```python
        import msgflux as mf
        import msgflux.nn as nn

        fuzzy = mf.Retriever.fuzzy("rapidfuzz")
        fuzzy.add([
            "Alice Johnson",
            "Bob Smith",
            "Carlos Mendoza",
            "Diana Prince",
        ])

        class NameSearcher(nn.Searcher):
            """Search contacts by approximate name match."""
            retriever = fuzzy
            config    = {"top_k": 2, "threshold": 60.0, "return_score": True}

        searcher = NameSearcher()
        results = searcher("Allice Jonson")
        # [{'results': [{'data': 'Alice Johnson', 'score': 93.3}], 'query': 'Allice Jonson'}]
        ```

    === "Direct"

        ```python
        searcher = nn.Searcher(
            retriever=bm25,
            config={"top_k": 3}
        )
        ```

---

## 2. **Output Format**

Searcher always returns a list with one entry per query:

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

- `data` — the retrieved document content
- `score` — relevance score, only present when `return_score=True`
- `query` — the original query string

---

## 3. **Search Modes**

!!! info "Single vs multi-query"

    === "Single"

        ```python
        result = searcher("Python programming")
        # [{'results': [...], 'query': 'Python programming'}]
        ```

    === "Multi"

        Pass a list to search multiple queries at once:

        ```python
        results = searcher(["Python programming", "Rust memory safe"])
        # [
        #     {'results': [...], 'query': 'Python programming'},
        #     {'results': [...], 'query': 'Rust memory safe'}
        # ]
        ```

---

## 4. **Configuration**

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
```

---

## 5. **Response Templates**

Use Jinja2 templates to format results into readable strings. When a `response` template is set, the return type changes from `list` to `str`.

!!! info "Template examples"

    === "Scored"

        ```python
        class ScoredSearcher(nn.Searcher):
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

    === "Context-only"

        Extract raw document content for RAG pipelines:

        ```python
        class ContextSearcher(nn.Searcher):
            retriever = bm25
            config    = {"top_k": 3}
            templates = {
                "response": "{% for r in results %}{{ r.data }}\n{% endfor %}"
            }

        searcher = ContextSearcher()
        context = searcher("Python")
        # "Python is a programming language created by Guido van Rossum\n..."
        ```

    === "Multi-query"

        Each result is formatted individually and joined with double newlines:

        ```python
        result = searcher(["Python", "Rust"])
        # "Query: Python
        #   1. Python is a programming language... (score: 1.17)
        #
        # Query: Rust
        #   1. Rust is fast and memory safe... (score: 1.23)"
        ```

---

## 6. **Message Fields & Response Mode**

!!! info "Structured input/output"

    === "Message Field Mapping"

        ```python
        class StructuredSearcher(nn.Searcher):
            retriever      = bm25
            message_fields = {"task": "query.user"}
            config         = {"top_k": 3}

        searcher = StructuredSearcher()

        msg = mf.dotdict()
        msg.query = mf.dotdict(user="What is dependency injection?")
        result = searcher(msg)
        ```

    === "Response Mode"

        Write results directly to a message field for pipeline composition:

        ```python
        class PipelineSearcher(nn.Searcher):
            """Writes results into msg.context for downstream modules."""
            retriever      = bm25
            message_fields = {"task": "question"}
            response_mode  = "context"
            config         = {"top_k": 3}

        searcher = PipelineSearcher()

        msg = mf.dotdict(question="Python programming")
        searcher(msg)   # returns None
        print(msg.context)  # results are here
        ```

    === "Hierarchies"

        Share configuration across related searchers via inheritance:

        ```python
        class BaseDocSearcher(nn.Searcher):
            retriever = bm25

        class CriticalDocSearcher(BaseDocSearcher):
            config = {"top_k": 3, "threshold": 2.0}  # strict

        class ExploratorySearcher(BaseDocSearcher):
            config = {"top_k": 10, "threshold": 0.0}  # permissive
        ```

---

## 7. **Integration with Agents**

A `Searcher` with a docstring can be plugged directly as a tool into an `Agent` — no wrapper needed. The docstring becomes the tool description, the class name the tool name, and the default annotations define the schema.

!!! info "Searcher as Agent tool"

    ```python
    import msgflux as mf
    import msgflux.nn as nn

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
