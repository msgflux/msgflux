# Retrievers

## ✦₊⁺ Overview

`Retriever` provides a unified interface for fetching relevant content from different sources. All retrievers return a consistent `dotdict` response and support both synchronous and async execution.

There are two retriever families:

| Family | Providers | Use Case |
|--------|-----------|----------|
| **Lexical** | `bm25`, `bm25s`, `rank_bm25` | Search a local document corpus by keyword relevance |
| **Web** | `wikipedia` | Fetch content from external sources at query time |

---

## 1. **Quick Start**

???+ example "Lexical — BM25"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.bm25()

    retriever.add([
        "Python is a high-level programming language.",
        "Machine learning is a subset of artificial intelligence.",
        "The Eiffel Tower is located in Paris.",
    ])

    response = retriever("What is machine learning?")

    for result in response.data[0].results:
        print(result.data)
    # Machine learning is a subset of artificial intelligence.
    ```

???+ example "Web — Wikipedia"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.wikipedia(summary=2)

    response = retriever("quantum entanglement")

    for result in response.data[0].results:
        print(result.data.title)
        print(result.data.content)
    ```

---

## 2. **Response Format**

All retrievers return a `dotdict` with a consistent top-level structure:

```python
response.response_type  # "lexical_search" or "web_search"
response.data           # list — one entry per query
response.data[0].results        # list of results for the first query
response.data[0].results[0].data  # the retrieved content
```

### Lexical response

```python
response = retriever("my query", return_score=True)

result = response.data[0].results[0]
print(result.data)   # "Document text..."
print(result.score)  # 3.14  (BM25 score, only if return_score=True)
```

### Web response

```python
response = retriever("Python programming")

result = response.data[0].results[0]
print(result.data.title)    # "Python (programming language)"
print(result.data.content)  # "Python\n\nPython is..."
```

---

## 3. **Batch Queries**

Pass a list to search multiple queries in a single call. Results are returned in the same order:

```python
import msgflux as mf

retriever = mf.Retriever.bm25()
retriever.add(["Doc A about Python", "Doc B about Java", "Doc C about Rust"])

queries = ["Python language", "systems programming"]
response = retriever(queries)

for i, query in enumerate(queries):
    print(f"\n--- {query} ---")
    for result in response.data[i].results:
        print(result.data)
```

---

## 4. **Async Support**

All retrievers expose `.acall()` for async usage:

```python
import msgflux as mf

retriever = mf.Retriever.wikipedia(summary=3)

response = await retriever.acall("artificial intelligence", top_k=2)

for result in response.data[0].results:
    print(result.data.title)
```
