# Fuzzy Retriever

## ✦₊⁺ Overview

Fuzzy retrieval finds documents that are **similar** to a query, not just exact matches. It tolerates typos, abbreviations, transpositions, and partial strings — making it ideal for user-facing search, entity lookup, and deduplication scenarios where exact lexical matching is too strict.

msgFlux ships one fuzzy provider powered by [RapidFuzz](https://github.com/maxbachmann/RapidFuzz), a fast C-extension library for approximate string matching.

| Provider | Class | Dependency |
|----------|-------|------------|
| `rapidfuzz` | `RapidFuzzFuzzyRetriever` | `rapidfuzz` |

Install: `pip install rapidfuzz`

---

## 1. **Quick Start**

???+ example

    ```python
    import msgflux as mf

    retriever = mf.Retriever.fuzzy("rapidfuzz")

    retriever.add([
        "Alice Johnson",
        "Bob Smith",
        "Carlos Mendoza",
        "Diana Prince",
    ])

    response = retriever("Allice Jonson", top_k=2, return_score=True)

    for result in response.data[0]:
        print(f"[{result.score:.1f}] {result.data}")
    # [93.3] Alice Johnson
    # [51.4] Carlos Mendoza
    ```

---

## 2. **Parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scorer` | `"WRatio"` | Scoring function — see [Scorers](#3-scorers) |
| `top_k` | `5` | Max results returned per query |
| `threshold` | `0.0` | Minimum similarity score (0–100) to include a result |
| `return_score` | `False` | Include the similarity score in each result |

---

## 3. **Scorers**

The `scorer` controls how similarity is measured between the query and each document:

| Scorer | Best For |
|--------|----------|
| `"WRatio"` | General purpose — combines multiple strategies (default) |
| `"ratio"` | Full-string character similarity |
| `"partial_ratio"` | Query appears as a substring of the document |
| `"token_sort_ratio"` | Word-order-insensitive matching |
| `"token_set_ratio"` | Handles extra or missing words between strings |

```python
import msgflux as mf

# Best for partial name lookup
retriever = mf.Retriever.fuzzy("rapidfuzz", scorer="partial_ratio")
retriever.add(["São Paulo", "Rio de Janeiro", "Belo Horizonte"])

response = retriever("Paulo", top_k=1, return_score=True)
print(response.data[0][0].data)   # São Paulo
print(response.data[0][0].score)  # 100.0
```

---

## 4. **Adding Documents**

Call `.add()` with a list of strings before searching. Documents accumulate — multiple calls are supported:

```python
import msgflux as mf

retriever = mf.Retriever.fuzzy("rapidfuzz")

# First batch
retriever.add([
    "Aspirin 500mg",
    "Ibuprofen 400mg",
])

# Add more later
retriever.add([
    "Paracetamol 750mg",
    "Amoxicillin 250mg",
])

response = retriever("paracetamol 750", top_k=1)
print(response.data[0][0].data)  # Paracetamol 750mg
```

---

## 5. **Threshold Filtering**

`threshold` accepts a value between `0` and `100`. Results with similarity below the threshold are excluded:

???+ example "Filtering weak matches"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.fuzzy("rapidfuzz")
    retriever.add([
        "João da Silva",
        "Maria Oliveira",
        "Carlos Souza",
    ])

    # Only return results with >= 70% similarity
    response = retriever("Joao Silva", threshold=70.0, return_score=True)

    for result in response.data[0]:
        print(f"[{result.score:.1f}] {result.data}")
    # [91.4] João da Silva
    # (others excluded — score below threshold)
    ```

---

## 6. **Batch Queries**

Pass a list to search multiple queries in a single call. Results are returned in the same order:

```python
import msgflux as mf

retriever = mf.Retriever.fuzzy("rapidfuzz")
retriever.add([
    "Python",
    "JavaScript",
    "TypeScript",
    "Rust",
    "Go",
])

queries = ["pyton", "javasript"]
response = retriever(queries, top_k=1, return_score=True)

for i, query in enumerate(queries):
    result = response.data[i][0]
    print(f"{query!r} → [{result.score:.1f}] {result.data}")
# 'pyton' → [90.9] Python
# 'javasript' → [94.1] JavaScript
```

---

## 7. **Async Support**

```python
import msgflux as mf

retriever = mf.Retriever.fuzzy("rapidfuzz")
retriever.add([
    "invoice_2024_01.pdf",
    "invoice_2024_02.pdf",
    "contract_draft_v3.docx",
])

response = await retriever.acall("invioce 2024", top_k=2, return_score=True)

for result in response.data[0]:
    print(f"[{result.score:.1f}] {result.data}")
```

---

## 8. **When to use Fuzzy vs Lexical**

| Scenario | Recommended |
|----------|-------------|
| User typed a query with a typo | **Fuzzy** |
| Searching acronyms or abbreviations | **Fuzzy** (`partial_ratio`) |
| Word-order varies ("Silva João" vs "João Silva") | **Fuzzy** (`token_sort_ratio`) |
| Exact keyword matching over large corpora | **Lexical (BM25)** |
| Relevance ranking with term frequency weighting | **Lexical (BM25)** |
