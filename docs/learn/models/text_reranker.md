# Text Reranker

The `text_reranker` model reorders a list of documents by relevance to a query. It is typically the last step in a retrieval pipeline, applied after an initial fast retrieval to improve precision before passing context to an LLM.

!!! info "Dependencies"
    See [Dependency Management](../../dependency-management.md) for the complete provider matrix.

## ✦₊⁺ Overview

Rerankers score each (query, document) pair directly, producing a relevance score that is more accurate than embedding similarity alone. They enable:

- **RAG Precision**: Select the most relevant chunks before LLM context
- **Search Quality**: Improve ranking over BM25 or vector search results
- **Multilingual Reranking**: Rank documents across different languages

## 1. **Quick Start**

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.text_reranker("jinaai/jina-reranker-v3")

    documents = [
        "Python is a general-purpose programming language.",
        "The Eiffel Tower is located in Paris, France.",
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are inspired by the human brain."
    ]

    results = model(
        query="What is machine learning?",
        documents=documents
    ).consume()

    for r in results:
        print(f"[{r['relevance_score']:.3f}] {documents[r['index']]}")
    # [ 0.452] Machine learning is a subset of artificial intelligence.
    # [-0.059] Neural networks are inspired by the human brain.
    # [-0.086] Python is a general-purpose programming language.
    # [-0.184] The Eiffel Tower is located in Paris, France.
    ```

## 2. **Supported Providers**

### JinaAI

???+ example

    === "jina-reranker-v3 (latest)"

        ```python
        import msgflux as mf

        # Latest model — raw logit scores, can be negative
        model = mf.Model.text_reranker("jinaai/jina-reranker-v3")
        ```

    === "jina-reranker-v2 (multilingual)"

        ```python
        import msgflux as mf

        # Normalized probability scores in [0, 1]
        model = mf.Model.text_reranker("jinaai/jina-reranker-v2-base-multilingual")
        ```

### vLLM (self-hosted)

???+ example

    === "BAAI BGE (recommended)"

        ```python
        import msgflux as mf

        # BAAI/bge-reranker-v2-m3 — multilingual, simplest vLLM setup
        # vllm serve BAAI/bge-reranker-v2-m3 --task score
        model = mf.Model.text_reranker(
            "vllm/BAAI/bge-reranker-v2-m3",
            base_url="http://localhost:8000/v1"
        )
        ```

    === "Qwen3 Reranker"

        ```python
        import msgflux as mf

        # Use the pre-converted variant — no hf_overrides needed
        # vllm serve tomaarsen/Qwen3-Reranker-0.6B-seq-cls --task score
        model = mf.Model.text_reranker(
            "vllm/tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
            base_url="http://localhost:8000/v1"
        )
        ```

    === "Jina (vLLM)"

        ```python
        import msgflux as mf

        # vllm serve jinaai/jina-reranker-m0 --task score
        model = mf.Model.text_reranker(
            "vllm/jinaai/jina-reranker-m0",
            base_url="http://localhost:8000/v1"
        )
        ```

## 3. **Response Format**

`consume()` returns a list of results sorted by `relevance_score` descending, each containing:

- `index` — original position in the `documents` list
- `relevance_score` — relevance score (higher = more relevant). Range depends on the model:
    - **v3**: raw logit score, can be negative — compare relative values, not absolute
    - **v2**: normalized probability in `[0, 1]`

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.text_reranker("jinaai/jina-reranker-v3")

    docs = ["Doc A", "Doc B", "Doc C"]
    results = model(query="my query", documents=docs).consume()

    # Pick only the top result
    best = results[0]
    print(docs[best["index"]])      # most relevant document text
    print(best["relevance_score"])  # e.g. 0.45 (highest among the batch)
    ```

## 4. **RAG Integration**

A typical pattern: retrieve candidates with BM25 or vector search, then rerank before passing to the LLM.

???+ example

    ```python
    import msgflux as mf

    reranker = mf.Model.text_reranker("jinaai/jina-reranker-v3")
    chat     = mf.Model.chat_completion("openai/gpt-4.1-mini")

    def rag(query: str, candidate_docs: list[str], top_k: int = 3) -> str:
        # Rerank candidates
        ranked = reranker(query=query, documents=candidate_docs).consume()

        # Take top-k most relevant
        context = "\n\n".join(
            candidate_docs[r["index"]] for r in ranked[:top_k]
        )

        response = chat(messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        }])
        return response.consume()

    answer = rag(
        query="What causes aurora borealis?",
        candidate_docs=[...],  # Retrieved from vector store
    )
    print(answer)
    ```

## 5. **Async Support**

???+ example

    ```python
    import msgflux as mf
    import msgflux.nn.functional as F

    model = mf.Model.text_reranker("jinaai/jina-reranker-v3")

    queries = ["What is AI?", "How does photosynthesis work?"]
    docs    = ["Doc A", "Doc B", "Doc C"]

    results = F.map_gather(
        model,
        kwargs_list=[
            {"query": q, "documents": docs} for q in queries
        ]
    )

    for query, result in zip(queries, results):
        top = result.consume()[0]
        print(f"{query!r} → best doc: {docs[top['index']]}")
    ```

## 6. **Response Caching**

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.text_reranker(
        "jinaai/jina-reranker-v3",
        enable_cache=True,
        cache_size=256
    )

    docs = ["Doc A", "Doc B", "Doc C"]

    # First call — hits API
    results1 = model(query="my query", documents=docs).consume()

    # Second call — served from cache
    results2 = model(query="my query", documents=docs).consume()
    ```

## 7. **Error Handling**

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.text_reranker("jinaai/jina-reranker-v3")

    try:
        results = model(
            query="What is deep learning?",
            documents=["Doc A", "Doc B"]
        ).consume()
    except ImportError:
        print("Provider not installed")
    except Exception as e:
        print(f"Reranking failed: {e}")
    ```
