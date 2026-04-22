# Web Retrievers

## ✦₊⁺ Overview

Web retrievers query online sources at request time and return structured results through `mf.Retriever.web(...)`. The built-in `wikipedia` provider fetches article content with optional summaries and images.

---

## 1. **Wikipedia Search**

The `wikipedia` retriever fetches and returns Wikipedia article content at query time. Unlike lexical retrievers, it requires no pre-indexed corpus — it queries the Wikipedia API directly and returns structured results with title, content, and optionally images.

!!! info "Dependencies"
    Requires the `wikipedia` package: `pip install wikipedia`

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `language` | `"en"` | Wikipedia language code (`"pt"`, `"es"`, `"fr"`, …) |
| `summary` | `None` | Number of sentences to return — `None` returns the full article |
| `return_images` | `False` | Whether to include image URLs in results |
| `max_return_images` | `5` | Maximum number of image URLs per result |

### Examples

=== "Search"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web("wikipedia")
    response = retriever("machine learning", top_k=2)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.data.content[:200])
    ```

=== "Summary"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web("wikipedia", summary=2)
    response = retriever("Eiffel Tower")

    print(response.data[0].results[0].data.content)
    # Eiffel Tower
    #
    # The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.
    # It is named after the engineer Gustave Eiffel, whose company designed and built it.
    ```

=== "Images"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web(
        "wikipedia",
        return_images=True,
        max_return_images=3,
    )

    response = retriever("Colosseum")

    result = response.data[0].results[0]
    print(result.data.title)
    print(result.images)
    ```

=== "Languages"

    ```python
    import msgflux as mf

    queries = [
        ("pt", "inteligência artificial"),
        ("es", "aprendizaje automático"),
        ("fr", "réseau de neurones"),
    ]

    for language, query in queries:
        retriever = mf.Retriever.web("wikipedia", language=language, summary=3)
        response = retriever(query)
        print(response.data[0].results[0].data.content)
    ```

=== "Batch"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web("wikipedia", summary=2)

    queries = ["Python programming", "Rust programming language", "Go programming"]
    response = retriever(queries, top_k=1)

    for i, query in enumerate(queries):
        result = response.data[i].results[0]
        print(f"\n{query}: {result.data.title}")
        print(result.data.content)
    ```

=== "RAG"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web("wikipedia", summary=5)
    chat = mf.Model.chat_completion("openai/gpt-4.1-mini")

    def answer_with_wikipedia(question: str) -> str:
        response = retriever(question, top_k=2)

        context = "\n\n".join(
            result.data.content
            for result in response.data[0].results
        )

        return chat(messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        }]).consume()

    print(answer_with_wikipedia("How does the James Webb Space Telescope work?"))
    ```

=== "Async"

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web("wikipedia", summary=3)

    queries = ["quantum computing", "photosynthesis", "black holes"]
    response = await retriever.acall(queries, top_k=1)

    for i, query in enumerate(queries):
        result = response.data[i].results[0]
        print(f"\n{query}: {result.data.title}")
    ```
