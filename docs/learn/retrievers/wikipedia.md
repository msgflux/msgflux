# Wikipedia Retriever

## ✦₊⁺ Overview

The `wikipedia` retriever fetches and returns Wikipedia article content at query time. Unlike lexical retrievers, it requires no pre-indexed corpus — it queries the Wikipedia API directly and returns structured results with title, content, and optionally images.

!!! info "Dependencies"
    Requires the `wikipedia` package: `pip install wikipedia`

---

## 1. **Quick Start**

???+ example

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web("wikipedia")

    response = retriever("machine learning", top_k=2)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.data.content[:200])
    ```

---

## 2. **Parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `language` | `"en"` | Wikipedia language code (`"pt"`, `"es"`, `"fr"`, …) |
| `summary` | `None` | Number of sentences to return — `None` returns the full article |
| `return_images` | `False` | Whether to include image URLs in results |
| `max_return_images` | `5` | Maximum number of image URLs per result |

```python
import msgflux as mf

retriever = mf.Retriever.web("wikipedia",
    language="en",
    summary=3,           # Return only the first 3 sentences
    return_images=True,
    max_return_images=3,
)
```

---

## 3. **Summary Mode**

By default, the full article content is returned. Set `summary` to an integer to limit the response to the first N sentences — useful when feeding context to an LLM:

???+ example

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

---

## 4. **Images**

Enable `return_images=True` to get a list of image URLs from each article. Icons, logos, and SVGs are filtered automatically:

???+ example

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web("wikipedia",
        return_images=True,
        max_return_images=3
    )

    response = retriever("Colosseum")

    result = response.data[0].results[0]
    print(result.data.title)    # "Colosseum"
    print(result.images)        # ["https://upload.wikimedia.org/...jpg", ...]
    ```

---

## 5. **Multilingual**

Set `language` to any Wikipedia language code:

???+ example

    === "Portuguese"

        ```python
        import msgflux as mf

        retriever = mf.Retriever.web("wikipedia", language="pt", summary=3)
        response = retriever("inteligência artificial")
        print(response.data[0].results[0].data.content)
        ```

    === "Spanish"

        ```python
        import msgflux as mf

        retriever = mf.Retriever.web("wikipedia", language="es", summary=3)
        response = retriever("aprendizaje automático")
        print(response.data[0].results[0].data.content)
        ```

    === "French"

        ```python
        import msgflux as mf

        retriever = mf.Retriever.web("wikipedia", language="fr", summary=3)
        response = retriever("réseau de neurones")
        print(response.data[0].results[0].data.content)
        ```

---

## 6. **Batch Queries**

```python
import msgflux as mf

retriever = mf.Retriever.web("wikipedia", summary=2)

queries = ["Python programming", "Rust programming language", "Go programming"]
response = retriever(queries, top_k=1)

for i, query in enumerate(queries):
    result = response.data[i].results[0]
    print(f"\n{result.data.title}")
    print(result.data.content)
```

---

## 7. **RAG Integration**

A typical pattern: retrieve Wikipedia context, then pass it to an LLM:

???+ example

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
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }]).consume()

    print(answer_with_wikipedia("How does the James Webb Space Telescope work?"))
    ```

---

## 8. **Async Support**

```python
import msgflux as mf

retriever = mf.Retriever.web("wikipedia", summary=3)

queries = ["quantum computing", "photosynthesis", "black holes"]
response = await retriever.acall(queries, top_k=1)

for i, query in enumerate(queries):
    result = response.data[i].results[0]
    print(f"\n{query} → {result.data.title}")
```
