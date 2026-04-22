# Web Retrievers

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

---

## 9. **SerpApi Search**

The `serpapi` retriever queries SerpApi and returns structured search results from engines such as Google. Use it when you need general web, news, image, shopping, or localized search through SerpApi.

!!! info "Dependencies"
    Requires `google-search-results` and the `SERPAPI_API_KEY` env variable:
    `pip install google-search-results`

### Quick Start

???+ example

    ```python
    import msgflux as mf

    mf.set_envs(SERPAPI_API_KEY="...")

    retriever = mf.Retriever.web("serpapi")
    response = retriever("latest Python release", top_k=3)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.data.url)
        print(result.data.content)
    ```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `engine` | `"google"` | Search engine to use, such as `"google"`, `"bing"`, or `"yahoo"` |
| `location` | `None` | Location for localized results, such as `"Austin,Texas"` |
| `gl` | `None` | Google country code, such as `"us"` or `"br"` |
| `hl` | `None` | Google UI language, such as `"en"` or `"pt"` |
| `safe` | `None` | Safe search mode, such as `"active"` or `"off"` |
| `tbm` | `None` | Search type, such as `"nws"` for news or `"isch"` for images |

```python
import msgflux as mf

retriever = mf.Retriever.web(
    "serpapi",
    location="Sao Paulo, Brazil",
    gl="br",
    hl="pt",
)
```

### News Search

Set `tbm="nws"` to retrieve news results:

???+ example

    ```python
    import msgflux as mf

    retriever = mf.Retriever.web("serpapi", tbm="nws", gl="us", hl="en")
    response = retriever("AI regulation", top_k=5)

    for result in response.data[0].results:
        print(result.data.title)
        print(result.data.date)
        print(result.data.url)
    ```

### Batch Queries

```python
import msgflux as mf

retriever = mf.Retriever.web("serpapi", engine="google")

queries = ["Python packaging", "Rust async runtime"]
response = retriever(queries, top_k=2)

for i, query in enumerate(queries):
    print(f"\n{query}")
    for result in response.data[i].results:
        print(result.data.title)
```

### Async Search

```python
import msgflux as mf

retriever = mf.Retriever.web("serpapi", gl="us", hl="en")

response = await retriever.acall(["Python 3.14", "Django release"], top_k=2)

for item in response.data:
    print(item.results[0].data.title)
```
