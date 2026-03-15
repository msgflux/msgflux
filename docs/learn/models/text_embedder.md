# Text Embedder

The `text_embedder` model transforms text into dense vector representations (embeddings) that capture semantic meaning. These vectors enable similarity search, semantic retrieval, clustering, and classification tasks.

!!! info "Dependencies"
    See [Dependency Management](../../dependency-management.md) for the complete provider matrix.

## ✦₊⁺ Overview

Text embeddings convert sentences, paragraphs, or documents into numerical vectors that encode their semantic meaning. Unlike simple word counts or TF-IDF, embeddings capture:

- **Semantic similarity**: Similar meanings have similar vectors
- **Contextual understanding**: Same words in different contexts get different embeddings
- **Dimensionality reduction**: High-dimensional text → fixed-size vectors

### Common Use Cases

- **Semantic Search**: Find documents similar to a query
- **RAG (Retrieval-Augmented Generation)**: Retrieve relevant context for LLMs
- **Clustering**: Group similar documents
- **Classification**: Train classifiers on embeddings
- **Recommendation**: Find similar items

## 1. **Quick Start**

### 1.1 **Basic Usage**

???+ example

    ```python
    import msgflux as mf

    # Create embedder
    embedder = mf.Model.text_embedder("openai/text-embedding-3-small")

    # Generate embedding
    response = embedder("Hello, world!")

    # Get vector
    embedding = response.consume()
    print(len(embedding))  # 1536
    print(embedding[:5])   # [0.123, -0.456, 0.789, -0.234, 0.567]
    ```

### 1.2 **With Custom Dimensions**

???+ example

    ```python
    import msgflux as mf

    # OpenAI models support custom dimensions
    embedder = mf.Model.text_embedder(
        "openai/text-embedding-3-small",
        dimensions=256  # Matryoshka, Reduce from 1536 to 256
    )

    response = embedder("Compact embedding")
    embedding = response.consume()
    print(len(embedding))  # 256
    ```

## 2. **Batch Processing**

Most providers support native batch processing by accepting a `List[str]` in a single API call. This is more efficient than multiple individual requests because it reduces round-trips and allows the provider to optimize internally.

Providers with native batch support (OpenAI, JinaAI, Together AI, vLLM, Ollama) set `batch_support = True` internally. When you pass a list through the `Embedder` module, it automatically uses native batch mode for these providers.

???+ example

    === "Native Batch (Recommended)"

        ```python
        import msgflux as mf

        embedder = mf.Model.text_embedder("openai/text-embedding-3-small")

        product_descriptions = [
            "Wireless noise-cancelling headphones with 30-hour battery life",
            "Ergonomic mechanical keyboard with RGB backlighting",
            "Ultra-wide 34-inch curved monitor for productivity",
            "Portable SSD with 2TB storage and USB-C connection",
        ]

        # Single API call — provider embeds all texts at once
        response = embedder(product_descriptions)
        embeddings = response.consume()  # List[List[float]]

        print(f"Generated {len(embeddings)} embeddings")
        print(f"Embedding dims: {len(embeddings[0])}")  # 1536
        ```

    === "Async Batch"

        ```python
        import msgflux as mf
        import asyncio

        embedder = mf.Model.text_embedder("openai/text-embedding-3-small")

        async def embed_batch(texts):
            response = await embedder.acall(texts)
            return response.consume()  # List[List[float]]

        support_tickets = [
            "My order hasn't arrived after 10 days, please help",
            "I was charged twice for the same purchase",
            "How do I return a damaged item?",
        ]
        embeddings = asyncio.run(embed_batch(support_tickets))
        ```

    === "Concurrent (Fallback)"

        ```python
        import msgflux as mf
        import msgflux.nn.functional as F

        # For providers without native batch support (batch_support=False)
        embedder = mf.Model.text_embedder("some-provider/model")

        faq_questions = [
            "What is your refund policy?",
            "How long does shipping take?",
            "Do you ship internationally?",
            "Can I change my order after placing it?",
        ]

        # Issues one API call per text, executed concurrently
        results = F.map_gather(
            embedder,
            args_list=[(q,) for q in faq_questions]
        )

        embeddings = [r.consume() for r in results]
        print(f"Generated {len(embeddings)} embeddings concurrently")
        ```

!!! note
    The `Embedder` nn module (from `msgflux.nn`) handles this automatically — it uses native batch when `batch_support=True` and falls back to `F.map_gather` otherwise. When using `mf.Model.text_embedder()` directly, you control the strategy yourself.

## 3. **Response Caching**

Cache embeddings to avoid redundant API calls:

### 3.1 **Enabling Cache**

???+ example

    ```python
    import msgflux as mf

    # Enable cache (highly recommended for embeddings)
    embedder = mf.Model.text_embedder(
        "openai/text-embedding-3-small",
        enable_cache=True,
        cache_size=1000  # Cache up to 1000 embeddings
    )

    # First call - hits API
    response1 = embedder("machine learning")
    print(response1.consume()[:5])

    # Second call - returns cached result
    response2 = embedder("machine learning")
    print(response2.consume()[:5])  # Same result, no API call

    # Check cache stats
    if embedder._response_cache:
        stats = embedder._response_cache.cache_info()
        print(f"Cache hits: {stats['hits']}")
        print(f"Cache misses: {stats['misses']}")
    ```

## 4. **Working with Embeddings**

### 4.1 **Cosine Similarity**

???+ example

    ```python
    import msgflux as mf
    import numpy as np

    def cosine_similarity(a, b):
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    embedder = mf.Model.text_embedder("openai/text-embedding-3-small")

    # Generate embeddings
    text1 = "I love machine learning"
    text2 = "Machine learning is great"
    text3 = "The weather is nice today"

    emb1 = embedder(text1).consume()
    emb2 = embedder(text2).consume()
    emb3 = embedder(text3).consume()

    # Calculate similarities
    sim_1_2 = cosine_similarity(emb1, emb2)
    sim_1_3 = cosine_similarity(emb1, emb3)

    print(f"Similarity (text1, text2): {sim_1_2:.4f}")  # ~0.85
    print(f"Similarity (text1, text3): {sim_1_3:.4f}")  # ~0.30
    ```

### 4.2 **Semantic Search**

???+ example

    ```python
    import msgflux as mf
    import numpy as np

    def semantic_search(query, documents, embedder, top_k=3):
        """Find most similar documents to query."""

        # Embed query
        query_emb = np.array(embedder(query).consume())

        # Embed all documents
        doc_embs = [np.array(embedder(doc).consume()) for doc in documents]

        # Calculate similarities
        similarities = [
            np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
            for doc_emb in doc_embs
        ]

        # Get top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [(documents[i], similarities[i]) for i in top_indices]

    # Example usage
    embedder = mf.Model.text_embedder("openai/text-embedding-3-small", enable_cache=True)

    documents = [
        "Python is a programming language",
        "Machine learning uses algorithms",
        "The weather is sunny today",
        "Neural networks are powerful",
        "I like to eat pizza"
    ]

    query = "Tell me about AI and ML"
    results = semantic_search(query, documents, embedder)

    for doc, score in results:
        print(f"{score:.4f}: {doc}")
    # 0.7234: Machine learning uses algorithms
    # 0.6891: Neural networks are powerful
    # 0.4123: Python is a programming language
    ```

## 5. **RAG Integration**

Embeddings are essential for Retrieval-Augmented Generation:

### 5.1 **Building a Simple RAG System**

???+ example

    ```python
    import msgflux as mf
    import numpy as np

    class SimpleRAG:
        def __init__(self, embedder_model, chat_model):
            self.embedder = embedder_model
            self.chat = chat_model
            self.documents = []
            self.embeddings = []

        def add_documents(self, docs):
            """Add documents to knowledge base."""
            self.documents.extend(docs)

            # Generate embeddings
            for doc in docs:
                emb = self.embedder(doc).consume()
                self.embeddings.append(np.array(emb))

        def retrieve(self, query, top_k=3):
            """Retrieve most relevant documents."""
            query_emb = np.array(self.embedder(query).consume())

            similarities = [
                np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
                for doc_emb in self.embeddings
            ]

            top_indices = np.argsort(similarities)[-top_k:][::-1]
            return [self.documents[i] for i in top_indices]

        def query(self, question):
            """Ask question with RAG."""
            # Retrieve relevant docs
            context_docs = self.retrieve(question)
            context = "\n\n".join(context_docs)

            # Generate answer with context
            prompt = f"""Answer the question based on the context below.

    Context:
    {context}

    Question: {question}

    Answer:"""

            response = self.chat(messages=[{"role": "user", "content": prompt}])
            return response.consume()

    # Usage
    embedder = mf.Model.text_embedder("openai/text-embedding-3-small", enable_cache=True)
    chat = mf.Model.chat_completion("openai/gpt-4.1-mini")

    rag = SimpleRAG(embedder, chat)

    # Add knowledge
    rag.add_documents([
        "msgflux is a Python library for building AI systems.",
        "The Model class provides unified access to different AI providers.",
        "AutoParams allows dataclass-style module definitions.",
        "msgflux supports multiple OpenAI-compatible providers."
    ])

    # Ask questions
    answer = rag.query("What is msgflux?")
    print(answer)
    ```

## 6. **Dimensions and Performance**

### 6.1 **Choosing Dimensions**

???+ example

    ```python
    import msgflux as mf

    # Higher dimensions = better accuracy, more storage/compute
    embedder_large = mf.Model.text_embedder(
        "openai/text-embedding-3-large",
        dimensions=3072  # Full size
    )

    # Lower dimensions = faster, less storage, slightly lower accuracy
    embedder_small = mf.Model.text_embedder(
        "openai/text-embedding-3-small",
        dimensions=256   # Reduced from 1536
    )

    # Trade-off example:
    # - 3072 dims: 99% accuracy, 12x storage
    # - 1536 dims: 98% accuracy, 6x storage
    # - 512 dims:  95% accuracy, 2x storage
    # - 256 dims:  92% accuracy, 1x storage
    ```

## 7. **Response Metadata**

Access usage and cost information:

???+ example

    ```python
    import msgflux as mf

    embedder = mf.Model.text_embedder("openai/text-embedding-3-small")

    response = embedder("This is a test sentence")

    # Check token usage
    print(response.metadata)
    # {'usage': {'prompt_tokens': 5, 'total_tokens': 5}}

    # Access the model profile directly from the embedder
    print(embedder.profile)
    # ModelProfile(id='text-embedding-3-small', name='text-embedding-3-small',
    #   provider_id='openai',
    #   capabilities=ModelCapabilities(tool_call=False, structured_output=False,
    #     reasoning=False, attachment=False, temperature=False),
    #   modalities=ModelModalities(input=['text'], output=['text']),
    #   cost=ModelCost(input_per_million=0.02, output_per_million=0,
    #     cache_read_per_million=None),
    #   limits=ModelLimits(context=8191, output=1536),
    #   knowledge='2024-01', release_date='2024-01-25',
    #   last_updated='2024-01-25', open_weights=False)

    # Calculate cost
    tokens = response.metadata.usage.total_tokens
    cost = tokens * (embedder.profile.cost.input_per_million / 1_000_000)
    print(f"Cost: ${cost:.6f}")
    ```

## 8. **Error Handling**

???+ example

    ```python
    import msgflux as mf

    embedder = mf.Model.text_embedder("openai/text-embedding-3-small")

    try:
        response = embedder("Some text")
        embedding = response.consume()
    except ImportError:
        print("Provider not installed")
    except ValueError as e:
        print(f"Invalid input: {e}")
    except Exception as e:
        print(f"API error: {e}")
    ```

## See Also

- [Model](model.md) - Model factory and registry
- [Chat Completion](chat_completion.md) - Chat models
- [Data Retrievers](../data/retrievers.md) - Vector databases and retrievers
- [RAG Patterns](../patterns/rag.md) - RAG implementation patterns
