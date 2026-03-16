# Image Embedder

The `image_embedder` model converts images into dense vector representations that capture their visual and semantic meaning. When paired with a text embedder from the same model family, it enables cross-modal search.

!!! info "Dependencies"
    See [Dependency Management](../../dependency-management.md) for the complete provider matrix.

## ✦₊⁺ Overview

Image embeddings encode visual content into fixed-size vectors that can be compared, stored, and searched. They enable:

- **Visual Search**: Find images similar to a query image
- **Cross-Modal Search**: Search images with text queries (and vice versa)
- **Clustering**: Group visually or semantically similar images
- **Recommendation**: Suggest related visual content

## 1. **Quick Start**

???+ example

    ```python
    import msgflux as mf

    embedder = mf.Model.image_embedder("jinaai/jina-clip-v2")

    response = embedder("path/to/image.jpg")
    embedding = response.consume()

    print(len(embedding))   # 1024
    print(embedding[:5])    # [0.021, -0.134, 0.087, ...]
    ```

## 2. **Supported Providers**

### JinaAI

???+ example

    ```python
    import msgflux as mf

    embedder = mf.Model.image_embedder("jinaai/jina-clip-v2")
    ```

!!! note "Cross-modal model"
    `jina-clip-v2` is trained jointly on text and images in the same embedding space. This means embeddings from `image_embedder` and `text_embedder` (using the same model) are directly comparable — enabling cross-modal search without any adaptation layer.

## 3. **Custom Dimensions**

???+ example

    ```python
    import msgflux as mf

    # Reduce dimensions via Matryoshka truncation
    embedder = mf.Model.image_embedder(
        "jinaai/jina-clip-v2",
        dimensions=512  # Reduced from 1024
    )

    embedding = embedder("photo.jpg").consume()
    print(len(embedding))  # 512
    ```

## 4. **Batch Processing**

Pass a list of image paths to embed multiple images in a single call:

???+ example

    ```python
    import msgflux as mf

    embedder = mf.Model.image_embedder("jinaai/jina-clip-v2")

    images = [
        "products/shirt_blue.jpg",
        "products/shirt_red.jpg",
        "products/pants_black.jpg"
    ]

    embeddings = embedder(images).consume()  # List[List[float]]
    print(f"Embedded {len(embeddings)} images, {len(embeddings[0])} dims each")
    ```

## 5. **Cross-Modal Search**

Search images using text queries by embedding both in the same space:

???+ example

    ```python
    import msgflux as mf
    import numpy as np

    img_embedder  = mf.Model.image_embedder("jinaai/jina-clip-v2")
    text_embedder = mf.Model.text_embedder("jinaai/jina-clip-v2")

    # Build image index
    image_paths = ["cat.jpg", "dog.jpg", "car.jpg", "tree.jpg"]
    image_embeddings = np.array(img_embedder(image_paths).consume())

    def search(query: str, top_k: int = 2):
        query_emb = np.array(text_embedder(query).consume())
        scores = image_embeddings @ query_emb / (
            np.linalg.norm(image_embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        top = np.argsort(scores)[-top_k:][::-1]
        return [(image_paths[i], float(scores[i])) for i in top]

    results = search("a furry animal")
    for path, score in results:
        print(f"{score:.3f}  {path}")
    # 0.842  cat.jpg
    # 0.791  dog.jpg
    ```

## 6. **Response Caching**

???+ example

    ```python
    import msgflux as mf

    embedder = mf.Model.image_embedder(
        "jinaai/jina-clip-v2",
        enable_cache=True,
        cache_size=256
    )

    # First call — hits API
    emb1 = embedder("photo.jpg").consume()

    # Second call — served from cache
    emb2 = embedder("photo.jpg").consume()
    ```

## 7. **Error Handling**

???+ example

    ```python
    import msgflux as mf

    embedder = mf.Model.image_embedder("jinaai/jina-clip-v2")

    try:
        embedding = embedder("photo.jpg").consume()
    except ImportError:
        print("Provider not installed")
    except FileNotFoundError:
        print("Image file not found")
    except Exception as e:
        print(f"Embedding failed: {e}")
    ```
