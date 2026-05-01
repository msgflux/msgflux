import asyncio
import re
from typing import Any, List, Mapping, Optional

try:
    import wikipedia
except ImportError:
    wikipedia = None

import msgflux.nn.functional as F
from msgflux.core.dotdict import dotdict
from msgflux.data.retrievers.base import BaseRetriever, BaseWebSearch
from msgflux.data.retrievers.registry import register_retriever
from msgflux.data.retrievers.types import WebRetriever
from msgflux.logger import logger


@register_retriever
class WikipediaWebRetriever(BaseWebSearch, BaseRetriever, WebRetriever):
    """A customizable Wikipedia client for searching and retrieving Wikipedia content.

    This class provides a flexible interface to search Wikipedia articles, retrieve
    content with optional summaries, and extract images from pages.
    """

    provider = "wikipedia"

    def __init__(
        self,
        *,
        language: Optional[str] = None,
        summary: Optional[int] = None,
        return_images: Optional[bool] = None,
        max_return_images: Optional[int] = None,
    ):
        """Args:
            language:
                The language code for Wikipedia searches. Defaults to "en".
            summary:
                Number of sentences to include in summary.
                Defaults to None (full content).
            return_images:
                Whether to include images in the results. Defaults to False.
            max_return_images:
                Maximum number of images returned. Defaults to 5.

        !!! example

            ```python
            retriever = WikipediaWebRetriever(
                language="pt", return_images=True, summary=3
            )
            results = retriever(
                ["Python programming", "Machine learning"], top_k=2
            )
            print(results)
            ```
        """
        if wikipedia is None:
            raise ImportError(
                "The 'wikipedia' package is not installed. "
                "Please install it via pip: pip install wikipedia"
            )

        # Apply defaults
        if language is None:
            language = "en"
        if return_images is None:
            return_images = False
        if max_return_images is None:
            max_return_images = 5

        self.language = language
        self.max_return_images = max_return_images
        self.return_images = return_images
        self.summary = summary
        self.set_language(self.language)

    def set_language(self, language: str) -> None:
        """Change the Wikipedia language setting."""
        wikipedia.set_lang(language)

    def _process_content(self, content: str, title: str) -> str:
        """Process page content based on summary parameter.

        Args:
            content:
                Raw page content.
            title:
                Page title.

        Returns:
            str: Processed content (title + summary or title + full content).
        """
        if self.summary is not None:
            # Extract specified number of sentences
            sentences = self._extract_sentences(content)
            summary_text = " ".join(sentences[: self.summary])
            return f"{title}\n\n{summary_text}"
        else:
            return f"{title}\n\n{content}"

    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text using regex.

        Args:
            text: Input text.

        Returns:
            List of sentences.
        """
        # Clean text and split into sentences
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Split by sentence endings, considering abbreviations
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

        # Filter out very short sentences and clean them
        clean_sentences = []
        for raw_sentence in sentences:
            sentence = raw_sentence.strip()
            if len(sentence) > 10:  # Filter very short sentences
                clean_sentences.append(sentence)

        return clean_sentences

    def _get_images(self, page) -> List[str]:
        """Extract image URLs from Wikipedia page.

        Args:
            page: Wikipedia page object.

        Returns:
            List[str]: List of image URLs.
        """
        try:
            # Get images from the page
            images = page.images

            # Filter for common image formats and remove SVGs
            valid_images = []
            for img_url in images:
                valid_ext = [".jpg", ".jpeg", ".png", ".gif"]
                if any(ext in img_url.lower() for ext in valid_ext):
                    # Skip small icons and logos by checking if it's likely a image
                    icons = ["commons-logo", "wikimedia", "edit-icon"]
                    if not any(skip in img_url.lower() for skip in icons):
                        valid_images.append(img_url)

            return valid_images[: self.max_return_images]

        except Exception:
            return []

    def _single_search(self, query: str, top_k: int) -> List[Mapping[str, Any]]:
        """Internal method to search Wikipedia for a single query."""
        try:
            # Search for pages
            search_results = wikipedia.search(query, results=top_k)

            results = []
            for title in search_results[:top_k]:
                try:
                    # Get page content
                    page = wikipedia.page(title)

                    # Process content based on summary parameter
                    content = self._process_content(page.content, page.title)

                    result = {"data": {"title": page.title, "content": content}}

                    if self.return_images:  # Add images if requested
                        result["images"] = self._get_images(page)

                    results.append(result)

                except wikipedia.exceptions.DisambiguationError as e:
                    # Handle disambiguation by taking the first option
                    try:
                        page = wikipedia.page(e.options[0])
                        content = self._process_content(page.content, page.title)

                        result = {"data": {"title": page.title, "content": content}}

                        if self.return_images:
                            result["images"] = self._get_images(page)

                        results.append(result)
                    except Exception as e:
                        logger.debug(str(e))
                        continue

                except Exception as e:
                    logger.debug(str(e))
                    continue

            return results

        except Exception:
            return []

    def _search(self, queries: List[str], top_k: int) -> List[dotdict]:
        args_list = [(query,) for query in queries]
        kwargs_list = [{"top_k": top_k} for _ in queries]
        query_results = F.map_gather(
            self._single_search, args_list=args_list, kwargs_list=kwargs_list
        )
        results = []
        for result in query_results:
            results.append(dotdict({"results": result}))
        return results

    async def _asingle_search(self, query: str, top_k: int) -> List[Mapping[str, Any]]:
        """Async internal method to search Wikipedia for a single query.

        Uses asyncio.run_in_executor to run synchronous wikipedia calls
        in a thread pool.
        """
        loop = asyncio.get_event_loop()

        try:
            # Search for pages using executor
            search_results = await loop.run_in_executor(
                None, lambda: wikipedia.search(query, results=top_k)
            )

            results = []
            for title in search_results[:top_k]:
                try:
                    # Get page content using executor
                    page = await loop.run_in_executor(
                        None, lambda t=title: wikipedia.page(t)
                    )

                    # Process content based on summary parameter
                    content = self._process_content(page.content, page.title)

                    result = {"data": {"title": page.title, "content": content}}

                    if self.return_images:  # Add images if requested
                        result["images"] = self._get_images(page)

                    results.append(result)

                except wikipedia.exceptions.DisambiguationError as disambiguation_error:
                    # Handle disambiguation by taking the first option
                    try:
                        first_option = disambiguation_error.options[0]
                        page = await loop.run_in_executor(
                            None, lambda opt=first_option: wikipedia.page(opt)
                        )
                        content = self._process_content(page.content, page.title)

                        result = {"data": {"title": page.title, "content": content}}

                        if self.return_images:
                            result["images"] = self._get_images(page)

                        results.append(result)
                    except Exception as e:
                        logger.debug(str(e))
                        continue

                except Exception as e:
                    logger.debug(str(e))
                    continue

            return results

        except Exception:
            return []

    async def _asearch(self, queries: List[str], top_k: int):
        """Async search that runs multiple queries in parallel."""
        tasks = [self._asingle_search(query, top_k) for query in queries]
        query_results = await asyncio.gather(*tasks)
        results = []
        for result in query_results:
            results.append(dotdict({"results": result}))
        return results
