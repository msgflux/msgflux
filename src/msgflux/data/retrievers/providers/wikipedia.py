import re
from typing import List, Optional, Union

import wikipedia

from msgflux.data.retrievers.base import BaseRetriever
from msgflux.data.retrievers.response import RetrieverResults
from msgflux.data.retrievers.types import WebRetriever 
from msgflux.dotdict import dotdict
from msgflux.nn import functional as F


class WikipediaWebRetriever(BaseRetriever, WebRetriever):
    """
    A customizable Wikipedia client for searching and retrieving Wikipedia content.
    
    This class provides a flexible interface to search Wikipedia articles, retrieve
    content with optional summaries, and extract images from pages.
    """

    provider = "wikipedia"
    
    def __init__(
        self,
        language: Optional[str] = "en",
        summary: Optional[int] = None,        
        return_images: Optional[bool] = False,
        max_return_images: Optional[int] = 5
    ):
        """
        Args:
            language:
                The language code for Wikipedia searches.
            summary:
                Number of sentences to include in summary.
            return_images:
                Whether to include images in the results.
            max_return_images:
                Maximum number of images returned.
        
        !!! example

            ```python
            client = WikipediaClient(language="pt", return_images=True, summary=3)
            results = client(["Python programming", "Machine learning"], top_k=2)
            print(results)
            ```
        """
        self.language = language
        self.max_return_images = max_return_images
        self.return_images = return_images
        self.summary = summary
        self.set_language(self.language)

    def set_language(self, language: str) -> None:
        """Change the Wikipedia language setting."""
        wikipedia.set_lang(language)
    
    def _process_content(self, content: str, title: str) -> str:
        """
        Process page content based on summary parameter.
        
        Args:
            content (str): Raw page content.
            title (str): Page title.
        
        Returns:
            str: Processed content (title + summary or title + full content).
        """
        if self.summary is not None:
            # Extract specified number of sentences
            sentences = self._extract_sentences(content)
            summary_text = " ".join(sentences[:self.summary])
            return f"{title}\n\n{summary_text}"
        else:
            return f"{title}\n\n{content}"
    
    def _extract_sentences(self, text: str) -> List[str]:
        """
        Extract sentences from text using regex.
        
        Args:
            text (str): Input text.
        
        Returns:
            List[str]: List of sentences.
        """
        # Clean text and split into sentences
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        
        # Split by sentence endings, considering abbreviations
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        
        # Filter out very short sentences and clean them
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter very short sentences
                clean_sentences.append(sentence)
        
        return clean_sentences
    
    def _get_images(self, page) -> List[str]:
        """
        Extract image URLs from Wikipedia page.
        
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
                if any(ext in img_url.lower() for ext in [".jpg", ".jpeg", ".png", ".gif"]):
                    # Skip small icons and logos by checking if it"s likely a content image
                    if not any(skip in img_url.lower() for skip in ["commons-logo", "wikimedia", "edit-icon"]):
                        valid_images.append(img_url)
            
            return valid_images[:self.max_return_images]
            
        except Exception:
            return []

    def _search(self, query: str, top_k: int) -> List[RetrieverResults]:
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
                    
                    result = {
                        "data": {
                            "title": page.title,
                            "content": content
                        }
                    }
                    
                    # Add images if requested
                    if self.return_images:
                        result["images"] = self._get_images(page)
                    
                    results.append(result)
                    
                except wikipedia.exceptions.DisambiguationError as e:
                    # Handle disambiguation by taking the first option
                    try:
                        page = wikipedia.page(e.options[0])
                        content = self._process_content(page.content, page.title)
                        
                        result = {
                            "data": {
                                "title": page.title,
                                "content": content
                            }
                        }
                        
                        if self.return_images:
                            result["images"] = self._get_images(page)
                        
                        results.append(result)
                    except Exception:
                        continue
                        
                except wikipedia.exceptions.PageError:
                    # Skip pages that don"t exist
                    continue
                except Exception:
                    # Skip any other errors
                    continue
            
            return results
            
        except Exception as e:
            print(f"Error searching for `{query}`: {str(e)}")
            return []

    def __call__(
        self,
        queries: Union[str, List[str]],
        top_k: Optional[int] = 1
    ) -> List[RetrieverResults]:
        """
        Search Wikipedia and retrieve results for given queries.
        
        Args:
            queries:
                Single query string or list of queries.
            top_k:
                Number of results to return per query.

        Returns:
            List of retriever results containing `data` and `images` (if return_images=True).
        """
        if isinstance(queries, str):
            queries = [queries]
        
        response = []

        args_list = [(query,) for query in queries]
        kwargs_list = [{"top_k": top_k} for _ in queries]

        results = F.map_gather(self._search, args_list=args_list, kwargs_list=kwargs_list)

        response = []
        for result in results:
            response.append(dotdict({"results": result}))

        return response