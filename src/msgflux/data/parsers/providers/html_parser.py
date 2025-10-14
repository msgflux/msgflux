from typing import Dict, List, Optional, Union

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

from msgflux.data.parsers.base import BaseParser
from msgflux.data.parsers.registry import register_parser
from msgflux.data.parsers.response import ParserResponse
from msgflux.data.parsers.types import HtmlParser
from msgflux.dotdict import dotdict


@register_parser
class BeautifulSoupHtmlParser(BaseParser, HtmlParser):
    """BeautifulSoup-based HTML Parser.

    Converts HTML documents to clean Markdown format.
    Uses BeautifulSoup4 for HTML parsing.

    Features:
    - Clean text extraction (removes tags)
    - Structure preservation (headings, lists, etc.)
    - Link extraction
    - Image detection
    - Conversion to Markdown

    Example:
        >>> parser = Parser.html("beautifulsoup", extract_links=True)
        >>> response = parser("page.html")
        >>> print(response.data["text"])
        >>> print(response.data["links"])
    """

    provider = "beautifulsoup"

    def __init__(
        self,
        *,
        extract_links: Optional[bool] = True,
        extract_images: Optional[bool] = True,
        remove_scripts: Optional[bool] = True,
        remove_styles: Optional[bool] = True,
    ):
        """Initialize HTML parser.

        Args:
            extract_links:
                If True, extract all links from the document.
            extract_images:
                If True, extract image URLs.
            remove_scripts:
                If True, remove <script> tags and content.
            remove_styles:
                If True, remove <style> tags and content.
        """
        if BeautifulSoup is None:
            raise ImportError(
                "`beautifulsoup4` is not available. "
                "Install with `pip install beautifulsoup4`"
            )

        self.extract_links = extract_links
        self.extract_images = extract_images
        self.remove_scripts = remove_scripts
        self.remove_styles = remove_styles
        self._initialize()

    def _initialize(self):
        """Initialize parser state."""
        pass

    def __call__(
        self, data: Union[str, bytes], **kwargs
    ) -> ParserResponse:
        """Parse an HTML document.

        Args:
            data:
                HTML file path, URL, or bytes/string content.
            **kwargs:
                Additional parsing options (currently unused).

        Returns:
            ParserResponse containing:
            - text: Markdown-formatted content
            - links: List of extracted links (if extract_links=True)
            - images: List of image URLs (if extract_images=True)
            - metadata: Document metadata

        Raises:
            FileNotFoundError:
                If file path doesn't exist.
            ValueError:
                If data type is not supported.
        """
        # Validate file type if it's a path
        if isinstance(data, str) and not data.startswith(("http://", "https://")):
            if "." in data:  # Has extension
                self._validate_file_type(data, [".html", ".htm"])

        # Parse the document
        result = self._parse(data)

        # Create response
        response = ParserResponse()
        response.set_response_type("html_parse")
        response.add(result)

        return response

    def _parse(self, data: Union[str, bytes]) -> Dict[str, any]:
        """Parse HTML and extract content.

        Args:
            data:
                HTML file path, bytes, or string content.

        Returns:
            Dictionary with:
            - text: Markdown content
            - links: List of links
            - images: List of image URLs
            - metadata: Document metadata
        """
        # Load content
        if isinstance(data, bytes):
            content = data.decode("utf-8")
        elif isinstance(data, str):
            if data.startswith(("http://", "https://")):
                # Load from URL
                file_bytes = self._load_file(data)
                content = file_bytes.decode("utf-8")
            elif data.startswith(("<!DOCTYPE", "<html", "<?xml")):
                # Already HTML content
                content = data
            else:
                # Load from file path
                with open(data, "r", encoding="utf-8") as f:
                    content = f.read()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Parse HTML
        soup = BeautifulSoup(content, "html.parser")

        # Remove unwanted elements
        if self.remove_scripts:
            for script in soup.find_all("script"):
                script.decompose()

        if self.remove_styles:
            for style in soup.find_all("style"):
                style.decompose()

        # Extract links
        links = []
        if self.extract_links:
            for link in soup.find_all("a", href=True):
                links.append({
                    "text": link.get_text(strip=True),
                    "url": link["href"]
                })

        # Extract images
        images = []
        if self.extract_images:
            for img in soup.find_all("img", src=True):
                images.append({
                    "alt": img.get("alt", ""),
                    "url": img["src"]
                })

        # Convert to Markdown
        markdown = self._html_to_markdown(soup)

        # Get title
        title = soup.find("title")
        title_text = title.get_text(strip=True) if title else None

        # Prepare metadata
        metadata = dotdict({
            "title": title_text,
            "num_links": len(links),
            "num_images": len(images),
            "extract_links": self.extract_links,
            "extract_images": self.extract_images,
        })

        return {
            "text": markdown.strip(),
            "links": links if self.extract_links else None,
            "images": images if self.extract_images else None,
            "metadata": metadata,
        }

    def _html_to_markdown(self, soup: "BeautifulSoup") -> str:
        """Convert BeautifulSoup object to Markdown.

        Args:
            soup: BeautifulSoup parsed HTML.

        Returns:
            Markdown formatted string.
        """
        markdown = []

        # Process body (or entire soup if no body)
        body = soup.find("body") or soup

        for element in body.descendants:
            if element.name is None:
                # Text node
                text = str(element).strip()
                if text:
                    markdown.append(text)

            elif element.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                level = int(element.name[1])
                text = element.get_text(strip=True)
                if text:
                    markdown.append(f"\n\n{'#' * level} {text}\n")

            elif element.name == "p":
                text = element.get_text(strip=True)
                if text:
                    markdown.append(f"\n\n{text}\n")

            elif element.name == "br":
                markdown.append("\n")

            elif element.name in ["ul", "ol"]:
                # Lists are handled by their li children
                markdown.append("\n")

            elif element.name == "li":
                text = element.get_text(strip=True)
                if text and element.parent:
                    if element.parent.name == "ol":
                        markdown.append(f"\n1. {text}")
                    else:
                        markdown.append(f"\n- {text}")

            elif element.name == "a":
                text = element.get_text(strip=True)
                href = element.get("href", "")
                if text and href:
                    markdown.append(f"[{text}]({href})")

            elif element.name == "code":
                text = element.get_text()
                markdown.append(f"`{text}`")

            elif element.name == "pre":
                text = element.get_text()
                markdown.append(f"\n```\n{text}\n```\n")

            elif element.name == "blockquote":
                text = element.get_text(strip=True)
                if text:
                    quoted = "\n".join(f"> {line}" for line in text.split("\n"))
                    markdown.append(f"\n{quoted}\n")

        # Clean up the output
        result = "".join(markdown)
        # Remove excessive newlines
        while "\n\n\n" in result:
            result = result.replace("\n\n\n", "\n\n")

        return result

    async def acall(
        self, data: Union[str, bytes], **kwargs
    ) -> ParserResponse:
        """Async version of __call__. Parse an HTML document asynchronously.

        Args:
            data:
                HTML file path, URL, or bytes/string content.
            **kwargs:
                Additional parsing options (currently unused).

        Returns:
            ParserResponse containing parsed data.

        Raises:
            FileNotFoundError:
                If file path doesn't exist.
            ValueError:
                If data type is not supported.
        """
        # Validate file type if it's a path
        if isinstance(data, str) and not data.startswith(("http://", "https://")):
            if "." in data:  # Has extension
                self._validate_file_type(data, [".html", ".htm"])

        # Load file asynchronously if it's a string path/URL (but not HTML content)
        if isinstance(data, str) and not data.startswith(("<!DOCTYPE", "<html", "<?xml")):
            data = await self._aload_file(data)

        # Parse the document
        result = self._parse(data)

        # Create response
        response = ParserResponse()
        response.set_response_type("html_parse")
        response.add(result)

        return response
