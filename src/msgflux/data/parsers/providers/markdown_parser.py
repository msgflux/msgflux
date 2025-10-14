import re
from typing import Dict, List, Optional, Union

from msgflux.data.parsers.base import BaseParser
from msgflux.data.parsers.registry import register_parser
from msgflux.data.parsers.response import ParserResponse
from msgflux.data.parsers.types import MarkdownParser
from msgflux.dotdict import dotdict


@register_parser
class StandardMarkdownParser(BaseParser, MarkdownParser):
    """Standard Markdown Parser.

    Parses Markdown documents and extracts structured content.
    Supports YAML front matter and extracts metadata.

    Features:
    - Front matter extraction (YAML, TOML, JSON)
    - Code block extraction
    - Link and image reference extraction
    - Heading hierarchy
    - Metadata extraction

    Example:
        >>> parser = Parser.markdown("markdown", extract_code_blocks=True)
        >>> response = parser("README.md")
        >>> print(response.data["text"])
        >>> print(response.data["code_blocks"])
        >>> print(response.data["front_matter"])
    """

    provider = "markdown"

    def __init__(
        self,
        *,
        extract_code_blocks: Optional[bool] = True,
        extract_links: Optional[bool] = True,
        extract_images: Optional[bool] = True,
        parse_front_matter: Optional[bool] = True,
    ):
        """Initialize Markdown parser.

        Args:
            extract_code_blocks:
                If True, extract all code blocks with language info.
            extract_links:
                If True, extract all links from the document.
            extract_images:
                If True, extract image references.
            parse_front_matter:
                If True, parse YAML/TOML/JSON front matter.
        """
        self.extract_code_blocks = extract_code_blocks
        self.extract_links = extract_links
        self.extract_images = extract_images
        self.parse_front_matter = parse_front_matter
        self._initialize()

    def _initialize(self):
        """Initialize parser state."""
        pass

    def __call__(
        self, data: Union[str, bytes], **kwargs
    ) -> ParserResponse:
        """Parse a Markdown document.

        Args:
            data:
                Markdown file path, URL, or string content.
            **kwargs:
                Additional parsing options (currently unused).

        Returns:
            ParserResponse containing:
            - text: Markdown content (without front matter)
            - front_matter: Parsed front matter (dict)
            - code_blocks: List of code blocks
            - links: List of extracted links
            - images: List of image references
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
                self._validate_file_type(data, [".md", ".markdown", ".txt"])

        # Parse the document
        result = self._parse(data)

        # Create response
        response = ParserResponse()
        response.set_response_type("markdown_parse")
        response.add(result)

        return response

    def _parse(self, data: Union[str, bytes]) -> Dict[str, any]:
        """Parse Markdown and extract content.

        Args:
            data:
                Markdown file path, bytes, or string content.

        Returns:
            Dictionary with:
            - text: Markdown content
            - front_matter: Parsed front matter
            - code_blocks: List of code blocks
            - links: List of links
            - images: List of image references
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
            elif data.startswith(("#", "-", "*", "```")):
                # Already Markdown content
                content = data
            else:
                # Load from file path
                with open(data, "r", encoding="utf-8") as f:
                    content = f.read()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Parse front matter
        front_matter = None
        if self.parse_front_matter:
            content, front_matter = self._extract_front_matter(content)

        # Extract code blocks
        code_blocks = []
        if self.extract_code_blocks:
            code_blocks = self._extract_code_blocks(content)

        # Extract links
        links = []
        if self.extract_links:
            links = self._extract_links(content)

        # Extract images
        images = []
        if self.extract_images:
            images = self._extract_images(content)

        # Extract headings for structure
        headings = self._extract_headings(content)

        # Prepare metadata
        metadata = dotdict({
            "has_front_matter": front_matter is not None,
            "num_code_blocks": len(code_blocks),
            "num_links": len(links),
            "num_images": len(images),
            "num_headings": len(headings),
            "heading_levels": [h["level"] for h in headings],
        })

        return {
            "text": content.strip(),
            "front_matter": front_matter,
            "code_blocks": code_blocks if self.extract_code_blocks else None,
            "links": links if self.extract_links else None,
            "images": images if self.extract_images else None,
            "headings": headings,
            "metadata": metadata,
        }

    def _extract_front_matter(self, content: str) -> tuple[str, Optional[Dict]]:
        """Extract YAML/TOML/JSON front matter from Markdown.

        Args:
            content: Markdown content.

        Returns:
            Tuple of (content without front matter, front matter dict).
        """
        # Check for YAML front matter (--- ... ---)
        yaml_pattern = r"^---\n(.*?)\n---\n"
        match = re.match(yaml_pattern, content, re.DOTALL)

        if match:
            front_matter_str = match.group(1)
            content = content[match.end():]

            # Try to parse YAML
            try:
                import yaml
                front_matter = yaml.safe_load(front_matter_str)
                return content, front_matter
            except ImportError:
                # YAML not available, return as dict with raw string
                return content, {"raw": front_matter_str}
            except Exception:
                # Parse error
                return content, {"raw": front_matter_str}

        # Check for TOML front matter (+++ ... +++)
        toml_pattern = r"^\+\+\+\n(.*?)\n\+\+\+\n"
        match = re.match(toml_pattern, content, re.DOTALL)

        if match:
            front_matter_str = match.group(1)
            content = content[match.end():]
            return content, {"raw": front_matter_str, "format": "toml"}

        return content, None

    def _extract_code_blocks(self, content: str) -> List[Dict]:
        """Extract code blocks from Markdown.

        Args:
            content: Markdown content.

        Returns:
            List of dicts with code block info.
        """
        code_blocks = []

        # Match fenced code blocks (``` ... ```)
        pattern = r"```(\w*)\n(.*?)\n```"
        for match in re.finditer(pattern, content, re.DOTALL):
            language = match.group(1) or "text"
            code = match.group(2)
            code_blocks.append({
                "language": language,
                "code": code,
                "start_pos": match.start(),
                "end_pos": match.end(),
            })

        return code_blocks

    def _extract_links(self, content: str) -> List[Dict]:
        """Extract links from Markdown.

        Args:
            content: Markdown content.

        Returns:
            List of dicts with link info.
        """
        links = []

        # Match inline links [text](url)
        inline_pattern = r"\[([^\]]+)\]\(([^\)]+)\)"
        for match in re.finditer(inline_pattern, content):
            links.append({
                "text": match.group(1),
                "url": match.group(2),
                "type": "inline",
            })

        # Match reference links [text][ref]
        ref_pattern = r"\[([^\]]+)\]\[([^\]]+)\]"
        for match in re.finditer(ref_pattern, content):
            links.append({
                "text": match.group(1),
                "reference": match.group(2),
                "type": "reference",
            })

        return links

    def _extract_images(self, content: str) -> List[Dict]:
        """Extract image references from Markdown.

        Args:
            content: Markdown content.

        Returns:
            List of dicts with image info.
        """
        images = []

        # Match inline images ![alt](url)
        pattern = r"!\[([^\]]*)\]\(([^\)]+)\)"
        for match in re.finditer(pattern, content):
            images.append({
                "alt": match.group(1),
                "url": match.group(2),
            })

        return images

    def _extract_headings(self, content: str) -> List[Dict]:
        """Extract headings from Markdown.

        Args:
            content: Markdown content.

        Returns:
            List of dicts with heading info.
        """
        headings = []

        # Match ATX-style headings (# ... ######)
        pattern = r"^(#{1,6})\s+(.+)$"
        for match in re.finditer(pattern, content, re.MULTILINE):
            level = len(match.group(1))
            text = match.group(2).strip()
            headings.append({
                "level": level,
                "text": text,
            })

        return headings

    async def acall(
        self, data: Union[str, bytes], **kwargs
    ) -> ParserResponse:
        """Async version of __call__. Parse a Markdown document asynchronously.

        Args:
            data:
                Markdown file path, URL, or string content.
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
                self._validate_file_type(data, [".md", ".markdown", ".txt"])

        # Load file asynchronously if it's a string path/URL (but not Markdown content)
        if isinstance(data, str) and not data.startswith(("#", "-", "*", "```")):
            data = await self._aload_file(data)

        # Parse the document
        result = self._parse(data)

        # Create response
        response = ParserResponse()
        response.set_response_type("markdown_parse")
        response.add(result)

        return response
