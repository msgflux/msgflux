from typing import Dict, Optional, Union

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

from msgflux.data.parsers.base import BaseParser
from msgflux.data.parsers.registry import register_parser
from msgflux.data.parsers.response import ParserResponse
from msgflux.data.parsers.types import PdfParser
from msgflux.dotdict import dotdict


@register_parser
class PyPDFPdfParser(BaseParser, PdfParser):
    """PyPDF-based PDF Parser.

    Converts PDF documents to Markdown format and extracts embedded images.
    Uses pypdf library for PDF processing.

    Features:
    - Text extraction with layout preservation
    - Image extraction from PDF pages
    - Page-level organization
    - Markdown output format

    Example:
        >>> parser = Parser.pdf("pypdf")
        >>> response = parser("document.pdf")
        >>> print(response.data["text"])
        >>> print(response.data["images"])
    """

    provider = "pypdf"

    def __init__(
        self,
        *,
        extraction_mode: Optional[str] = "layout",
        encode_images_base64: Optional[bool] = True,
    ):
        """Initialize PyPDF parser.

        Args:
            extraction_mode:
                Text extraction mode. Options: "layout" or "plain".
                "layout" preserves text positioning.
            encode_images_base64:
                If True, encode images as base64 strings.
                If False, keep as raw bytes.
        """
        if PdfReader is None:
            raise ImportError(
                "`pypdf` is not available. Install with `pip install pypdf`"
            )

        self.extraction_mode = extraction_mode
        self.encode_images_base64 = encode_images_base64
        self._initialize()

    def _initialize(self):
        """Initialize parser state."""
        pass

    def __call__(
        self, data: Union[str, bytes], **kwargs
    ) -> ParserResponse:
        """Parse a PDF document.

        Args:
            data:
                PDF file path, URL, or bytes.
            **kwargs:
                Additional parsing options (currently unused).

        Returns:
            ParserResponse containing:
            - text: Markdown-formatted content
            - images: Dictionary of extracted images
            - metadata: Document metadata (num_pages, etc.)

        Raises:
            FileNotFoundError:
                If file path doesn't exist.
            ValueError:
                If data type is not supported.
        """
        # Validate file type if it's a path
        if isinstance(data, str) and not data.startswith(("http://", "https://")):
            self._validate_file_type(data, ".pdf")

        # Parse the document
        result = self._parse(data)

        # Create response
        response = ParserResponse()
        response.set_response_type("pdf_parse")
        response.add(result)

        return response

    def _parse(self, data: Union[str, bytes]) -> Dict[str, any]:
        """Parse PDF and extract content.

        Args:
            data:
                PDF file path or bytes.

        Returns:
            Dictionary with:
            - text: Markdown content
            - images: Dictionary of images
            - metadata: Document metadata
        """
        md_content = ""
        images_dict = {}

        # Load PDF
        if isinstance(data, bytes):
            from io import BytesIO
            reader = PdfReader(BytesIO(data))
        else:
            reader = PdfReader(data)

        num_pages = len(reader.pages)

        # Extract content from each page
        for idx in range(num_pages):
            page = reader.pages[idx]

            # Add page marker
            md_content += f"\n\n<!-- Page number: {idx + 1} -->\n"

            # Extract text
            text = page.extract_text(
                extraction_mode=self.extraction_mode,
                layout_mode_space_vertically=False,
            )
            md_content += text.strip() + "\n"

            # Extract images from page
            for count, image_file_object in enumerate(page.images):
                img_extension = image_file_object.name.split(".")[-1]
                img_name = f"image_page{idx + 1}_{count}.{img_extension}"
                img_data = image_file_object.data

                images_dict[img_name] = img_data

                # Add image reference in markdown
                md_content += f"\n![{img_name}]({img_name})\n"

        # Prepare images (optionally encode to base64)
        images_dict = self._prepare_images_dict(
            images_dict, encode_base64=self.encode_images_base64
        )

        # Prepare metadata
        metadata = dotdict({
            "num_pages": num_pages,
            "extraction_mode": self.extraction_mode,
        })

        # Try to extract PDF metadata if available
        if reader.metadata:
            pdf_metadata = {}
            if reader.metadata.title:
                pdf_metadata["title"] = reader.metadata.title
            if reader.metadata.author:
                pdf_metadata["author"] = reader.metadata.author
            if reader.metadata.creator:
                pdf_metadata["creator"] = reader.metadata.creator
            if reader.metadata.subject:
                pdf_metadata["subject"] = reader.metadata.subject
            if pdf_metadata:
                metadata.pdf_info = pdf_metadata

        return {
            "text": md_content.strip(),
            "images": images_dict,
            "metadata": metadata,
        }

    async def acall(
        self, data: Union[str, bytes], **kwargs
    ) -> ParserResponse:
        """Async version of __call__. Parse a PDF document asynchronously.

        Args:
            data:
                PDF file path, URL, or bytes.
            **kwargs:
                Additional parsing options (currently unused).

        Returns:
            ParserResponse containing:
            - text: Markdown-formatted content
            - images: Dictionary of extracted images
            - metadata: Document metadata (num_pages, etc.)

        Raises:
            FileNotFoundError:
                If file path doesn't exist.
            ValueError:
                If data type is not supported.
        """
        # Validate file type if it's a path
        if isinstance(data, str) and not data.startswith(("http://", "https://")):
            self._validate_file_type(data, ".pdf")

        # Load file asynchronously if needed
        if isinstance(data, str):
            data = await self._aload_file(data)

        # Parse the document (parsing itself is synchronous, but we loaded the file async)
        result = self._parse(data)

        # Create response
        response = ParserResponse()
        response.set_response_type("pdf_parse")
        response.add(result)

        return response
