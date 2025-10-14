from typing import Literal

from msgflux._private.response import BaseResponse


class ParserResponse(BaseResponse):
    """Response object for parser operations.

    Contains the parsed content from documents including:
    - text: The main text content (usually in Markdown format)
    - images: Dictionary mapping image names to their data (base64 or bytes)
    - metadata: Additional information about the parsed document

    Example:
        >>> parser = Parser.pdf("pypdf")
        >>> response = parser("document.pdf")
        >>> print(response.data["text"])  # Markdown content
        >>> print(response.data["images"])  # {"image1.png": <bytes>}
        >>> print(response.metadata)  # {"num_pages": 10, ...}
    """
    response_type: Literal[
        "any_parse",
        "email_parse",
        "pdf_parse",
        "pptx_parse",
        "xlsx_parse",
        "docx_parse",
        "csv_parse",
        "html_parse",
        "markdown_parse",
    ]
