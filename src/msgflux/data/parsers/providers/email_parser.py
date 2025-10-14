import email
from email import policy
from email.parser import BytesParser
from typing import Dict, List, Optional, Union

from msgflux.data.parsers.base import BaseParser
from msgflux.data.parsers.registry import register_parser
from msgflux.data.parsers.response import ParserResponse
from msgflux.data.parsers.types import EmailParser
from msgflux.dotdict import dotdict


@register_parser
class StandardEmailParser(BaseParser, EmailParser):
    """Standard Email Parser.

    Parses email files (.eml) and extracts headers, body, and attachments.
    Uses Python's built-in email module.

    Features:
    - Header extraction (from, to, subject, date, etc.)
    - Plain text and HTML body extraction
    - Attachment extraction
    - Multipart message handling
    - Inline image detection

    Example:
        >>> parser = Parser.email("email", extract_attachments=True)
        >>> response = parser("message.eml")
        >>> print(response.data["headers"])
        >>> print(response.data["body"])
        >>> print(response.data["attachments"])
    """

    provider = "email"

    def __init__(
        self,
        *,
        extract_attachments: Optional[bool] = True,
        extract_html: Optional[bool] = True,
        encode_attachments_base64: Optional[bool] = True,
    ):
        """Initialize Email parser.

        Args:
            extract_attachments:
                If True, extract file attachments.
            extract_html:
                If True, include HTML version of body.
            encode_attachments_base64:
                If True, encode attachments as base64 strings.
        """
        self.extract_attachments = extract_attachments
        self.extract_html = extract_html
        self.encode_attachments_base64 = encode_attachments_base64
        self._initialize()

    def _initialize(self):
        """Initialize parser state."""
        pass

    def __call__(
        self, data: Union[str, bytes], **kwargs
    ) -> ParserResponse:
        """Parse an email document.

        Args:
            data:
                Email file path, URL, or bytes.
            **kwargs:
                Additional parsing options (currently unused).

        Returns:
            ParserResponse containing:
            - headers: Dictionary of email headers
            - body: Plain text body
            - html_body: HTML body (if extract_html=True)
            - attachments: List of attachments
            - metadata: Document metadata

        Raises:
            FileNotFoundError:
                If file path doesn't exist.
            ValueError:
                If data type is not supported.
        """
        # Validate file type if it's a path
        if isinstance(data, str) and not data.startswith(("http://", "https://")):
            self._validate_file_type(data, [".eml", ".msg", ".txt"])

        # Parse the document
        result = self._parse(data)

        # Create response
        response = ParserResponse()
        response.set_response_type("email_parse")
        response.add(result)

        return response

    def _parse(self, data: Union[str, bytes]) -> Dict[str, any]:
        """Parse email and extract content.

        Args:
            data:
                Email file path or bytes.

        Returns:
            Dictionary with:
            - headers: Email headers
            - body: Plain text body
            - html_body: HTML body
            - attachments: List of attachments
            - metadata: Document metadata
        """
        # Load content as bytes
        if isinstance(data, str):
            if data.startswith(("http://", "https://")):
                # Load from URL
                msg_bytes = self._load_file(data)
            else:
                # Load from file path
                with open(data, "rb") as f:
                    msg_bytes = f.read()
        elif isinstance(data, bytes):
            msg_bytes = data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Parse email
        msg = BytesParser(policy=policy.default).parsebytes(msg_bytes)

        # Extract headers
        headers = {
            "from": msg.get("From", ""),
            "to": msg.get("To", ""),
            "cc": msg.get("Cc", ""),
            "bcc": msg.get("Bcc", ""),
            "subject": msg.get("Subject", ""),
            "date": msg.get("Date", ""),
            "message_id": msg.get("Message-ID", ""),
            "in_reply_to": msg.get("In-Reply-To", ""),
            "references": msg.get("References", ""),
        }

        # Extract body
        plain_body = ""
        html_body = ""
        attachments = []

        if msg.is_multipart():
            # Handle multipart messages
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))

                # Skip multipart containers
                if part.is_multipart():
                    continue

                # Extract text body
                if content_type == "text/plain" and "attachment" not in content_disposition:
                    try:
                        plain_body += part.get_content()
                    except Exception:
                        pass

                # Extract HTML body
                elif content_type == "text/html" and "attachment" not in content_disposition:
                    if self.extract_html:
                        try:
                            html_body += part.get_content()
                        except Exception:
                            pass

                # Extract attachments
                elif self.extract_attachments and "attachment" in content_disposition:
                    filename = part.get_filename()
                    if filename:
                        try:
                            attachment_data = part.get_payload(decode=True)
                            if self.encode_attachments_base64:
                                attachment_data = self._encode_image_to_base64(attachment_data)

                            attachments.append({
                                "filename": filename,
                                "content_type": content_type,
                                "data": attachment_data,
                            })
                        except Exception:
                            pass
        else:
            # Handle simple messages
            try:
                content = msg.get_content()
                if msg.get_content_type() == "text/html":
                    html_body = content
                else:
                    plain_body = content
            except Exception:
                plain_body = str(msg.get_payload())

        # Create Markdown output
        markdown = self._create_markdown_output(headers, plain_body, html_body)

        # Prepare metadata
        metadata = dotdict({
            "has_html": bool(html_body),
            "num_attachments": len(attachments),
            "is_multipart": msg.is_multipart(),
            "content_type": msg.get_content_type(),
        })

        return {
            "text": markdown,
            "headers": headers,
            "body": plain_body,
            "html_body": html_body if self.extract_html else None,
            "attachments": attachments if self.extract_attachments else None,
            "metadata": metadata,
        }

    def _create_markdown_output(
        self, headers: Dict, plain_body: str, html_body: str
    ) -> str:
        """Create Markdown formatted output from email content.

        Args:
            headers: Email headers dict.
            plain_body: Plain text body.
            html_body: HTML body.

        Returns:
            Markdown formatted string.
        """
        lines = []

        # Add headers
        lines.append("# Email")
        lines.append("")
        lines.append(f"**From:** {headers.get('from', 'N/A')}")
        lines.append(f"**To:** {headers.get('to', 'N/A')}")
        if headers.get('cc'):
            lines.append(f"**Cc:** {headers['cc']}")
        lines.append(f"**Subject:** {headers.get('subject', 'N/A')}")
        lines.append(f"**Date:** {headers.get('date', 'N/A')}")
        lines.append("")

        # Add body
        lines.append("## Body")
        lines.append("")
        lines.append(plain_body)

        return "\n".join(lines)

    async def acall(
        self, data: Union[str, bytes], **kwargs
    ) -> ParserResponse:
        """Async version of __call__. Parse an email document asynchronously.

        Args:
            data:
                Email file path, URL, or bytes.
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
            self._validate_file_type(data, [".eml", ".msg", ".txt"])

        # Load file asynchronously if it's a string path/URL
        if isinstance(data, str):
            data = await self._aload_file(data)

        # Parse the document
        result = self._parse(data)

        # Create response
        response = ParserResponse()
        response.set_response_type("email_parse")
        response.add(result)

        return response
