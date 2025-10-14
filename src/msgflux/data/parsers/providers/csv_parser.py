import csv
from io import StringIO
from typing import Dict, List, Optional, Union

from msgflux.data.parsers.base import BaseParser
from msgflux.data.parsers.registry import register_parser
from msgflux.data.parsers.response import ParserResponse
from msgflux.data.parsers.types import CsvParser
from msgflux.dotdict import dotdict


@register_parser
class StandardCsvParser(BaseParser, CsvParser):
    """Standard CSV/TSV Parser.

    Parses delimited files (CSV, TSV, etc.) and converts to structured format.
    Uses Python's built-in csv module.

    Features:
    - Automatic delimiter detection
    - Header row support
    - Conversion to Markdown or HTML tables
    - Support for different encodings
    - Quote character handling

    Example:
        >>> parser = Parser.csv("csv", delimiter=",", has_header=True)
        >>> response = parser("data.csv")
        >>> print(response.data["text"])
        >>> print(response.data["metadata"]["num_rows"])
    """

    provider = "csv"

    def __init__(
        self,
        *,
        delimiter: Optional[str] = None,
        has_header: Optional[bool] = True,
        table_format: Optional[str] = "markdown",
        encoding: Optional[str] = "utf-8",
        quotechar: Optional[str] = '"',
    ):
        """Initialize CSV parser.

        Args:
            delimiter:
                Field delimiter. If None, auto-detect from ",", "\t", ";", "|".
            has_header:
                Whether the first row is a header row.
            table_format:
                Output format: "markdown" or "html".
            encoding:
                File encoding (default: utf-8).
            quotechar:
                Character used to quote fields containing special characters.
        """
        self.delimiter = delimiter
        self.has_header = has_header
        self.table_format = table_format
        self.encoding = encoding
        self.quotechar = quotechar
        self._initialize()

    def _initialize(self):
        """Initialize parser state."""
        pass

    def __call__(
        self, data: Union[str, bytes], **kwargs
    ) -> ParserResponse:
        """Parse a CSV/TSV document.

        Args:
            data:
                CSV file path, URL, or bytes.
            **kwargs:
                Additional parsing options (currently unused).

        Returns:
            ParserResponse containing:
            - text: Markdown or HTML formatted table
            - metadata: Document metadata (num_rows, num_cols, etc.)

        Raises:
            FileNotFoundError:
                If file path doesn't exist.
            ValueError:
                If data type is not supported.
        """
        # Validate file type if it's a path
        if isinstance(data, str) and not data.startswith(("http://", "https://")):
            self._validate_file_type(data, [".csv", ".tsv", ".txt"])

        # Parse the document
        result = self._parse(data)

        # Create response
        response = ParserResponse()
        response.set_response_type("csv_parse")
        response.add(result)

        return response

    def _parse(self, data: Union[str, bytes]) -> Dict[str, any]:
        """Parse CSV and extract content.

        Args:
            data:
                CSV file path or bytes.

        Returns:
            Dictionary with:
            - text: Markdown/HTML formatted table
            - metadata: Document metadata
        """
        # Load content
        if isinstance(data, bytes):
            content = data.decode(self.encoding)
        elif isinstance(data, str):
            if data.startswith(("http://", "https://")):
                # Load from URL
                file_bytes = self._load_file(data)
                content = file_bytes.decode(self.encoding)
            else:
                # Load from file path
                with open(data, "r", encoding=self.encoding) as f:
                    content = f.read()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Auto-detect delimiter if not specified
        delimiter = self.delimiter
        if delimiter is None:
            delimiter = self._detect_delimiter(content)

        # Parse CSV
        reader = csv.reader(
            StringIO(content),
            delimiter=delimiter,
            quotechar=self.quotechar
        )
        rows = list(reader)

        if not rows:
            return {
                "text": "",
                "metadata": dotdict({
                    "num_rows": 0,
                    "num_cols": 0,
                    "has_header": self.has_header,
                    "delimiter": delimiter,
                })
            }

        # Extract header if specified
        header = None
        data_rows = rows
        if self.has_header and rows:
            header = rows[0]
            data_rows = rows[1:]

        # Convert to table format
        if self.table_format == "html":
            text_output = self._to_html_table(header, data_rows)
        else:  # markdown
            text_output = self._to_markdown_table(header, data_rows)

        # Prepare metadata
        num_cols = len(rows[0]) if rows else 0
        metadata = dotdict({
            "num_rows": len(data_rows),
            "num_cols": num_cols,
            "total_rows": len(rows),
            "has_header": self.has_header,
            "delimiter": delimiter,
            "table_format": self.table_format,
            "encoding": self.encoding,
        })

        return {
            "text": text_output,
            "metadata": metadata,
        }

    def _detect_delimiter(self, content: str) -> str:
        """Auto-detect the delimiter used in the CSV.

        Args:
            content: CSV file content.

        Returns:
            Detected delimiter character.
        """
        # Get first few lines for detection
        lines = content.split('\n')[:5]
        sample = '\n'.join(lines)

        # Try to detect using csv.Sniffer
        try:
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample, delimiters=',\t;|')
            return dialect.delimiter
        except Exception:
            # Fallback: count occurrences of common delimiters
            delimiters = [',', '\t', ';', '|']
            counts = {d: sample.count(d) for d in delimiters}
            return max(counts, key=counts.get)

    def _to_markdown_table(
        self, header: Optional[List[str]], rows: List[List[str]]
    ) -> str:
        """Convert data to Markdown table format.

        Args:
            header: Header row (optional).
            rows: Data rows.

        Returns:
            Markdown formatted table.
        """
        if not rows and not header:
            return ""

        lines = []

        # Add header
        if header:
            header_line = "| " + " | ".join(str(cell) for cell in header) + " |"
            separator = "| " + " | ".join(["---"] * len(header)) + " |"
            lines.extend([header_line, separator])

        # Add data rows
        for row in rows:
            # Pad row if necessary
            if header and len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            row_line = "| " + " | ".join(str(cell) for cell in row) + " |"
            lines.append(row_line)

        return "\n".join(lines)

    def _to_html_table(
        self, header: Optional[List[str]], rows: List[List[str]]
    ) -> str:
        """Convert data to HTML table format.

        Args:
            header: Header row (optional).
            rows: Data rows.

        Returns:
            HTML formatted table.
        """
        if not rows and not header:
            return ""

        html = "<table>\n"

        # Add header
        if header:
            html += "  <thead>\n    <tr>\n"
            for cell in header:
                html += f"      <th>{str(cell)}</th>\n"
            html += "    </tr>\n  </thead>\n"

        # Add body
        html += "  <tbody>\n"
        for row in rows:
            html += "    <tr>\n"
            # Pad row if necessary
            if header and len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            for cell in row:
                html += f"      <td>{str(cell)}</td>\n"
            html += "    </tr>\n"
        html += "  </tbody>\n"

        html += "</table>"
        return html

    async def acall(
        self, data: Union[str, bytes], **kwargs
    ) -> ParserResponse:
        """Async version of __call__. Parse a CSV document asynchronously.

        Args:
            data:
                CSV file path, URL, or bytes.
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
            self._validate_file_type(data, [".csv", ".tsv", ".txt"])

        # Load file asynchronously if it's a string path/URL
        if isinstance(data, str):
            data = await self._aload_file(data)

        # Parse the document
        result = self._parse(data)

        # Create response
        response = ParserResponse()
        response.set_response_type("csv_parse")
        response.add(result)

        return response
