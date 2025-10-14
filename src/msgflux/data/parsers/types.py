class AnyParser:
    """Base type for parsers that can handle any file format.

    This is a generic parser type that attempts to parse any supported
    file format automatically based on file extension or content detection.
    """
    parser_type = "any"


class EmailParser:
    """Type for parsers that handle email files.

    Supports parsing of email formats like .eml, .msg, and can extract:
    - Email headers (from, to, subject, date)
    - Email body (plain text and HTML)
    - Attachments
    """
    parser_type = "email"


class PdfParser:
    """Type for parsers that handle PDF documents.

    Extracts content from PDF files including:
    - Text content (converted to Markdown)
    - Images embedded in the PDF
    - Page metadata
    """
    parser_type = "pdf"


class PptxParser:
    """Type for parsers that handle PowerPoint presentations.

    Extracts content from .pptx files including:
    - Slide text and titles
    - Images and shapes
    - Speaker notes
    - Slide metadata
    """
    parser_type = "pptx"


class XlsxParser:
    """Type for parsers that handle Excel spreadsheets.

    Extracts content from .xlsx files including:
    - Table data (converted to Markdown or HTML tables)
    - Multiple sheets
    - Images and charts
    - Cell metadata
    """
    parser_type = "xlsx"


class DocxParser:
    """Type for parsers that handle Word documents.

    Extracts content from .docx files including:
    - Text content with formatting
    - Paragraphs and headings
    - Tables (converted to Markdown or HTML)
    - Images embedded in the document
    """
    parser_type = "docx"


class CsvParser:
    """Type for parsers that handle CSV/TSV files.

    Extracts content from delimited files including:
    - Structured tabular data
    - Support for different delimiters
    - Header detection
    - Conversion to Markdown or HTML tables
    """
    parser_type = "csv"


class HtmlParser:
    """Type for parsers that handle HTML documents.

    Extracts content from HTML files including:
    - Clean text extraction (no tags)
    - Preservation of structure (headings, lists)
    - Link extraction
    - Conversion to Markdown
    """
    parser_type = "html"


class MarkdownParser:
    """Type for parsers that handle Markdown documents.

    Extracts and processes Markdown files including:
    - Structured text with formatting
    - Code blocks extraction
    - Link and image references
    - Metadata (front matter)
    """
    parser_type = "markdown"
