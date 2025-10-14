"""Parser providers package.

This package contains implementations of parsers for different file formats.
Each parser provider follows the standard msgflux parser interface.

Available parsers:
- PyPDFPdfParser: PDF parsing using pypdf
- OpenPyxlXlsxParser: Excel parsing using openpyxl
- PythonPptxPptxParser: PowerPoint parsing using python-pptx
- PythonDocxDocxParser: Word document parsing using python-docx
- StandardCsvParser: CSV/TSV parsing using Python csv module
- BeautifulSoupHtmlParser: HTML parsing using BeautifulSoup4
- StandardMarkdownParser: Markdown parsing with metadata extraction
- StandardEmailParser: Email (.eml) parsing using Python email module
"""

from msgflux.data.parsers.providers.csv_parser import StandardCsvParser
from msgflux.data.parsers.providers.email_parser import StandardEmailParser
from msgflux.data.parsers.providers.html_parser import BeautifulSoupHtmlParser
from msgflux.data.parsers.providers.markdown_parser import StandardMarkdownParser
from msgflux.data.parsers.providers.openpyxl import OpenPyxlXlsxParser
from msgflux.data.parsers.providers.pypdf import PyPDFPdfParser
from msgflux.data.parsers.providers.python_docx import PythonDocxDocxParser
from msgflux.data.parsers.providers.python_pptx import PythonPptxPptxParser

__all__ = [
    "PyPDFPdfParser",
    "OpenPyxlXlsxParser",
    "PythonPptxPptxParser",
    "PythonDocxDocxParser",
    "StandardCsvParser",
    "BeautifulSoupHtmlParser",
    "StandardMarkdownParser",
    "StandardEmailParser",
]
