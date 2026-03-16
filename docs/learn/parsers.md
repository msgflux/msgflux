# Parsers

## ✦₊⁺ Overview

The `Parser` class provides a unified interface for extracting text, images, and metadata from a wide variety of document formats. Every parser converts its input into a consistent `ParserResponse` with a `data` dict containing at minimum a `text` field in Markdown format.

All code examples use the recommended import pattern:

```python
import msgflux as mf
```

---

## 1. **Supported Formats**

| Format | Factory method | Provider |
|--------|---------------|----------|
| PDF | `mf.Parser.pdf()` | `"pypdf"` |
| Word (.docx) | `mf.Parser.docx()` | `"python_docx"` |
| PowerPoint (.pptx) | `mf.Parser.pptx()` | `"python_pptx"` |
| Excel (.xlsx) | `mf.Parser.xlsx()` | `"openpyxl"` |
| CSV / TSV | `mf.Parser.csv()` | `"csv"` |
| HTML | `mf.Parser.html()` | `"beautifulsoup"` |
| Email (.eml) | `mf.Parser.email()` | `"email"` |

For installation instructions see [Dependency Management](../../dependency-management.md#parsers).

To inspect available providers at runtime:

```python
mf.Parser.providers()
# {"pdf": ["pypdf"], "docx": ["python_docx"], ...}
```

---

## 2. **ParserResponse**

Every parser returns a `ParserResponse`. Its main attributes are:

| Attribute | Type | Description |
|-----------|------|-------------|
| `data` | `dict` | Extracted content (keys vary per parser, see below) |
| `metadata` | - | Mirrored from `data["metadata"]` |
| `response_type` | `str` | E.g. `"pdf_parse"`, `"csv_parse"`, … |

### Common `data` keys

| Key | Present in | Description |
|-----|-----------|-------------|
| `text` | all parsers | Markdown-formatted document text |
| `images` | pdf, docx, pptx, xlsx | `{filename: base64_str_or_bytes}` |
| `metadata` | all parsers | `dotdict` with parser-specific info |

---

## 3. **PDF**

```python
from io import BytesIO
import msgflux as mf
from pypdf import PdfWriter

# Create a minimal PDF in memory
writer = PdfWriter()
writer.add_blank_page(width=612, height=792)
buf = BytesIO()
writer.write(buf)

# Parse
parser = mf.Parser.pdf("pypdf")
response = parser(buf.getvalue())

print(response.data["text"])
print(response.data["metadata"].num_pages)        # 1
print(response.data["metadata"].extraction_mode)  # "layout"
```

**`data` keys:** `text`, `images`, `metadata`

**`metadata` fields:** `num_pages`, `extraction_mode`, `pdf_info` *(optional — title, author, subject)*

---

## 4. **Word (.docx)**

```python
from io import BytesIO
import msgflux as mf
from docx import Document

# Create a minimal DOCX in memory
doc = Document()
doc.add_heading("Hello World", 0)
doc.add_paragraph("This is a paragraph.")
table = doc.add_table(rows=2, cols=2)
table.cell(0, 0).text = "Name"
table.cell(0, 1).text = "Value"
table.cell(1, 0).text = "Alice"
table.cell(1, 1).text = "42"
buf = BytesIO()
doc.save(buf)

# Parse
parser = mf.Parser.docx("python_docx")
response = parser(buf.getvalue())

print(response.data["text"])
# # Hello World
#
# This is a paragraph.
#
# | Name | Value |
# | --- | --- |
# | Alice | 42 |

print(response.data["metadata"].num_tables)  # 1
```

**`data` keys:** `text`, `images`, `metadata`

**`metadata` fields:** `num_paragraphs`, `num_tables`, `num_images`, `table_format`, `docx_info` *(optional)*

---

## 5. **PowerPoint (.pptx)**

```python
from io import BytesIO
import msgflux as mf
from pptx import Presentation

# Create a minimal PPTX in memory
prs = Presentation()
slide = prs.slides.add_slide(prs.slide_layouts[0])
slide.shapes.title.text = "Hello World"
slide.placeholders[1].text = "Subtitle text"
buf = BytesIO()
prs.save(buf)

# Parse
parser = mf.Parser.pptx("python_pptx")
response = parser(buf.getvalue())

print(response.data["text"])
# <!-- Slide number: 1 -->
# # Hello World
#
# Subtitle text

print(response.data["metadata"].num_slides)  # 1
```

**`data` keys:** `text`, `images`, `metadata`

**`metadata` fields:** `num_slides`, `include_notes`, `pptx_info` *(optional)*

---

## 6. **Excel (.xlsx)**

```python
from io import BytesIO
import msgflux as mf
from openpyxl import Workbook

# Create a minimal XLSX in memory
wb = Workbook()
ws = wb.active
ws.append(["Name", "Value"])
ws.append(["Alice", 42])
ws.append(["Bob", 7])
buf = BytesIO()
wb.save(buf)

# Parse
parser = mf.Parser.xlsx("openpyxl")
response = parser(buf.getvalue())

print(response.data["text"])
# <!-- Sheet: Sheet -->
# | Name | Value |
# | --- | --- |
# | Alice | 42 |
# | Bob | 7 |

print(response.data["metadata"].num_sheets)  # 1
```

**`data` keys:** `text`, `images`, `metadata`

**`metadata` fields:** `num_sheets`, `sheet_names`, `table_format`

---

## 7. **CSV / TSV**

```python
import msgflux as mf

csv_content = b"product,qty,price\nwidget,10,5.99\ngadget,3,19.99\n"

parser = mf.Parser.csv("csv")
response = parser(csv_content)

print(response.data["text"])
# | product | qty | price |
# | --- | --- | --- |
# | widget | 10 | 5.99 |
# | gadget | 3 | 19.99 |

print(response.data["metadata"].num_rows)   # 2
print(response.data["metadata"].num_cols)   # 3
print(response.data["metadata"].delimiter)  # ","
```

**`data` keys:** `text`, `metadata`

**`metadata` fields:** `num_rows`, `num_cols`, `total_rows`, `has_header`, `delimiter`, `table_format`, `encoding`

---

## 8. **HTML**

```python
import msgflux as mf

html_content = """\
<html>
  <head><title>My Page</title></head>
  <body>
    <h1>Hello World</h1>
    <p>This is a paragraph.</p>
    <a href="https://example.com">Click here</a>
  </body>
</html>"""

parser = mf.Parser.html("beautifulsoup")
response = parser(html_content)

print(response.data["text"])
# ## Hello World
#
# This is a paragraph.

print(response.data["links"])
# [{"text": "Click here", "url": "https://example.com"}]

print(response.data["metadata"].title)  # "My Page"
```

**`data` keys:** `text`, `links`, `images`, `metadata`

**`metadata` fields:** `title`, `num_links`, `num_images`, `extract_links`, `extract_images`

---

## 9. **Email (.eml)**

```python
import msgflux as mf

email_content = (
    b"From: alice@example.com\r\n"
    b"To: bob@example.com\r\n"
    b"Subject: Hello\r\n"
    b"Date: Mon, 1 Jan 2024 10:00:00 +0000\r\n"
    b"\r\n"
    b"Hi Bob, this is a test email."
)

parser = mf.Parser.email("email")
response = parser(email_content)

print(response.data["text"])
# # Email
#
# **From:** alice@example.com
# **To:** bob@example.com
# **Subject:** Hello
# **Date:** Mon, 1 Jan 2024 10:00:00 +0000
#
# ## Body
# Hi Bob, this is a test email.

print(response.data["headers"]["subject"])  # "Hello"
print(response.data["body"])                # "Hi Bob, this is a test email."
```

**`data` keys:** `text`, `headers`, `body`, `html_body`, `attachments`, `metadata`

**`metadata` fields:** `has_html`, `num_attachments`, `is_multipart`, `content_type`
