"""Tests for data parsers."""

import base64
from contextlib import contextmanager
from io import BytesIO
import sys
import types

import pytest

import msgflux as mf


def _install_fake_tiktoken(monkeypatch):
    class Encoding:
        def encode(self, text):
            return list(text)

        def decode(self, tokens):
            return "".join(tokens)

    tiktoken = types.ModuleType("tiktoken")
    tiktoken.get_encoding = lambda _name: Encoding()
    monkeypatch.setitem(sys.modules, "tiktoken", tiktoken)


# ---------------------------------------------------------------------------
# Fixtures — all generated in-memory using the target libraries themselves
# ---------------------------------------------------------------------------


@pytest.fixture
def csv_bytes():
    return b"name,age,city\nAlice,30,NYC\nBob,25,LA\n"


@pytest.fixture
def html_bytes():
    return (
        b"<html><head><title>Test Page</title></head>"
        b"<body><h1>Hello</h1><p>World</p>"
        b'<a href="https://example.com">Link</a>'
        b'<img src="/img/logo.png" alt="logo">'
        b"</body></html>"
    )


@pytest.fixture
def email_bytes():
    return (
        b"From: alice@example.com\r\n"
        b"To: bob@example.com\r\n"
        b"Subject: Test Email\r\n"
        b"Date: Mon, 1 Jan 2024 10:00:00 +0000\r\n"
        b"\r\n"
        b"Hello Bob, this is a test email."
    )


@pytest.fixture
def pdf_bytes():
    pypdf = pytest.importorskip("pypdf")
    # Minimal valid PDF with a text content stream
    content = b"BT /F1 12 Tf 100 700 Td (Hello World) Tj ET"
    content_len = len(content)
    raw = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R"
        b" /MediaBox [0 0 612 792]"
        b" /Contents 4 0 R"
        b" /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length "
        + str(content_len).encode()
        + b" >>\nstream\n"
        + content
        + b"\nendstream\nendobj\n"
        b"xref\n0 5\n"
        b"0000000000 65535 f \n"
        b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n9\n%%EOF\n"
    )
    # Use pypdf's writer for a well-formed PDF instead
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


@pytest.fixture
def docx_bytes():
    pytest.importorskip("docx")
    from docx import Document

    doc = Document()
    doc.add_heading("Test Document", 0)
    doc.add_paragraph("This is a test paragraph.")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Col A"
    table.cell(0, 1).text = "Col B"
    table.cell(1, 0).text = "val 1"
    table.cell(1, 1).text = "val 2"
    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()


@pytest.fixture
def pptx_bytes():
    pytest.importorskip("pptx")
    from pptx import Presentation

    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "Test Slide"
    slide.placeholders[1].text = "Subtitle text"
    buf = BytesIO()
    prs.save(buf)
    return buf.getvalue()


@pytest.fixture
def xlsx_bytes():
    pytest.importorskip("openpyxl")
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.append(["Name", "Value"])
    ws.append(["Alice", 42])
    ws.append(["Bob", 7])
    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def test_csv_response_structure(csv_bytes):
    parser = mf.Parser.csv("csv")
    response = parser(csv_bytes)

    assert response.response_type == "csv_parse"
    assert "text" in response.data
    assert "metadata" in response.data


def test_csv_text_is_markdown_table(csv_bytes):
    parser = mf.Parser.csv("csv")
    response = parser(csv_bytes)

    assert "name" in response.data["text"]
    assert "Alice" in response.data["text"]
    assert "|" in response.data["text"]


def test_csv_metadata(csv_bytes):
    parser = mf.Parser.csv("csv")
    response = parser(csv_bytes)

    meta = response.data["metadata"]
    assert meta.num_rows == 2
    assert meta.num_cols == 3
    assert meta.has_header is True


def test_csv_html_format(csv_bytes):
    parser = mf.Parser.csv("csv", table_format="html")
    response = parser(csv_bytes)

    assert "<table>" in response.data["text"]
    assert "<th>" in response.data["text"]


def test_csv_serialized_roundtrip_restores_usable_parser():
    parser = mf.Parser.csv(
        "csv",
        delimiter=";",
        has_header=True,
        table_format="html",
        encoding="utf-8",
        quotechar="'",
    )
    serialized = parser.serialize()

    restored = mf.Parser.from_serialized(
        serialized["provider"], serialized["parser_type"], serialized["state"]
    )
    response = restored(b"name;score\nAda;10\nGrace;9\n")

    assert response.response_type == "csv_parse"
    assert response.data["metadata"].delimiter == ";"
    assert response.data["metadata"].table_format == "html"
    assert "<table>" in response.data["text"]
    assert "Ada" in response.data["text"]


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


def test_html_response_structure(html_bytes):
    pytest.importorskip("bs4")
    parser = mf.Parser.html("beautifulsoup")
    response = parser(html_bytes)

    assert response.response_type == "html_parse"
    assert "text" in response.data
    assert "links" in response.data
    assert "images" in response.data
    assert "metadata" in response.data


def test_html_text_extraction(html_bytes):
    pytest.importorskip("bs4")
    parser = mf.Parser.html("beautifulsoup")
    response = parser(html_bytes)

    assert "Hello" in response.data["text"]


def test_html_links_extraction(html_bytes):
    pytest.importorskip("bs4")
    parser = mf.Parser.html("beautifulsoup")
    response = parser(html_bytes)

    assert len(response.data["links"]) == 1
    assert response.data["links"][0]["url"] == "https://example.com"


def test_html_images_extraction(html_bytes):
    pytest.importorskip("bs4")
    parser = mf.Parser.html("beautifulsoup")
    response = parser(html_bytes)

    assert len(response.data["images"]) == 1
    assert response.data["images"][0]["alt"] == "logo"


def test_html_metadata(html_bytes):
    pytest.importorskip("bs4")
    parser = mf.Parser.html("beautifulsoup")
    response = parser(html_bytes)

    assert response.data["metadata"].title == "Test Page"
    assert response.data["metadata"].num_links == 1


def test_html_raw_string():
    pytest.importorskip("bs4")
    parser = mf.Parser.html("beautifulsoup")
    response = parser("<html><body><h1>Direct HTML</h1><p>Content</p></body></html>")

    assert "Direct HTML" in response.data["text"]


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------


def test_email_response_structure(email_bytes):
    parser = mf.Parser.email("email")
    response = parser(email_bytes)

    assert response.response_type == "email_parse"
    assert "text" in response.data
    assert "headers" in response.data
    assert "body" in response.data
    assert "metadata" in response.data


def test_email_headers(email_bytes):
    parser = mf.Parser.email("email")
    response = parser(email_bytes)

    headers = response.data["headers"]
    assert "alice@example.com" in headers["from"]
    assert "bob@example.com" in headers["to"]
    assert headers["subject"] == "Test Email"


def test_email_body(email_bytes):
    parser = mf.Parser.email("email")
    response = parser(email_bytes)

    assert "Hello Bob" in response.data["body"]


def test_email_text_is_markdown(email_bytes):
    parser = mf.Parser.email("email")
    response = parser(email_bytes)

    assert "# Email" in response.data["text"]
    assert "**From:**" in response.data["text"]


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("max_tokens", "overlap_tokens", "message"),
    [
        (0, 0, "max_tokens"),
        (-1, 0, "max_tokens"),
        (1.5, 0, "max_tokens"),
        (True, 0, "max_tokens"),
        (4, -1, "overlap_tokens"),
        (4, 1.5, "overlap_tokens"),
        (4, False, "overlap_tokens"),
        (4, 4, "smaller than"),
        (4, 5, "smaller than"),
    ],
)
def test_chunk_by_tokens_rejects_invalid_bounds(max_tokens, overlap_tokens, message):
    parser = mf.Parser.csv("csv")

    with pytest.raises(ValueError, match=message):
        parser.chunk_by_tokens(
            "some text", max_tokens=max_tokens, overlap_tokens=overlap_tokens
        )


def test_chunk_by_tokens_makes_progress_and_preserves_boundaries(monkeypatch):
    _install_fake_tiktoken(monkeypatch)
    parser = mf.Parser.csv("csv")

    assert parser.chunk_by_tokens("abcdefgh", max_tokens=4, overlap_tokens=1) == [
        "abcd",
        "defg",
        "gh",
    ]
    assert parser.chunk_by_tokens("abcd", max_tokens=4, overlap_tokens=1) == ["abcd"]
    assert parser.chunk_by_tokens("", max_tokens=4, overlap_tokens=1) == [""]


def test_chunk_by_tokens_fallback_accepts_valid_bounds(monkeypatch):
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    parser = mf.Parser.csv("csv")

    assert parser.chunk_by_tokens("abcdefgh", max_tokens=2, overlap_tokens=0) == [
        "abcdefgh"
    ]
    assert parser.chunk_by_tokens("", max_tokens=2, overlap_tokens=0) == []


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def test_pdf_response_structure(pdf_bytes):
    pytest.importorskip("pypdf")
    parser = mf.Parser.pdf("pypdf")
    response = parser(pdf_bytes)

    assert response.response_type == "pdf_parse"
    assert "text" in response.data
    assert "images" in response.data
    assert "metadata" in response.data


def test_pdf_metadata(pdf_bytes):
    pytest.importorskip("pypdf")
    parser = mf.Parser.pdf("pypdf")
    response = parser(pdf_bytes)

    assert response.data["metadata"].num_pages == 1
    assert response.data["metadata"].extraction_mode == "layout"


def test_pdf_images_is_dict(pdf_bytes):
    pytest.importorskip("pypdf")
    parser = mf.Parser.pdf("pypdf")
    response = parser(pdf_bytes)

    assert isinstance(response.data["images"], dict)


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------


def test_docx_response_structure(docx_bytes):
    pytest.importorskip("docx")
    parser = mf.Parser.docx("python_docx")
    response = parser(docx_bytes)

    assert response.response_type == "docx_parse"
    assert "text" in response.data
    assert "images" in response.data
    assert "metadata" in response.data


def test_docx_heading_extraction(docx_bytes):
    pytest.importorskip("docx")
    parser = mf.Parser.docx("python_docx")
    response = parser(docx_bytes)

    assert "Test Document" in response.data["text"]


def test_docx_table_markdown(docx_bytes):
    pytest.importorskip("docx")
    parser = mf.Parser.docx("python_docx")
    response = parser(docx_bytes)

    assert "|" in response.data["text"]
    assert "Col A" in response.data["text"]


def test_docx_table_html(docx_bytes):
    pytest.importorskip("docx")
    parser = mf.Parser.docx("python_docx", table_format="html")
    response = parser(docx_bytes)

    assert "<table>" in response.data["text"]


def test_docx_metadata(docx_bytes):
    pytest.importorskip("docx")
    parser = mf.Parser.docx("python_docx")
    response = parser(docx_bytes)

    assert response.data["metadata"].num_tables == 1


def test_docx_parse_avoids_repeated_block_discovery(monkeypatch):
    pytest.importorskip("docx")
    from docx import Document
    from docx.document import Document as DocxDocument

    # Real, modest DOCX: 30 paragraphs (including empty ones), 10 tables, and an
    # image. The baseline parser rescans document.paragraphs/tables once per
    # body block, creating redundant paragraph and table wrappers quadratically.
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
        "/x8AAusB9Y9Zl1sAAAAASUVORK5CYII="
    )
    doc = Document()
    for index in range(10):
        doc.add_heading(f"Heading {index}", level=1)
        paragraph = doc.add_paragraph(f"Paragraph {index}")
        doc.add_paragraph("")
        if index == 0:
            paragraph.add_run().add_picture(BytesIO(png))
        table = doc.add_table(rows=2, cols=2)
        table.cell(0, 0).text = f"A{index}"
        table.cell(0, 1).text = f"B{index}"
        table.cell(1, 0).text = f"C{index}"
        table.cell(1, 1).text = f"D{index}"

    buffer = BytesIO()
    doc.save(buffer)

    accesses = {"paragraphs": 0, "tables": 0}
    original_paragraphs = DocxDocument.paragraphs
    original_tables = DocxDocument.tables

    def count_paragraphs(self):
        accesses["paragraphs"] += 1
        return original_paragraphs.fget(self)

    def count_tables(self):
        accesses["tables"] += 1
        return original_tables.fget(self)

    monkeypatch.setattr(DocxDocument, "paragraphs", property(count_paragraphs))
    monkeypatch.setattr(DocxDocument, "tables", property(count_tables))

    parser = mf.Parser.docx("python_docx")
    response = parser(buffer.getvalue())

    # Allow a separate metadata read while rejecting a rescan per body block.
    assert accesses["paragraphs"] <= 2
    assert accesses["tables"] <= 2

    text = response.data["text"]
    assert "# Heading 0" in text
    assert text.index("Heading 0") < text.index("| A0 |") < text.index("Heading 1")
    assert response.data["metadata"].num_paragraphs == 30
    assert response.data["metadata"].num_tables == 10
    assert response.data["metadata"].num_images == 1
    assert len(response.data["images"]) == 1


# ---------------------------------------------------------------------------
# PPTX
# ---------------------------------------------------------------------------


def test_pptx_response_structure(pptx_bytes):
    pytest.importorskip("pptx")
    parser = mf.Parser.pptx("python_pptx")
    response = parser(pptx_bytes)

    assert response.response_type == "pptx_parse"
    assert "text" in response.data
    assert "images" in response.data
    assert "metadata" in response.data


def test_pptx_title_extraction(pptx_bytes):
    pytest.importorskip("pptx")
    parser = mf.Parser.pptx("python_pptx")
    response = parser(pptx_bytes)

    assert "Test Slide" in response.data["text"]


def test_pptx_slide_marker(pptx_bytes):
    pytest.importorskip("pptx")
    parser = mf.Parser.pptx("python_pptx")
    response = parser(pptx_bytes)

    assert "Slide number: 1" in response.data["text"]


def test_pptx_metadata(pptx_bytes):
    pytest.importorskip("pptx")
    parser = mf.Parser.pptx("python_pptx")
    response = parser(pptx_bytes)

    assert response.data["metadata"].num_slides == 1


# ---------------------------------------------------------------------------
# XLSX
# ---------------------------------------------------------------------------


def _xlsx_payload(populate):
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    populate(workbook.active)
    buf = BytesIO()
    workbook.save(buf)
    return buf.getvalue()


def _xlsx_sheet(populate):
    openpyxl = pytest.importorskip("openpyxl")
    return openpyxl.load_workbook(
        BytesIO(_xlsx_payload(populate)), data_only=True
    ).active


class _RowAccessBudgetError(RuntimeError):
    pass


@contextmanager
def _worksheet_row_budget(limit):
    pytest.importorskip("openpyxl")
    from openpyxl.worksheet.worksheet import Worksheet

    original = Worksheet.__getitem__
    accesses = []

    def guarded(sheet, key):
        accesses.append(key)
        if len(accesses) > limit:
            raise _RowAccessBudgetError(
                f"worksheet row accesses exceeded the {limit} access budget"
            )
        return original(sheet, key)

    Worksheet.__getitem__ = guarded
    try:
        yield accesses
    finally:
        Worksheet.__getitem__ = original


def test_xlsx_response_structure(xlsx_bytes):
    pytest.importorskip("openpyxl")
    parser = mf.Parser.xlsx("openpyxl")
    response = parser(xlsx_bytes)

    assert response.response_type == "xlsx_parse"
    assert "text" in response.data
    assert "images" in response.data
    assert "metadata" in response.data


def test_xlsx_table_markdown(xlsx_bytes):
    pytest.importorskip("openpyxl")
    parser = mf.Parser.xlsx("openpyxl")
    response = parser(xlsx_bytes)

    assert "|" in response.data["text"]
    assert "Name" in response.data["text"]
    assert "Alice" in response.data["text"]


def test_xlsx_table_html(xlsx_bytes):
    pytest.importorskip("openpyxl")
    parser = mf.Parser.xlsx("openpyxl", table_format="html")
    response = parser(xlsx_bytes)

    assert "<table>" in response.data["text"]


def test_xlsx_metadata(xlsx_bytes):
    pytest.importorskip("openpyxl")
    parser = mf.Parser.xlsx("openpyxl")
    response = parser(xlsx_bytes)

    assert response.data["metadata"].num_sheets == 1


def test_xlsx_find_tables_compact_sheet_with_blank_gap():
    pytest.importorskip("openpyxl")

    def populate(ws):
        ws.append(["Name", "Value"])
        ws.append(["Alice", 42])
        ws.append([None, None])
        ws.append(["Bob", 7])

    parser = mf.Parser.xlsx("openpyxl")
    assert parser._find_tables(_xlsx_sheet(populate)) == [(1, 2), (4, 4)]

    text = parser(_xlsx_payload(populate)).data["text"]
    assert text.count("<!-- Table number:") == 2
    assert "Position: (1, 2)" in text
    assert "Position: (4, 4)" in text
    assert "Alice" in text
    assert "Bob" in text


def test_xlsx_find_tables_ignores_style_only_cells():
    pytest.importorskip("openpyxl")
    from openpyxl.styles import PatternFill

    fill = PatternFill("solid", fgColor="FFFF00")

    def populate(ws):
        ws.append(["Name", "Value"])
        ws.append(["Alice", 42])
        for coordinate in ("A4", "B4", "A7"):
            ws[coordinate].fill = fill

    parser = mf.Parser.xlsx("openpyxl")
    assert parser._find_tables(_xlsx_sheet(populate)) == [(1, 2)]

    text = parser(_xlsx_payload(populate)).data["text"]
    assert text.count("<!-- Table number:") == 1
    assert "Position: (1, 2)" in text


def test_xlsx_find_tables_preserves_false_and_zero():
    pytest.importorskip("openpyxl")

    def populate(ws):
        ws.append(["Flag", "Value"])
        ws.append([False, 0])

    parser = mf.Parser.xlsx("openpyxl")
    assert parser._find_tables(_xlsx_sheet(populate)) == [(1, 2)]

    text = parser(_xlsx_payload(populate)).data["text"]
    assert "False" in text
    assert "0" in text
    assert "Position: (1, 2)" in text


def test_xlsx_find_tables_handles_merged_cells():
    pytest.importorskip("openpyxl")

    def populate(ws):
        ws["A1"] = "merged"
        ws.merge_cells("A1:C2")
        ws["A3"] = "after"

    parser = mf.Parser.xlsx("openpyxl")
    assert parser._find_tables(_xlsx_sheet(populate)) == [(1, 1), (3, 3)]

    text = parser(_xlsx_payload(populate)).data["text"]
    assert "Position: (1, 1)" in text
    assert "Position: (3, 3)" in text
    assert "merged" in text
    assert "after" in text


def test_xlsx_find_tables_all_blank_sheet():
    pytest.importorskip("openpyxl")
    parser = mf.Parser.xlsx("openpyxl")
    assert parser._find_tables(_xlsx_sheet(lambda ws: None)) == []

    text = parser(_xlsx_payload(lambda ws: None)).data["text"]
    assert "<!-- Table number:" not in text
    assert "|" not in text


def test_xlsx_find_tables_far_row_and_column():
    pytest.importorskip("openpyxl")

    def populate(ws):
        ws.cell(row=1_048_576, column=16_384, value="far")

    parser = mf.Parser.xlsx("openpyxl")
    assert parser._find_tables(_xlsx_sheet(populate)) == [(1_048_576, 1_048_576)]


def test_xlsx_find_tables_does_not_materialize_sparse_rows():
    pytest.importorskip("openpyxl")

    def populate(ws):
        ws.cell(row=1, column=1, value="first")
        ws.cell(row=1_048_576, column=1, value="sparse")

    parser = mf.Parser.xlsx("openpyxl")

    sparse_sheet = _xlsx_sheet(populate)
    before = set(sparse_sheet._cells)
    with _worksheet_row_budget(8):
        assert parser._find_tables(sparse_sheet) == [
            (1, 1),
            (1_048_576, 1_048_576),
        ]
    assert set(sparse_sheet._cells) == before


def test_xlsx_public_parse_sparse_final_row_within_row_budget():
    pytest.importorskip("openpyxl")

    def populate(ws):
        ws.cell(row=1_048_576, column=1, value="sparse")

    payload = _xlsx_payload(populate)
    parser = mf.Parser.xlsx("openpyxl")

    with _worksheet_row_budget(4) as accesses:
        response = parser(payload)

    assert len(accesses) <= 4
    text = response.data["text"]
    assert "Position: (1048576, 1048576)" in text
    assert "sparse" in text
    assert response.data["metadata"].num_sheets == 1
    assert response.data["images"] == {}
