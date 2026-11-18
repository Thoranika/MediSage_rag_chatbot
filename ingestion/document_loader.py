"""Unified PDF, DOCX, and plain-text extraction with page/section metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pymupdf as fitz  # PyMuPDF
from docx import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from core.logging_utils import get_logger
from ingestion.text_cleaner import (
    detect_primary_heading,
    normalize_text,
    strip_repeated_headers_footers,
)
from utils.helpers import sanitize_error_message

logger = get_logger("ingestion.loader")


class DocumentParseError(Exception):
    """Raised when a document is corrupt, encrypted, unsupported, or empty."""


class ScannedDocumentError(DocumentParseError):
    """Raised when a PDF appears to be scanned/image-only without OCR text."""


@dataclass
class ExtractedPage:
    """Extracted text segment with traceable filename and page/section metadata."""

    filename: str
    page_number: int
    section_title: str
    text: str


@dataclass
class LoadedDocument:
    """Complete parsed representation of an uploaded medical reference document."""

    filename: str
    file_type: str
    page_count: int
    pages: list[ExtractedPage] = field(default_factory=list)


def _format_table_rows(rows: Iterable[list[str]]) -> str:
    """Format extracted table cells into clean pipe-delimited markdown rows."""
    cleaned_rows: list[list[str]] = []
    for row in rows:
        cells = [re.sub(r"\s+", " ", (cell or "").strip()) for cell in row]
        if any(cells):
            cleaned_rows.append(cells)

    if not cleaned_rows:
        return ""

    col_count = max(len(r) for r in cleaned_rows)
    lines: list[str] = []
    for idx, row in enumerate(cleaned_rows):
        padded = row + [""] * (col_count - len(row))
        lines.append("| " + " | ".join(padded) + " |")
        if idx == 0 and len(cleaned_rows) > 1:
            lines.append("| " + " | ".join(["---"] * col_count) + " |")
    return "\n".join(lines)


def _extract_pdf_page_text_and_tables(page: fitz.Page) -> str:
    """Extract page text plus structured tables if present on a PyMuPDF page."""
    raw_text = page.get_text("text", sort=True) or ""
    table_blocks: list[str] = []
    try:
        finder = page.find_tables()
        for tbl in getattr(finder, "tables", []) or []:
            rows = tbl.extract()
            if rows:
                formatted = _format_table_rows(rows)
                if formatted:
                    table_blocks.append(formatted)
    except Exception:
        # Table detection is best-effort; plain sorted text still preserves content
        pass

    combined = raw_text.strip()
    if table_blocks:
        # Append structured markdown table representation if not already pipe-formatted
        extra_tables = "\n\n".join(table_blocks)
        if "|" not in combined:
            combined = f"{combined}\n\n{extra_tables}".strip()
    return normalize_text(combined)


def load_pdf(file_path: Path, display_filename: str | None = None) -> LoadedDocument:
    """Extract PDF text page by page while detecting encrypted, corrupt, and scanned PDFs."""
    filename = display_filename or file_path.name
    try:
        doc = fitz.open(str(file_path))
    except Exception as exc:
        logger.warning("Failed to open PDF %s: %s", filename, sanitize_error_message(exc))
        raise DocumentParseError(
            "Unable to open PDF file. The file may be corrupted or malformed."
        ) from exc

    try:
        if doc.is_encrypted or getattr(doc, "needs_pass", False):
            raise DocumentParseError(
                "Encrypted or password-protected PDF files are not supported."
            )

        total_pages = len(doc)
        if total_pages == 0:
            raise DocumentParseError("PDF contains 0 pages.")

        raw_pages: list[str] = []
        image_or_drawing_pages = 0

        for idx in range(total_pages):
            page = doc[idx]
            page_text = _extract_pdf_page_text_and_tables(page)
            raw_pages.append(page_text)
            try:
                has_images = len(page.get_images(full=False)) > 0
                has_drawings = len(page.get_drawings()) > 0
                if has_images or has_drawings:
                    image_or_drawing_pages += 1
            except Exception:
                pass

        alnum_count = sum(1 for ch in "".join(raw_pages) if ch.isalnum())
        if alnum_count < 30:
            if image_or_drawing_pages > 0:
                raise ScannedDocumentError(
                    "Scanned or image-only PDF detected. OCR is not enabled; "
                    "please upload a text-searchable PDF or DOCX file."
                )
            raise DocumentParseError(
                "PDF contains no extractable text content."
            )

        cleaned_pages = strip_repeated_headers_footers(raw_pages)
        pages: list[ExtractedPage] = []
        last_section = "Overview"

        for idx, content in enumerate(cleaned_pages, start=1):
            if not content.strip():
                continue
            section = detect_primary_heading(content, fallback=last_section)
            last_section = section
            pages.append(
                ExtractedPage(
                    filename=filename,
                    page_number=idx,
                    section_title=section,
                    text=content,
                )
            )

        if not pages:
            raise DocumentParseError("No readable text pages were extracted from PDF.")

        return LoadedDocument(
            filename=filename,
            file_type="pdf",
            page_count=total_pages,
            pages=pages,
        )
    finally:
        doc.close()


def _iter_docx_blocks(doc: DocxDocument):
    """Yield Paragraph and Table objects in exact document body order."""
    body = doc.element.body
    for child in body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, doc)
        elif isinstance(child, CT_Tbl):
            yield Table(child, doc)


def _has_explicit_page_break(paragraph: Paragraph) -> bool:
    """Detect explicit Word page breaks inside a paragraph XML."""
    xml = paragraph._p.xml
    return 'w:type="page"' in xml or "w:lastRenderedPageBreak" in xml


def load_docx(file_path: Path, display_filename: str | None = None) -> LoadedDocument:
    """Extract DOCX paragraphs and tables in reading order with tracked section/page metadata."""
    filename = display_filename or file_path.name
    try:
        doc = DocxDocument(str(file_path))
    except Exception as exc:
        logger.warning("Failed to open DOCX %s: %s", filename, sanitize_error_message(exc))
        raise DocumentParseError(
            "Unable to open DOCX file. The archive may be corrupted or invalid."
        ) from exc

    blocks: list[tuple[str, bool, str]] = []  # (text, is_heading, heading_title)
    current_heading = "Overview"

    for item in _iter_docx_blocks(doc):
        if isinstance(item, Paragraph):
            text = normalize_text(item.text)
            has_break = _has_explicit_page_break(item)
            style_name = (item.style.name if item.style else "") or ""
            is_heading = style_name.lower().startswith("heading") or style_name.lower() == "title"
            if text:
                if is_heading:
                    current_heading = text[:120]
                    blocks.append((f"## {text}", True, current_heading))
                else:
                    blocks.append((text, False, current_heading))
            if has_break:
                blocks.append(("__PAGE_BREAK__", False, current_heading))
        elif isinstance(item, Table):
            rows = [[cell.text for cell in row.cells] for row in item.rows]
            formatted_table = _format_table_rows(rows)
            if formatted_table:
                blocks.append((formatted_table, False, current_heading))

    # Group blocks into logical pages (using explicit page breaks or ~2200 char budget)
    pages: list[ExtractedPage] = []
    current_parts: list[str] = []
    current_chars = 0
    page_num = 1
    page_section = "Overview"

    def _flush_page() -> None:
        nonlocal current_parts, current_chars, page_num
        joined = normalize_text("\n\n".join(current_parts))
        if joined:
            pages.append(
                ExtractedPage(
                    filename=filename,
                    page_number=page_num,
                    section_title=page_section,
                    text=joined,
                )
            )
            page_num += 1
        current_parts = []
        current_chars = 0

    for text, is_heading, heading_title in blocks:
        if text == "__PAGE_BREAK__":
            if current_parts:
                _flush_page()
                page_section = heading_title
            continue

        if not current_parts:
            page_section = heading_title
        elif is_heading and current_chars >= 1200:
            _flush_page()
            page_section = heading_title

        current_parts.append(text)
        current_chars += len(text)
        if current_chars >= 2400 and not is_heading:
            _flush_page()
            page_section = heading_title

    if current_parts:
        _flush_page()

    if not pages:
        raise DocumentParseError("DOCX document contains no extractable paragraphs or tables.")

    return LoadedDocument(
        filename=filename,
        file_type="docx",
        page_count=len(pages),
        pages=pages,
    )


def load_text_document(
    file_path: Path, display_filename: str | None = None
) -> LoadedDocument:
    """Extract TXT or Markdown files into logical pages with heading tracking."""
    filename = display_filename or file_path.name
    raw_bytes = file_path.read_bytes()
    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raw_text = raw_bytes.decode("latin-1")

    cleaned = normalize_text(raw_text)
    if not cleaned:
        raise DocumentParseError("Text file contains no extractable content.")

    paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
    pages: list[ExtractedPage] = []
    buf: list[str] = []
    buf_chars = 0
    page_num = 1
    current_section = detect_primary_heading(cleaned, fallback="Document Text")

    for para in paragraphs:
        if para.startswith("#"):
            current_section = para.lstrip("#").strip()[:120] or current_section
        buf.append(para)
        buf_chars += len(para)
        if buf_chars >= 2400:
            pages.append(
                ExtractedPage(
                    filename=filename,
                    page_number=page_num,
                    section_title=current_section,
                    text="\n\n".join(buf),
                )
            )
            page_num += 1
            buf = []
            buf_chars = 0

    if buf:
        pages.append(
            ExtractedPage(
                filename=filename,
                page_number=page_num,
                section_title=current_section,
                text="\n\n".join(buf),
            )
        )

    ext = file_path.suffix.lower().lstrip(".") or "txt"
    return LoadedDocument(
        filename=filename,
        file_type=ext,
        page_count=len(pages),
        pages=pages,
    )


def load_document(
    file_path: Path | str, display_filename: str | None = None
) -> LoadedDocument:
    """Dispatch document extraction based on extension."""
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise DocumentParseError("Uploaded file could not be located for extraction.")

    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path, display_filename=display_filename)
    if ext == ".docx":
        return load_docx(path, display_filename=display_filename)
    if ext in {".txt", ".md"}:
        return load_text_document(path, display_filename=display_filename)

    raise DocumentParseError(f"Unsupported file format: {ext}")
