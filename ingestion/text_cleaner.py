"""Text cleaning and normalization preserving headings, tables, and paragraph boundaries."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE_RE = re.compile(r"[ \t\u00a0]{2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_HYPHEN_LINEBREAK_RE = re.compile(r"([A-Za-z])-\n([a-z])")
_PAGE_NUMBER_LINE_RE = re.compile(
    r"^(?:page\s*\d+(?:\s*(?:of|/)\s*\d+)?|\d+\s*/\s*\d+|-\s*\d+\s*-)$",
    re.IGNORECASE,
)


def normalize_text(text: str, *, preserve_tables: bool = True) -> str:
    """Clean extracted text while keeping headings, paragraphs, and table rows intact."""
    if not text:
        return ""

    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _CONTROL_CHAR_RE.sub("", cleaned)

    # Rejoin soft hyphenations across line wraps inside prose
    cleaned = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", cleaned)

    lines_out: list[str] = []
    for raw_line in cleaned.split("\n"):
        line = raw_line.strip()
        if not line:
            lines_out.append("")
            continue
        if _PAGE_NUMBER_LINE_RE.match(line):
            continue
        if preserve_tables and line.startswith("|") and line.endswith("|"):
            # Keep pipe-delimited table row spacing clean
            cells = [re.sub(r"\s+", " ", c.strip()) for c in line.strip("|").split("|")]
            lines_out.append("| " + " | ".join(cells) + " |")
        else:
            lines_out.append(_MULTI_SPACE_RE.sub(" ", line))

    joined = "\n".join(lines_out)
    joined = _MULTI_NEWLINE_RE.sub("\n\n", joined)
    return joined.strip()


def strip_repeated_headers_footers(pages_text: list[str]) -> list[str]:
    """Remove identical short header/footer lines repeated across >=3 pages."""
    if len(pages_text) < 3:
        return pages_text

    first_lines: Counter[str] = Counter()
    last_lines: Counter[str] = Counter()
    split_pages: list[list[str]] = []

    for page in pages_text:
        lines = [ln.strip() for ln in page.splitlines() if ln.strip()]
        split_pages.append(lines)
        if lines:
            if len(lines[0]) <= 100:
                first_lines[lines[0]] += 1
            if len(lines) > 1 and len(lines[-1]) <= 100:
                last_lines[lines[-1]] += 1

    threshold = max(3, len(pages_text) // 2 + 1)
    repeated_headers = {line for line, count in first_lines.items() if count >= threshold}
    repeated_footers = {line for line, count in last_lines.items() if count >= threshold}

    if not repeated_headers and not repeated_footers:
        return pages_text

    cleaned_pages: list[str] = []
    for page in pages_text:
        kept: list[str] = []
        raw_lines = page.splitlines()
        non_empty_indices = [idx for idx, ln in enumerate(raw_lines) if ln.strip()]
        first_idx = non_empty_indices[0] if non_empty_indices else -1
        last_idx = non_empty_indices[-1] if non_empty_indices else -1

        for idx, raw_line in enumerate(raw_lines):
            stripped = raw_line.strip()
            if idx == first_idx and stripped in repeated_headers:
                continue
            if idx == last_idx and stripped in repeated_footers:
                continue
            kept.append(raw_line)
        cleaned_pages.append(normalize_text("\n".join(kept)))

    return cleaned_pages


def detect_primary_heading(text: str, fallback: str = "General Section") -> str:
    """Identify the first likely section heading from a page or block of text."""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("|"):
            continue
        if line.startswith("#"):
            return line.lstrip("#").strip()[:120]
        if 4 <= len(line) <= 90 and not line.endswith("."):
            # Numbered section or title-case/uppercase heading
            if re.match(r"^(?:\d+(?:\.\d+)*\.?\s+)?[A-Z][A-Za-z0-9\s,:/()\-–]+$", line):
                words = line.split()
                if 1 <= len(words) <= 12:
                    return line[:120]
    return fallback
