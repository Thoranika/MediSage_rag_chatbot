"""Citation parsing and validation against actual retrieved chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from rag.retriever import RetrievedChunk

_CITATION_BLOCK_RE = re.compile(r"\[([^\]]*?\bSource:[^\]]+)\]", re.IGNORECASE)
_CITATION_ITEM_RE = re.compile(
    r"(?:Source:\s*)?([^,;]+?)\s*,\s*(?:page|p\.?)\s*(\d+)",
    re.IGNORECASE,
)

_ABSTENTION_MARKERS = (
    "could not find sufficient supporting information",
    "lacks sufficient information",
    "do not contain enough evidence",
    "insufficient information in your uploaded",
    "not mentioned in the provided",
)


def is_abstention_response(text: str) -> bool:
    """Return True if the model explicitly abstained due to insufficient document evidence."""
    lowered = (text or "").lower()
    return any(marker in lowered for marker in _ABSTENTION_MARKERS)


@dataclass
class VerifiedCitation:
    """Verified citation referencing an actual retrieved passage."""

    document_id: str
    filename: str
    page_number: int
    section_title: str
    similarity_score: float
    excerpt: str
    verified: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "page_number": int(self.page_number),
            "section_title": self.section_title,
            "similarity_score": round(float(self.similarity_score), 4),
            "excerpt": self.excerpt,
            "verified": self.verified,
        }


@dataclass
class CitationValidationReport:
    """Outcome of verifying model-emitted citations against retrieved chunks."""

    cleaned_answer: str
    citations: list[VerifiedCitation] = field(default_factory=list)
    valid_citation_count: int = 0
    invalid_citations_removed: list[str] = field(default_factory=list)
    all_citations_valid: bool = True


def _make_excerpt(text: str, max_len: int = 360) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_len:
        return compact
    return compact[:max_len].rsplit(" ", 1)[0] + "..."


def validate_and_extract_citations(
    answer_text: str, retrieved_chunks: list[RetrievedChunk]
) -> CitationValidationReport:
    """
    Validate every `[Source: filename, page N]` tag in answer_text against retrieved_chunks.

    Unverified/hallucinated citation tags are stripped and recorded in `invalid_citations_removed`.
    """
    if not answer_text:
        return CitationValidationReport(cleaned_answer="", all_citations_valid=True)

    # Index retrieved chunks by lowercase filename and (lowercase filename, page_number)
    chunk_by_pair: dict[tuple[str, int], RetrievedChunk] = {}
    for ch in retrieved_chunks:
        key = (ch.filename.strip().lower(), int(ch.page_number))
        if key not in chunk_by_pair or ch.similarity > chunk_by_pair[key].similarity:
            chunk_by_pair[key] = ch

    verified_map: dict[tuple[str, int], VerifiedCitation] = {}
    invalid_tags: list[str] = []
    valid_count = 0

    def _replace_block(match: re.Match[str]) -> str:
        nonlocal valid_count
        raw_full = match.group(0)
        inner = match.group(1)
        parts = [p.strip() for p in inner.split(";") if p.strip()]
        normalized_tags: list[str] = []

        for part in parts:
            item_match = _CITATION_ITEM_RE.search(part)
            if not item_match:
                invalid_tags.append(f"[{part}]")
                continue

            fname = item_match.group(1).strip()
            try:
                page_num = int(item_match.group(2))
            except ValueError:
                invalid_tags.append(f"[{part}]")
                continue

            matched_chunk = chunk_by_pair.get((fname.lower(), page_num))
            if matched_chunk is None:
                invalid_tags.append(f"[Source: {fname}, page {page_num}]")
                continue

            valid_count += 1
            pair_key = (matched_chunk.filename.lower(), matched_chunk.page_number)
            if pair_key not in verified_map:
                verified_map[pair_key] = VerifiedCitation(
                    document_id=matched_chunk.document_id,
                    filename=matched_chunk.filename,
                    page_number=matched_chunk.page_number,
                    section_title=matched_chunk.section_title,
                    similarity_score=matched_chunk.similarity,
                    excerpt=_make_excerpt(matched_chunk.text),
                    verified=True,
                )
            normalized_tags.append(
                f"[Source: {matched_chunk.filename}, page {matched_chunk.page_number}]"
            )

        if not normalized_tags and not parts:
            invalid_tags.append(raw_full)
            return ""
        return " ".join(normalized_tags)

    cleaned = _CITATION_BLOCK_RE.sub(_replace_block, answer_text)
    cleaned = re.sub(r" {2,}", " ", cleaned).strip()

    # If the model omitted inline citations despite having retrieved evidence, attach top supporting sources
    # unless the model explicitly abstained from answering due to insufficient evidence.
    citations = list(verified_map.values())
    if not citations and retrieved_chunks and not is_abstention_response(cleaned):
        # Expose retrieved supporting evidence for transparency in source drawer
        for ch in retrieved_chunks[:2]:
            pair_key = (ch.filename.lower(), ch.page_number)
            if pair_key not in verified_map:
                verified_map[pair_key] = VerifiedCitation(
                    document_id=ch.document_id,
                    filename=ch.filename,
                    page_number=ch.page_number,
                    section_title=ch.section_title,
                    similarity_score=ch.similarity,
                    excerpt=_make_excerpt(ch.text),
                    verified=True,
                )
        citations = list(verified_map.values())

    return CitationValidationReport(
        cleaned_answer=cleaned,
        citations=citations,
        valid_citation_count=valid_count,
        invalid_citations_removed=invalid_tags,
        all_citations_valid=len(invalid_tags) == 0,
    )
