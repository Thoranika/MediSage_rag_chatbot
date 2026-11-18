"""Heading- and paragraph-aware chunking with configurable token size and overlap."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ingestion.document_loader import ExtractedPage, LoadedDocument
from ingestion.text_cleaner import detect_primary_heading
from utils.helpers import estimate_tokens, utc_now_iso

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(|])")


@dataclass
class TextChunk:
    """Chunk with complete source, page, section, and ingestion metadata."""

    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    section_title: str
    chunk_index: int
    text: str
    token_count: int
    ingestion_timestamp: str
    file_hash: str

    def to_metadata(self) -> dict[str, str | int]:
        """Serialize metadata into ChromaDB-compatible primitive fields."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "page_number": int(self.page_number),
            "section_title": self.section_title,
            "chunk_index": int(self.chunk_index),
            "token_count": int(self.token_count),
            "ingestion_timestamp": self.ingestion_timestamp,
            "file_hash": self.file_hash,
        }


class DocumentChunker:
    """Chunk documents preferring heading, table, and paragraph boundaries."""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100) -> None:
        if chunk_size < 100:
            raise ValueError("chunk_size must be at least 100 tokens.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size.")
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)

    @staticmethod
    def _is_heading(block: str) -> bool:
        stripped = block.strip()
        if not stripped or "|" in stripped:
            return False
        if stripped.startswith("#"):
            return True
        words = stripped.split()
        return (
            1 <= len(words) <= 12
            and len(stripped) <= 95
            and not stripped.endswith((".", ";", ","))
            and stripped[0].isalnum()
        )

    def _split_oversized_block(self, block: str) -> list[str]:
        """Split a block exceeding chunk_size on sentence boundaries."""
        if estimate_tokens(block) <= self.chunk_size:
            return [block]

        # For oversized pipe tables, split by row while preserving header row
        lines = block.splitlines()
        if len(lines) >= 3 and lines[0].startswith("|") and lines[1].startswith("|"):
            header = "\n".join(lines[:2])
            parts: list[str] = []
            current_rows = [header]
            for row in lines[2:]:
                candidate = "\n".join(current_rows + [row])
                if estimate_tokens(candidate) > self.chunk_size and len(current_rows) > 1:
                    parts.append("\n".join(current_rows))
                    current_rows = [header, row]
                else:
                    current_rows.append(row)
            if current_rows:
                parts.append("\n".join(current_rows))
            return parts

        sentences = _SENTENCE_SPLIT_RE.split(block)
        if len(sentences) <= 1:
            # Fallback word split
            words = block.split()
            max_words = max(40, int(self.chunk_size * 0.72))
            step = max(20, max_words - int(self.chunk_overlap * 0.72))
            slices = []
            for start in range(0, len(words), step):
                chunk_words = words[start : start + max_words]
                if chunk_words:
                    slices.append(" ".join(chunk_words))
                if start + max_words >= len(words):
                    break
            return slices

        out: list[str] = []
        buf: list[str] = []
        for sent in sentences:
            candidate = " ".join(buf + [sent]).strip()
            if buf and estimate_tokens(candidate) > self.chunk_size:
                out.append(" ".join(buf).strip())
                buf = [sent]
            else:
                buf.append(sent)
        if buf:
            out.append(" ".join(buf).strip())
        return out

    def _tail_overlap_text(self, text: str) -> str:
        """Extract the trailing ~chunk_overlap tokens along paragraph/sentence boundaries."""
        if self.chunk_overlap <= 0 or not text:
            return ""
        words = text.split()
        target_words = max(1, int(self.chunk_overlap * 0.75))
        if len(words) <= target_words:
            return text
        tail = " ".join(words[-target_words:])
        # Snap to sentence start if nearby
        match = re.search(r"[.!?]\s+([A-Z0-9].*)$", tail)
        if match and estimate_tokens(match.group(1)) >= max(20, self.chunk_overlap // 3):
            return match.group(1).strip()
        return tail.strip()

    def chunk_page(
        self,
        *,
        page: ExtractedPage,
        document_id: str,
        file_hash: str,
        start_chunk_index: int = 0,
        ingestion_timestamp: str | None = None,
    ) -> list[TextChunk]:
        """Split a single extracted page into metadata-rich overlapping chunks."""
        ts = ingestion_timestamp or utc_now_iso()
        raw_paragraphs = [p.strip() for p in page.text.split("\n\n") if p.strip()]
        blocks: list[str] = []
        for para in raw_paragraphs:
            blocks.extend(self._split_oversized_block(para))

        if not blocks:
            return []

        chunks: list[TextChunk] = []
        current_blocks: list[str] = []
        current_section = page.section_title or detect_primary_heading(page.text)
        idx = start_chunk_index

        def _emit_chunk(chunk_blocks: list[str], section_name: str) -> None:
            nonlocal idx
            body = "\n\n".join(chunk_blocks).strip()
            if not body:
                return
            chunks.append(
                TextChunk(
                    chunk_id=f"{document_id}_p{page.page_number}_c{idx}",
                    document_id=document_id,
                    filename=page.filename,
                    page_number=page.page_number,
                    section_title=section_name,
                    chunk_index=idx,
                    text=body,
                    token_count=estimate_tokens(body),
                    ingestion_timestamp=ts,
                    file_hash=file_hash,
                )
            )
            idx += 1

        for block in blocks:
            if self._is_heading(block):
                heading_clean = block.lstrip("#").strip()[:120]
                # If accumulated text is already substantial, flush before starting new section
                if current_blocks and estimate_tokens("\n\n".join(current_blocks)) >= int(
                    self.chunk_size * 0.55
                ):
                    _emit_chunk(current_blocks, current_section)
                    overlap = self._tail_overlap_text("\n\n".join(current_blocks))
                    current_blocks = [overlap] if overlap else []
                current_section = heading_clean or current_section

            candidate = "\n\n".join(current_blocks + [block]).strip()
            if current_blocks and estimate_tokens(candidate) > self.chunk_size:
                prev_text = "\n\n".join(current_blocks).strip()
                _emit_chunk(current_blocks, current_section)
                overlap = self._tail_overlap_text(prev_text)
                if overlap and estimate_tokens(overlap + "\n\n" + block) <= self.chunk_size:
                    current_blocks = [overlap, block]
                else:
                    current_blocks = [block]
            else:
                current_blocks.append(block)

        if current_blocks:
            _emit_chunk(current_blocks, current_section)

        return chunks

    def chunk_document(
        self,
        loaded_doc: LoadedDocument,
        *,
        document_id: str,
        file_hash: str,
        ingestion_timestamp: str | None = None,
    ) -> list[TextChunk]:
        """Chunk all pages of a LoadedDocument preserving page boundaries."""
        ts = ingestion_timestamp or utc_now_iso()
        all_chunks: list[TextChunk] = []
        next_idx = 0
        for page in loaded_doc.pages:
            page_chunks = self.chunk_page(
                page=page,
                document_id=document_id,
                file_hash=file_hash,
                start_chunk_index=next_idx,
                ingestion_timestamp=ts,
            )
            all_chunks.extend(page_chunks)
            next_idx += len(page_chunks)
        return all_chunks
