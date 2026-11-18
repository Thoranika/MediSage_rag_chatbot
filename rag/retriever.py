"""Semantic retrieval pipeline with threshold filtering, deduplication, diversity, and caching."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from core.logging_utils import get_logger
from ingestion.embeddings import EmbeddingService
from utils.cache import get_shared_cache
from vectorstore.chroma_manager import ChromaManager

logger = get_logger("rag.retriever")


@dataclass
class RetrievedChunk:
    """Single retrieved passage with provenance and similarity metrics."""

    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    section_title: str
    text: str
    similarity: float
    distance: float
    chunk_index: int = 0


@dataclass
class RetrievalResult:
    """Structured outcome of semantic retrieval."""

    query: str
    status: str  # "ok", "low_confidence", "no_evidence"
    chunks: list[RetrievedChunk] = field(default_factory=list)
    latency_ms: float = 0.0
    from_cache: bool = False
    total_candidates: int = 0


def _token_shingles(text: str) -> set[str]:
    words = [w.lower().strip(".,;:!?()[]{}\"'") for w in text.split()]
    words = [w for w in words if w]
    if len(words) < 3:
        return set(words)
    return {" ".join(words[i : i + 3]) for i in range(len(words) - 2)}


def _overlap_ratio(a: str, b: str) -> float:
    sa = _token_shingles(a)
    sb = _token_shingles(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    denom = min(len(sa), len(sb))
    return inter / denom if denom else 0.0


class SemanticRetriever:
    """Retrieves, filters, deduplicates, and ranks relevant source chunks from ChromaDB."""

    def __init__(
        self,
        *,
        chroma: ChromaManager,
        embeddings: EmbeddingService,
    ) -> None:
        self.chroma = chroma
        self.embeddings = embeddings
        self.cache = get_shared_cache()

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 4,
        similarity_threshold: float = 0.32,
        metadata_filters: dict[str, Any] | None = None,
        deduplicate: bool = True,
        overlap_cutoff: float = 0.78,
        enforce_doc_diversity: bool = True,
        use_cache: bool = True,
    ) -> RetrievalResult:
        """Embed query and retrieve ranked, deduplicated, threshold-filtered passages."""
        clean_query = (query or "").strip()
        if not clean_query:
            return RetrievalResult(query="", status="no_evidence", chunks=[])

        if use_cache:
            cached = self.cache.get_retrieval(
                clean_query,
                top_k=top_k,
                threshold=similarity_threshold,
                filters=metadata_filters,
                collection_name=self.chroma.collection_name,
            )
            if cached is not None:
                return RetrievalResult(
                    query=clean_query,
                    status=cached.status,
                    chunks=list(cached.chunks),
                    latency_ms=0.0,
                    from_cache=True,
                    total_candidates=cached.total_candidates,
                )

        t0 = time.perf_counter()
        if self.chroma.count() == 0:
            return RetrievalResult(
                query=clean_query,
                status="no_evidence",
                chunks=[],
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        query_vec = self.embeddings.embed_query(clean_query)
        candidate_k = max(top_k * 3, 8)
        raw_hits = self.chroma.search(
            query_vec,
            top_k=candidate_k,
            where=metadata_filters or None,
        )

        # Threshold filter
        above_threshold = [
            hit for hit in raw_hits if float(hit["similarity"]) >= float(similarity_threshold)
        ]

        # Deduplicate overlapping passages and encourage multi-doc diversity when unfiltered
        selected: list[RetrievedChunk] = []
        per_doc_counts: dict[str, int] = {}
        max_per_doc = max(2, int(math.ceil(top_k * 0.65))) if (
            enforce_doc_diversity and not metadata_filters
        ) else top_k

        overflow_pool: list[RetrievedChunk] = []

        for hit in above_threshold:
            candidate = RetrievedChunk(
                chunk_id=hit["chunk_id"],
                document_id=hit["document_id"],
                filename=hit["filename"],
                page_number=int(hit["page_number"]),
                section_title=hit.get("section_title", ""),
                text=hit["text"],
                similarity=float(hit["similarity"]),
                distance=float(hit["distance"]),
                chunk_index=int(hit.get("chunk_index", 0)),
            )

            if deduplicate and any(
                _overlap_ratio(candidate.text, prev.text) >= overlap_cutoff
                for prev in selected
            ):
                continue

            doc_key = candidate.document_id or candidate.filename
            if per_doc_counts.get(doc_key, 0) >= max_per_doc:
                overflow_pool.append(candidate)
                continue

            selected.append(candidate)
            per_doc_counts[doc_key] = per_doc_counts.get(doc_key, 0) + 1
            if len(selected) >= top_k:
                break

        # Fill remaining slots from overflow if corpus only had 1 matching doc
        if len(selected) < top_k and overflow_pool:
            for extra in overflow_pool:
                if deduplicate and any(
                    _overlap_ratio(extra.text, prev.text) >= overlap_cutoff
                    for prev in selected
                ):
                    continue
                selected.append(extra)
                if len(selected) >= top_k:
                    break

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if not selected:
            status = "no_evidence"
        elif selected[0].similarity < max(similarity_threshold + 0.08, 0.34):
            status = "low_confidence"
        else:
            status = "ok"

        result = RetrievalResult(
            query=clean_query,
            status=status,
            chunks=selected,
            latency_ms=round(elapsed_ms, 2),
            from_cache=False,
            total_candidates=len(raw_hits),
        )
        if use_cache:
            self.cache.set_retrieval(
                clean_query,
                top_k=top_k,
                threshold=similarity_threshold,
                filters=metadata_filters,
                collection_name=self.chroma.collection_name,
                value=result,
            )
        return result
