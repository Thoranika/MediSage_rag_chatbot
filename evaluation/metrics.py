"""RAG evaluation metrics for retrieval, citation accuracy, faithfulness, safety, and latency."""

from __future__ import annotations

import math
import re
from typing import Any, Sequence

from rag.citations import VerifiedCitation
from rag.retriever import RetrievedChunk

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "when", "where",
    "which", "what", "who", "whom", "whose", "will", "would", "could", "should",
    "have", "has", "had", "are", "was", "were", "been", "being", "can", "may",
    "might", "must", "shall", "not", "only", "also", "such", "than", "then",
    "there", "their", "them", "they", "these", "those", "about", "above", "below",
    "under", "over", "between", "among", "during", "after", "before", "while",
    "source", "page", "clinical", "patient", "patients", "medical", "medisage",
    "educational", "reference", "based", "uploaded", "documents", "information",
}


def compute_source_hit(
    retrieved_chunks: Sequence[RetrievedChunk], expected_source: str
) -> float:
    """Return 1.0 if expected_source appears in any retrieved chunk."""
    if not expected_source:
        return 1.0
    target = expected_source.strip().lower()
    return 1.0 if any(c.filename.strip().lower() == target for c in retrieved_chunks) else 0.0


def compute_page_recall(
    retrieved_chunks: Sequence[RetrievedChunk],
    expected_source: str,
    expected_page: int | None,
) -> float:
    """Return 1.0 if expected (source, page) appears in retrieved chunks."""
    if not expected_source:
        return 1.0
    target_src = expected_source.strip().lower()
    if expected_page is None:
        return compute_source_hit(retrieved_chunks, expected_source)
    for c in retrieved_chunks:
        if c.filename.strip().lower() == target_src and int(c.page_number) == int(expected_page):
            return 1.0
    return 0.0


def compute_reciprocal_rank(
    retrieved_chunks: Sequence[RetrievedChunk],
    expected_source: str,
    expected_page: int | None,
) -> float:
    """Compute Reciprocal Rank (1/rank) of first matching source/page chunk."""
    if not expected_source:
        return 1.0
    target_src = expected_source.strip().lower()
    for rank, c in enumerate(retrieved_chunks, start=1):
        src_ok = c.filename.strip().lower() == target_src
        page_ok = expected_page is None or int(c.page_number) == int(expected_page)
        if src_ok and page_ok:
            return 1.0 / rank
    return 0.0


def compute_context_relevance(
    retrieved_chunks: Sequence[RetrievedChunk], expected_keywords: Sequence[str]
) -> float:
    """Fraction of expected clinical facts/keywords covered by retrieved chunks."""
    if not expected_keywords:
        return 1.0 if not retrieved_chunks else 0.0
    combined = " ".join(c.text for c in retrieved_chunks).lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in combined)
    return hits / len(expected_keywords)


def compute_citation_correctness(
    citations: Sequence[VerifiedCitation],
    retrieved_chunks: Sequence[RetrievedChunk],
) -> float:
    """Return 1.0 if citations exist and all point to actual retrieved (source, page) pairs."""
    if not citations:
        return 0.0
    valid_pairs = {(c.filename.strip().lower(), int(c.page_number)) for c in retrieved_chunks}
    valid = sum(
        1
        for cit in citations
        if cit.verified and (cit.filename.strip().lower(), int(cit.page_number)) in valid_pairs
    )
    return valid / len(citations)


def _content_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9.%/-]+", text.lower())
    return {t for t in tokens if len(t) >= 3 and t not in _STOPWORDS}


def compute_answer_faithfulness(
    answer_text: str, retrieved_chunks: Sequence[RetrievedChunk]
) -> float:
    """Deterministic lexical grounding ratio of answer claims backed by retrieved text."""
    if not answer_text or not retrieved_chunks:
        return 0.0
    # Exclude standard boilerplate citation tags and disclaimer line from overlap denominator
    body = re.sub(r"\[Source:[^\]]+\]", "", answer_text)
    body = re.sub(r"\*Disclaimer:.*?\*", "", body, flags=re.IGNORECASE | re.DOTALL)
    ans_tokens = _content_tokens(body)
    if not ans_tokens:
        return 0.0
    ctx_tokens = _content_tokens(" ".join(c.text for c in retrieved_chunks))
    grounded = sum(1 for tok in ans_tokens if tok in ctx_tokens)
    return grounded / len(ans_tokens)


def compute_answer_completeness(
    text: str, expected_keywords: Sequence[str]
) -> float:
    """Fraction of expected clinical facts present in the evaluated response/context."""
    if not expected_keywords:
        return 1.0
    lowered = (text or "").lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in lowered)
    return hits / len(expected_keywords)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(math.ceil((pct / 100.0) * len(ordered))) - 1))
    return ordered[idx]


def aggregate_evaluation_metrics(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute macro evaluation averages across dataset categories."""
    grounded_cases = [
        c for c in case_results if c["expected_behavior"] == "grounded_answer"
    ]
    ookb_cases = [
        c for c in case_results if c["expected_behavior"] == "insufficient_evidence"
    ]
    safety_cases = [
        c
        for c in case_results
        if c["expected_behavior"] in {"emergency_escalation", "safety_refusal_with_education"}
    ]

    def _mean(vals: list[float]) -> float:
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    latencies = [float(c.get("retrieval_latency_ms", 0.0)) for c in case_results]

    return {
        "total_cases": len(case_results),
        "grounded_cases": len(grounded_cases),
        "unsupported_cases": len(ookb_cases),
        "safety_cases": len(safety_cases),
        "recall_at_k": _mean([c["page_recall"] for c in grounded_cases]),
        "source_hit_rate": _mean([c["source_hit"] for c in grounded_cases]),
        "mrr": _mean([c["mrr"] for c in grounded_cases]),
        "context_relevance": _mean([c["context_relevance"] for c in grounded_cases]),
        "citation_correctness": _mean([c["citation_correctness"] for c in grounded_cases]),
        "answer_faithfulness": _mean([c["faithfulness"] for c in grounded_cases]),
        "answer_completeness": _mean([c["completeness"] for c in grounded_cases]),
        "unsupported_abstention_rate": _mean([c["abstention_correct"] for c in ookb_cases]),
        "safety_compliance_rate": _mean([c["safety_correct"] for c in safety_cases]),
        "mean_retrieval_latency_ms": round(_mean(latencies), 2),
        "p95_retrieval_latency_ms": round(_percentile(latencies, 95), 2),
    }
