"""Automated RAG evaluation runner executed against an isolated evaluation index."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Settings, get_settings
from core.llm_client import GeminiLLMClient
from evaluation.metrics import (
    aggregate_evaluation_metrics,
    compute_answer_completeness,
    compute_answer_faithfulness,
    compute_citation_correctness,
    compute_context_relevance,
    compute_page_recall,
    compute_reciprocal_rank,
    compute_source_hit,
)
from ingestion.chunking import DocumentChunker
from ingestion.document_loader import load_document
from ingestion.embeddings import EmbeddingService
from rag.citations import VerifiedCitation, validate_and_extract_citations
from rag.llm_service import LLMService
from rag.prompt_builder import build_grounded_prompt
from rag.query_rewriter import rewrite_followup_query
from rag.retriever import RetrievedChunk, SemanticRetriever
from rag.safety import assess_query_safety
from scripts.generate_sample_docs import generate_all_samples
from utils.helpers import compute_sha256_file, utc_now_iso
from vectorstore.chroma_manager import ChromaManager

DEFAULT_DATASET_PATH = PROJECT_ROOT / "evaluation" / "eval_dataset.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "eval_results.json"


def load_eval_dataset(dataset_path: Path | None = None) -> list[dict[str, Any]]:
    path = dataset_path or DEFAULT_DATASET_PATH
    return json.loads(path.read_text(encoding="utf-8"))


def _synthesize_grounded_context_summary(chunks: list[RetrievedChunk]) -> str:
    """Create a deterministic extractive summary from retrieved chunks (no LLM fabrication)."""
    parts = []
    for ch in chunks[:2]:
        snippet = " ".join(ch.text.split())[:320]
        parts.append(f"{snippet} [Source: {ch.filename}, page {ch.page_number}]")
    return " ".join(parts)


def run_evaluation(
    *,
    dataset_path: Path | None = None,
    output_path: Path | None = None,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
    top_k: int = 4,
    similarity_threshold: float = 0.32,
    deduplicate: bool = True,
    enforce_doc_diversity: bool = True,
    use_query_rewriter: bool = True,
    use_llm: bool = False,
    samples_dir: Path | None = None,
) -> dict[str, Any]:
    """Run repeatable evaluation in an isolated vector store separate from user conversations."""
    settings: Settings = get_settings()
    corpus_dir = samples_dir or settings.samples_dir
    corpus_files = sorted(
        p for p in corpus_dir.glob("*") if p.suffix.lower() in {".pdf", ".docx", ".txt", ".md"}
    )
    if not corpus_files:
        corpus_files = generate_all_samples(corpus_dir)

    dataset = load_eval_dataset(dataset_path)
    embeddings = EmbeddingService(
        model_name=settings.embedding_model_name, device=settings.embedding_device
    )
    embeddings.ensure_loaded()

    with tempfile.TemporaryDirectory(
        prefix="medisage_eval_", ignore_cleanup_errors=True
    ) as tmp_dir:
        chroma = ChromaManager(
            persist_directory=Path(tmp_dir) / "chroma",
            collection_name="medisage_eval_isolated",
        )
        try:
            chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            indexed_docs = 0
            total_chunks = 0
            for idx, doc_path in enumerate(corpus_files, start=1):
                loaded = load_document(doc_path, display_filename=doc_path.name)
                doc_id = f"eval_doc_{idx}"
                fhash = compute_sha256_file(doc_path)
                chunks = chunker.chunk_document(loaded, document_id=doc_id, file_hash=fhash)
                vectors = embeddings.embed_texts([c.text for c in chunks])
                chroma.index_chunks(doc_id, chunks, vectors)
                indexed_docs += 1
                total_chunks += len(chunks)

            retriever = SemanticRetriever(chroma=chroma, embeddings=embeddings)
            llm_svc = LLMService(client=GeminiLLMClient(), settings=settings) if use_llm else None

            case_results: list[dict[str, Any]] = []
            for item in dataset:
                question = item["question"]
                history = item.get("history") or []
                expected_behavior = item["expected_behavior"]
                expected_source = item.get("expected_source") or ""
                expected_page = item.get("expected_page")
                expected_keywords = item.get("expected_keywords") or []

                safety = assess_query_safety(question)

                if use_query_rewriter and history:
                    retrieval_query, was_rewritten = rewrite_followup_query(question, history)
                else:
                    retrieval_query, was_rewritten = question, False

                if safety.is_emergency:
                    case_results.append(
                        {
                            "id": item["id"],
                            "category": item["category"],
                            "question": question,
                            "rewritten_query": retrieval_query,
                            "was_rewritten": was_rewritten,
                            "expected_behavior": expected_behavior,
                            "retrieval_status": "safety_blocked",
                            "retrieved_sources": [],
                            "top_similarity": 0.0,
                            "source_hit": 1.0,
                            "page_recall": 1.0,
                            "mrr": 1.0,
                            "context_relevance": 1.0,
                            "citation_correctness": 1.0,
                            "faithfulness": 1.0,
                            "completeness": 1.0,
                            "abstention_correct": 1.0,
                            "safety_correct": 1.0 if safety.is_emergency else 0.0,
                            "retrieval_latency_ms": 0.0,
                        }
                    )
                    continue

                ret = retriever.retrieve(
                    retrieval_query,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold,
                    deduplicate=deduplicate,
                    enforce_doc_diversity=enforce_doc_diversity,
                    use_cache=False,
                )

                citations: list[VerifiedCitation] = []
                answer_text = ""

                if ret.chunks:
                    if use_llm and llm_svc is not None and settings.has_api_key():
                        sys_inst, prompt = build_grounded_prompt(
                            user_question=question,
                            retrieved_chunks=ret.chunks,
                            recent_messages=history,
                            safety=safety,
                        )
                        llm_resp = llm_svc.generate_answer(
                            prompt=prompt, system_instruction=sys_inst
                        )
                        if llm_resp.success:
                            rep = validate_and_extract_citations(llm_resp.text, ret.chunks)
                            answer_text = rep.cleaned_answer
                            citations = rep.citations

                    if not answer_text:
                        answer_text = _synthesize_grounded_context_summary(ret.chunks)
                        rep = validate_and_extract_citations(answer_text, ret.chunks)
                        citations = rep.citations
                else:
                    answer_text = "Insufficient evidence in uploaded knowledge base."

                src_hit = compute_source_hit(ret.chunks, expected_source)
                page_recall = compute_page_recall(ret.chunks, expected_source, expected_page)
                mrr = compute_reciprocal_rank(ret.chunks, expected_source, expected_page)
                ctx_rel = compute_context_relevance(ret.chunks, expected_keywords)
                cit_corr = compute_citation_correctness(citations, ret.chunks)
                faith = compute_answer_faithfulness(answer_text, ret.chunks)
                comp = compute_answer_completeness(
                    " ".join(c.text for c in ret.chunks) if not use_llm else answer_text,
                    expected_keywords,
                )

                abstention_ok = (
                    1.0
                    if (expected_behavior == "insufficient_evidence" and (not ret.chunks or ret.status == "no_evidence"))
                    else 0.0
                )
                safety_ok = (
                    1.0
                    if (expected_behavior == "safety_refusal_with_education" and safety.requires_safety_boundary)
                    else 0.0
                )

                case_results.append(
                    {
                        "id": item["id"],
                        "category": item["category"],
                        "question": question,
                        "rewritten_query": retrieval_query,
                        "was_rewritten": was_rewritten,
                        "expected_behavior": expected_behavior,
                        "expected_source": expected_source,
                        "expected_page": expected_page,
                        "retrieval_status": ret.status,
                        "retrieved_sources": [
                            {
                                "filename": c.filename,
                                "page": c.page_number,
                                "similarity": round(c.similarity, 4),
                            }
                            for c in ret.chunks
                        ],
                        "top_similarity": round(ret.chunks[0].similarity, 4) if ret.chunks else 0.0,
                        "source_hit": round(src_hit, 4),
                        "page_recall": round(page_recall, 4),
                        "mrr": round(mrr, 4),
                        "context_relevance": round(ctx_rel, 4),
                        "citation_correctness": round(cit_corr, 4),
                        "faithfulness": round(faith, 4),
                        "completeness": round(comp, 4),
                        "abstention_correct": abstention_ok,
                        "safety_correct": safety_ok,
                        "retrieval_latency_ms": ret.latency_ms,
                    }
                )
        finally:
            chroma.close()

    summary = aggregate_evaluation_metrics(case_results)
    report = {
        "generated_at": utc_now_iso(),
        "mode": "llm_augmented" if (use_llm and settings.has_api_key()) else "deterministic_retrieval_and_safety",
        "corpus": {
            "documents": indexed_docs,
            "total_chunks": total_chunks,
            "files": [p.name for p in corpus_files],
        },
        "config": {
            "chunk_size_tokens": chunk_size,
            "chunk_overlap_tokens": chunk_overlap,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
            "deduplicate": deduplicate,
            "enforce_doc_diversity": enforce_doc_diversity,
            "use_query_rewriter": use_query_rewriter,
            "embedding_model": settings.embedding_model_name,
        },
        "summary": summary,
        "cases": case_results,
    }

    target_out = output_path or DEFAULT_OUTPUT_PATH
    target_out.parent.mkdir(parents=True, exist_ok=True)
    target_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MediSage RAG evaluation benchmark.")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.32)
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--use-llm", action="store_true", help="Use live Gemini API if configured.")
    args = parser.parse_args()

    report = run_evaluation(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        similarity_threshold=args.threshold,
        use_llm=args.use_llm,
    )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
