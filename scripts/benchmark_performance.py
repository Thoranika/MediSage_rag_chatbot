"""Measure CPU embedding cold-start, warm encoding, retrieval, and E2E pipeline latency."""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from core.llm_client import BaseLLMClient, LLMResponse
from database.db import DatabaseManager
from ingestion.embeddings import EmbeddingService
from ingestion.ingestion_service import IngestionService
from rag.llm_service import LLMService
from rag.rag_pipeline import RAGPipeline
from rag.retriever import SemanticRetriever
from scripts.generate_sample_docs import generate_all_samples
from vectorstore.chroma_manager import ChromaManager


class _InstantBenchmarkLLM(BaseLLMClient):
    """Deterministic zero-network stub to isolate local CPU pipeline latency."""

    def generate(self, prompt: str, settings=None) -> LLMResponse:
        return LLMResponse(
            success=True,
            text=(
                "Stage 1 hypertension is 130-139/80-89 mmHg and Stage 2 is >=140/90 mmHg "
                "[Source: WHO_Hypertension_Clinical_Guideline.pdf, page 1]."
            ),
            provider="GEMINI",
            model_id="mock-benchmark",
            latency_ms=0.1,
        )


def run_benchmark() -> dict:
    settings = get_settings()
    samples = generate_all_samples(settings.samples_dir)

    emb = EmbeddingService(
        model_name=settings.embedding_model_name, device=settings.embedding_device
    )
    cold_start_ms = emb.ensure_loaded()

    t_warm_q = time.perf_counter()
    emb.embed_texts(["What blood pressure defines Stage 2 hypertension?"])
    warm_single_encode_ms = (time.perf_counter() - t_warm_q) * 1000.0

    with tempfile.TemporaryDirectory(
        prefix="medisage_bench_", ignore_cleanup_errors=True
    ) as tmp_dir:
        tmp = Path(tmp_dir)
        cfg = settings.model_copy(
            update={
                "sqlite_db_path": tmp / "bench.db",
                "upload_dir": tmp / "uploads",
                "chroma_persist_dir": tmp / "chroma",
                "chroma_collection_name": "bench_chunks",
            }
        )
        db = DatabaseManager(cfg.sqlite_db_path)
        chroma = ChromaManager(cfg.chroma_persist_dir, cfg.chroma_collection_name)
        try:
            ing = IngestionService(db=db, chroma=chroma, embeddings=emb, settings=cfg)

            t_ingest = time.perf_counter()
            total_chunks = 0
            for sp in samples:
                out = ing.ingest_file_path(sp, source_origin="sample")
                total_chunks += out.chunk_count
            total_ingest_ms = (time.perf_counter() - t_ingest) * 1000.0

            retriever = SemanticRetriever(chroma=chroma, embeddings=emb)
            queries = [
                "What systolic blood pressure defines Stage 1 and Stage 2 hypertension?",
                "At what eGFR threshold is metformin contraindicated?",
                "What are the five CURB-65 pneumonia severity criteria?",
            ]
            retrieval_times = []
            for q in queries:
                res = retriever.retrieve(q, top_k=4, similarity_threshold=0.32, use_cache=False)
                retrieval_times.append(res.latency_ms)

            pipeline = RAGPipeline(
                retriever=retriever,
                llm_service=LLMService(client=_InstantBenchmarkLLM(), settings=cfg),
                db=db,
                settings=cfg,
            )
            t_e2e = time.perf_counter()
            _ = pipeline.answer_question(queries[0])
            e2e_local_ms = (time.perf_counter() - t_e2e) * 1000.0
        finally:
            chroma.close()

    report = {
        "device": settings.embedding_device,
        "embedding_model": settings.embedding_model_name,
        "embedding_cold_start_ms": round(cold_start_ms, 2),
        "warm_single_query_encode_ms": round(warm_single_encode_ms, 2),
        "corpus_documents": len(samples),
        "corpus_chunks": total_chunks,
        "full_corpus_ingest_ms": round(total_ingest_ms, 2),
        "mean_retrieval_ms": round(sum(retrieval_times) / len(retrieval_times), 2),
        "local_e2e_without_cloud_rtt_ms": round(e2e_local_ms, 2),
    }
    return report


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2))
