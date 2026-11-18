"""Tests for ChromaDB persistence, semantic retrieval, filtering, deduplication, and cache invalidation."""

from __future__ import annotations

from ingestion.ingestion_service import IngestionService
from rag.retriever import SemanticRetriever
from utils.cache import get_shared_cache
from vectorstore.chroma_manager import ChromaManager


def test_chroma_persistence_retrieval_and_metadata_filtering(
    isolated_env, shared_embedding_service, sample_pdf_bytes: bytes, sample_docx_bytes: bytes
) -> None:
    cfg, db, chroma = isolated_env
    ing = IngestionService(
        db=db, chroma=chroma, embeddings=shared_embedding_service, settings=cfg
    )
    pdf_out = ing.ingest_bytes(filename="hypertension.pdf", content=sample_pdf_bytes)
    docx_out = ing.ingest_bytes(filename="pneumonia.docx", content=sample_docx_bytes)
    assert pdf_out.success and docx_out.success

    # Simulate app restart by instantiating a new ChromaManager pointing to same directory
    reopened = ChromaManager(cfg.chroma_persist_dir, cfg.chroma_collection_name)
    assert reopened.count() == pdf_out.chunk_count + docx_out.chunk_count

    retriever = SemanticRetriever(chroma=reopened, embeddings=shared_embedding_service)
    res = retriever.retrieve(
        "Which drugs are contraindicated in pregnancy for hypertension?",
        top_k=3,
        similarity_threshold=0.20,
    )
    assert res.status in {"ok", "low_confidence"}
    assert len(res.chunks) >= 1
    assert any(
        c.filename == "hypertension.pdf" and c.page_number == 2 for c in res.chunks
    )

    # Metadata filter restricts strictly to selected document
    filtered = retriever.retrieve(
        "CURB-65 respiratory rate criteria",
        top_k=3,
        similarity_threshold=0.15,
        metadata_filters={"document_id": docx_out.document_id},
    )
    assert filtered.chunks
    assert all(c.document_id == docx_out.document_id for c in filtered.chunks)

    # Out-of-knowledge-base query triggers explicit empty result with threshold
    ookb = retriever.retrieve(
        "quantum entanglement superconductivity lattice crystal Hamiltonian",
        top_k=3,
        similarity_threshold=0.45,
    )
    assert ookb.status == "no_evidence"
    assert ookb.chunks == []


def test_cache_invalidation_on_document_delete(
    isolated_env, shared_embedding_service, sample_pdf_bytes: bytes
) -> None:
    cfg, db, chroma = isolated_env
    ing = IngestionService(
        db=db, chroma=chroma, embeddings=shared_embedding_service, settings=cfg
    )
    out = ing.ingest_bytes(filename="hypertension.pdf", content=sample_pdf_bytes)
    retriever = SemanticRetriever(chroma=chroma, embeddings=shared_embedding_service)

    cache = get_shared_cache()
    v_before = cache.index_version

    first = retriever.retrieve("Stage 1 hypertension", top_k=2, similarity_threshold=0.15)
    assert first.chunks
    second = retriever.retrieve("Stage 1 hypertension", top_k=2, similarity_threshold=0.15)
    assert second.from_cache is True

    # Deleting document invalidates retrieval cache
    ing.delete_document_completely(out.document_id)
    assert cache.index_version > v_before

    after_del = retriever.retrieve("Stage 1 hypertension", top_k=2, similarity_threshold=0.15)
    assert after_del.from_cache is False
    assert after_del.status == "no_evidence"
    assert after_del.chunks == []
