"""Tests for heading-aware chunking and cached local CPU embeddings."""

from __future__ import annotations

from ingestion.chunking import DocumentChunker
from ingestion.document_loader import ExtractedPage, LoadedDocument
from ingestion.embeddings import EmbeddingService


def test_chunker_attaches_metadata_and_preserves_table_headers() -> None:
    table_rows = "\n".join(
        [f"| Drug_{i} | Class_{i} | Reference note for row {i} with extra clinical detail |" for i in range(25)]
    )
    table_text = "| Drug | Class | Notes |\n| --- | --- | --- |\n" + table_rows
    loaded = LoadedDocument(
        filename="pharma.pdf",
        file_type="pdf",
        page_count=1,
        pages=[
            ExtractedPage(
                filename="pharma.pdf",
                page_number=2,
                section_title="Drug Reference Table",
                text="## Antihypertensive Agents\n\n" + table_text,
            )
        ],
    )
    chunker = DocumentChunker(chunk_size=180, chunk_overlap=40)
    chunks = chunker.chunk_document(
        loaded, document_id="doc_test1", file_hash="hash123"
    )
    assert len(chunks) >= 2
    for idx, ch in enumerate(chunks):
        meta = ch.to_metadata()
        assert meta["document_id"] == "doc_test1"
        assert meta["filename"] == "pharma.pdf"
        assert meta["page_number"] == 2
        assert meta["chunk_index"] == idx
        assert meta["file_hash"] == "hash123"
        assert meta["ingestion_timestamp"]
    # Table header row should be preserved across split table chunks
    assert "| Drug | Class | Notes |" in chunks[1].text


def test_cpu_embedding_shape_and_singleton_caching(
    shared_embedding_service: EmbeddingService,
) -> None:
    assert shared_embedding_service.is_loaded() is True
    vectors = shared_embedding_service.embed_texts(
        ["Stage 1 hypertension blood pressure range", "Metformin renal contraindication"]
    )
    assert len(vectors) == 2
    assert len(vectors[0]) == 384

    # Query caching returns deterministic identical vector
    v1 = shared_embedding_service.embed_query("blood pressure classification")
    v2 = shared_embedding_service.embed_query("blood pressure classification")
    assert v1 == v2

    # Second EmbeddingService instance shares singleton model without reload
    second = EmbeddingService(model_name="all-MiniLM-L6-v2", device="cpu")
    assert second.is_loaded() is True
