"""Tests for validation, PDF/DOCX extraction, scanned/corrupt handling, and ingestion service."""

from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.document_loader import (
    DocumentParseError,
    ScannedDocumentError,
    load_document,
)
from ingestion.ingestion_service import IngestionService
from utils.validators import build_safe_storage_path, validate_uploaded_file


def test_file_validation_rejects_invalid_empty_oversized_and_bad_magic() -> None:
    # Unsupported extension
    res_ext = validate_uploaded_file("malware.exe", b"MZ...", max_size_mb=5)
    assert res_ext.is_valid is False

    # Empty file
    res_empty = validate_uploaded_file("empty.pdf", b"", max_size_mb=5)
    assert res_empty.is_valid is False

    # Oversized file
    big_pdf = b"%PDF-1.4\n" + b"a" * (2 * 1024 * 1024)
    res_big = validate_uploaded_file("big.pdf", big_pdf, max_size_mb=1)
    assert res_big.is_valid is False

    # Corrupt magic bytes pretending to be PDF
    res_bad_pdf = validate_uploaded_file("fake.pdf", b"NOT_A_PDF", max_size_mb=5)
    assert res_bad_pdf.is_valid is False

    # Corrupt DOCX archive
    res_bad_docx = validate_uploaded_file("fake.docx", b"PK\x03\x04corrupt", max_size_mb=5)
    assert res_bad_docx.is_valid is False


def test_path_traversal_prevented_in_storage_path(tmp_path: Path) -> None:
    safe_name, target = build_safe_storage_path(
        tmp_path, "../../etc/passwd.pdf", "abcd1234" * 8
    )
    assert "/" not in safe_name and "\\" not in safe_name
    assert target.resolve().is_relative_to(tmp_path.resolve())


def test_pdf_extraction_preserves_pages_and_strips_repeated_headers(
    tmp_path: Path, sample_pdf_bytes: bytes
) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(sample_pdf_bytes)

    loaded = load_document(pdf_path, display_filename="sample.pdf")
    assert loaded.page_count == 3
    assert [p.page_number for p in loaded.pages] == [1, 2, 3]
    assert "Stage 1 hypertension" in loaded.pages[0].text
    assert "ACE inhibitors" in loaded.pages[1].text
    # Repeated header across all 3 pages should be stripped
    assert "Clinical Header Standard" not in loaded.pages[0].text


def test_docx_extraction_preserves_headings_and_tables(
    tmp_path: Path, sample_docx_bytes: bytes
) -> None:
    docx_path = tmp_path / "pneumonia.docx"
    docx_path.write_bytes(sample_docx_bytes)

    loaded = load_document(docx_path, display_filename="pneumonia.docx")
    assert loaded.page_count >= 1
    combined = "\n".join(p.text for p in loaded.pages)
    assert "CURB-65" in combined
    assert "| Criterion | Threshold |" in combined
    assert "| Confusion |" in combined


def test_scanned_and_corrupt_pdf_honest_failure(
    tmp_path: Path, scanned_pdf_bytes: bytes
) -> None:
    scan_path = tmp_path / "scanned.pdf"
    scan_path.write_bytes(scanned_pdf_bytes)
    with pytest.raises(ScannedDocumentError, match="OCR is not enabled"):
        load_document(scan_path)

    corrupt_path = tmp_path / "corrupt.pdf"
    corrupt_path.write_bytes(b"%PDF-1.4\n%%EOF_BROKEN_GARBAGE")
    with pytest.raises(DocumentParseError):
        load_document(corrupt_path)


def test_ingestion_idempotency_duplicate_detection_and_clean_deletion(
    isolated_env, shared_embedding_service, sample_pdf_bytes: bytes, scanned_pdf_bytes: bytes
) -> None:
    cfg, db, chroma = isolated_env
    svc = IngestionService(
        db=db, chroma=chroma, embeddings=shared_embedding_service, settings=cfg
    )

    out1 = svc.ingest_bytes(filename="hypertension.pdf", content=sample_pdf_bytes)
    assert out1.success is True
    assert out1.status == "indexed"
    assert out1.page_count == 3
    assert out1.chunk_count >= 3
    assert chroma.count() == out1.chunk_count

    # Duplicate upload skipped cleanly
    out2 = svc.ingest_bytes(filename="hypertension_copy.pdf", content=sample_pdf_bytes)
    assert out2.status == "duplicate"
    assert chroma.count() == out1.chunk_count

    # Scanned PDF fails gracefully without adding vectors
    out_scan = svc.ingest_bytes(filename="scan.pdf", content=scanned_pdf_bytes)
    assert out_scan.success is False
    assert out_scan.status == "failed"
    assert "OCR is not enabled" in out_scan.message

    # Complete deletion cleans disk file, ChromaDB vectors, and SQLite row
    doc_rec = db.get_document(out1.document_id)
    stored_file = cfg.upload_dir / doc_rec["stored_filename"]
    assert stored_file.exists()

    assert svc.delete_document_completely(out1.document_id) is True
    assert not stored_file.exists()
    assert chroma.count() == 0
    assert db.get_document(out1.document_id) is None
