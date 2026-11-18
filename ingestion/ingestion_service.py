"""End-to-end document ingestion service connecting validation, parsing, chunking, embedding, SQLite, and ChromaDB."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.settings import Settings, get_settings
from core.logging_utils import get_logger
from database.db import DatabaseManager
from ingestion.chunking import DocumentChunker
from ingestion.document_loader import DocumentParseError, ScannedDocumentError, load_document
from ingestion.embeddings import EmbeddingService
from utils.cache import get_shared_cache
from utils.helpers import compute_sha256_bytes, sanitize_error_message
from utils.validators import (
    build_safe_storage_path,
    resolve_safe_stored_file,
    validate_uploaded_file,
)
from vectorstore.chroma_manager import ChromaManager

logger = get_logger("ingestion.service")


@dataclass
class IngestionOutcome:
    """Structured result returned after ingesting a document."""

    success: bool
    status: str  # "indexed", "duplicate", "failed", "invalid"
    message: str
    document_id: str = ""
    original_filename: str = ""
    page_count: int = 0
    chunk_count: int = 0
    document_record: dict[str, Any] | None = None


class IngestionService:
    """Coordinates safe document upload, extraction, embedding, and synchronized cleanup."""

    def __init__(
        self,
        *,
        db: DatabaseManager,
        chroma: ChromaManager,
        embeddings: EmbeddingService,
        settings: Settings | None = None,
    ) -> None:
        self.db = db
        self.chroma = chroma
        self.embeddings = embeddings
        self.settings = settings or get_settings()
        self.settings.upload_dir.mkdir(parents=True, exist_ok=True)
        self.cache = get_shared_cache()

    def _effective_chunk_params(self) -> tuple[int, int]:
        chunk_size = int(
            self.db.get_setting("chunk_size_tokens", self.settings.chunk_size_tokens)
        )
        chunk_overlap = int(
            self.db.get_setting(
                "chunk_overlap_tokens", self.settings.chunk_overlap_tokens
            )
        )
        if chunk_overlap >= chunk_size:
            chunk_overlap = max(0, chunk_size // 5)
        return chunk_size, chunk_overlap

    def ingest_bytes(
        self,
        *,
        filename: str,
        content: bytes,
        mime_type: str | None = None,
        source_origin: str = "upload",
        replace_existing: bool = False,
    ) -> IngestionOutcome:
        """Validate, store, parse, chunk, embed, and index uploaded document bytes."""
        validation = validate_uploaded_file(
            filename=filename,
            content=content,
            mime_type=mime_type,
            max_size_mb=self.settings.max_upload_size_mb,
        )
        if not validation.is_valid:
            return IngestionOutcome(
                success=False,
                status="invalid",
                message=validation.error_message,
                original_filename=validation.original_filename,
            )

        file_hash = compute_sha256_bytes(content)
        existing = self.db.get_document_by_hash(file_hash)
        if existing is None and source_origin == "sample":
            for doc in self.db.list_documents(status="indexed"):
                if (
                    doc.get("source_origin") == "sample"
                    and doc.get("original_filename") == validation.original_filename
                ):
                    existing = doc
                    break

        if existing and existing["processing_status"] == "indexed" and not replace_existing:
            return IngestionOutcome(
                success=True,
                status="duplicate",
                message=(
                    f"'{validation.original_filename}' matches an already indexed "
                    f"document ('{existing['original_filename']}'). Skipped duplicate ingestion."
                ),
                document_id=existing["document_id"],
                original_filename=validation.original_filename,
                page_count=existing["page_count"],
                chunk_count=existing["chunk_count"],
                document_record=existing,
            )

        if existing:
            # Clean up prior failed/replaced entry safely
            self.delete_document_completely(existing["document_id"])

        safe_name, target_path = build_safe_storage_path(
            self.settings.upload_dir, validation.original_filename, file_hash
        )
        target_path.write_bytes(content)

        doc_record = self.db.add_document(
            stored_filename=safe_name,
            original_filename=validation.original_filename,
            file_hash=file_hash,
            file_type=validation.extension.lstrip("."),
            file_size_bytes=validation.size_bytes,
            processing_status="processing",
            source_origin=source_origin,
        )
        document_id = doc_record["document_id"]

        try:
            loaded = load_document(
                target_path, display_filename=validation.original_filename
            )
            chunk_size, chunk_overlap = self._effective_chunk_params()
            chunker = DocumentChunker(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            chunks = chunker.chunk_document(
                loaded, document_id=document_id, file_hash=file_hash
            )
            if not chunks:
                raise DocumentParseError("No text chunks could be generated from this file.")

            vectors = self.embeddings.embed_texts([c.text for c in chunks])
            indexed_count = self.chroma.index_chunks(document_id, chunks, vectors)

            updated = self.db.update_document_status(
                document_id,
                processing_status="indexed",
                page_count=loaded.page_count,
                chunk_count=indexed_count,
                error_details="",
            )
            self.cache.invalidate_on_index_change()
            return IngestionOutcome(
                success=True,
                status="indexed",
                message=(
                    f"Indexed '{validation.original_filename}' "
                    f"({loaded.page_count} pages, {indexed_count} chunks)."
                ),
                document_id=document_id,
                original_filename=validation.original_filename,
                page_count=loaded.page_count,
                chunk_count=indexed_count,
                document_record=updated,
            )
        except (ScannedDocumentError, DocumentParseError) as exc:
            err_msg = sanitize_error_message(exc)
            # Remove unindexed file from upload storage on failed parse
            if target_path.exists():
                target_path.unlink(missing_ok=True)
            updated = self.db.update_document_status(
                document_id,
                processing_status="failed",
                page_count=0,
                chunk_count=0,
                error_details=err_msg,
            )
            return IngestionOutcome(
                success=False,
                status="failed",
                message=err_msg,
                document_id=document_id,
                original_filename=validation.original_filename,
                document_record=updated,
            )
        except Exception as exc:
            err_msg = sanitize_error_message(exc)
            logger.exception("Unexpected ingestion failure for document_id=%s", document_id)
            if target_path.exists():
                target_path.unlink(missing_ok=True)
            updated = self.db.update_document_status(
                document_id,
                processing_status="failed",
                page_count=0,
                chunk_count=0,
                error_details=err_msg,
            )
            return IngestionOutcome(
                success=False,
                status="failed",
                message=err_msg,
                document_id=document_id,
                original_filename=validation.original_filename,
                document_record=updated,
            )

    def ingest_file_path(
        self, file_path: Path | str, *, source_origin: str = "upload"
    ) -> IngestionOutcome:
        """Read a local file path and ingest it through the validated pipeline."""
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            return IngestionOutcome(
                success=False,
                status="invalid",
                message="Source file does not exist.",
                original_filename=path.name,
            )
        return self.ingest_bytes(
            filename=path.name,
            content=path.read_bytes(),
            source_origin=source_origin,
        )

    def reindex_document(self, document_id: str) -> IngestionOutcome:
        """Re-chunk and re-embed an already stored document using current settings."""
        record = self.db.get_document(document_id)
        if not record:
            return IngestionOutcome(
                success=False, status="failed", message="Document record not found."
            )
        try:
            file_path = resolve_safe_stored_file(
                self.settings.upload_dir, record["stored_filename"]
            )
        except ValueError as exc:
            return IngestionOutcome(
                success=False,
                status="failed",
                message=sanitize_error_message(exc),
                document_id=document_id,
            )
        if not file_path.exists():
            return IngestionOutcome(
                success=False,
                status="failed",
                message="Stored source file is no longer present on disk.",
                document_id=document_id,
                original_filename=record["original_filename"],
            )

        try:
            loaded = load_document(
                file_path, display_filename=record["original_filename"]
            )
            chunk_size, chunk_overlap = self._effective_chunk_params()
            chunker = DocumentChunker(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            chunks = chunker.chunk_document(
                loaded, document_id=document_id, file_hash=record["file_hash"]
            )
            vectors = self.embeddings.embed_texts([c.text for c in chunks])
            indexed_count = self.chroma.index_chunks(document_id, chunks, vectors)
            updated = self.db.update_document_status(
                document_id,
                processing_status="indexed",
                page_count=loaded.page_count,
                chunk_count=indexed_count,
                error_details="",
            )
            self.cache.invalidate_on_index_change()
            return IngestionOutcome(
                success=True,
                status="indexed",
                message=f"Re-indexed '{record['original_filename']}' ({indexed_count} chunks).",
                document_id=document_id,
                original_filename=record["original_filename"],
                page_count=loaded.page_count,
                chunk_count=indexed_count,
                document_record=updated,
            )
        except Exception as exc:
            err = sanitize_error_message(exc)
            updated = self.db.update_document_status(
                document_id, processing_status="failed", error_details=err
            )
            return IngestionOutcome(
                success=False,
                status="failed",
                message=err,
                document_id=document_id,
                original_filename=record["original_filename"],
                document_record=updated,
            )

    def delete_document_completely(self, document_id: str) -> bool:
        """Delete source file on disk, vectors in ChromaDB, and metadata in SQLite."""
        record = self.db.get_document(document_id)
        if not record:
            return False

        # 1. Delete vectors
        self.chroma.delete_document_vectors(document_id)

        # 2. Delete uploaded file safely
        stored_name = record.get("stored_filename", "")
        if stored_name:
            try:
                file_path = resolve_safe_stored_file(
                    self.settings.upload_dir, stored_name
                )
                if file_path.exists():
                    file_path.unlink(missing_ok=True)
            except ValueError:
                pass

        # 3. Delete SQLite metadata
        deleted = self.db.delete_document(document_id)
        self.cache.invalidate_on_index_change()
        return deleted

    def verify_and_repair_consistency(self) -> dict[str, int]:
        """Deduplicate auto-generated sample duplicates and remove orphaned ChromaDB vectors."""
        docs = self.db.list_documents()
        removed_sample_dups = 0
        seen_samples: set[str] = set()
        for doc in docs:
            if doc.get("source_origin") == "sample" and doc.get("processing_status") == "indexed":
                fname = doc["original_filename"]
                if fname in seen_samples:
                    if self.delete_document_completely(doc["document_id"]):
                        removed_sample_dups += 1
                else:
                    seen_samples.add(fname)

        remaining_ids = {d["document_id"] for d in self.db.list_documents()}
        removed_orphans = self.chroma.delete_orphaned_vectors(remaining_ids)
        return {
            "removed_sample_duplicates": removed_sample_dups,
            "removed_orphaned_vectors": removed_orphans,
        }
