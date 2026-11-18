"""Persistent ChromaDB vector store manager with metadata filtering and cleanup."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import chromadb

from core.logging_utils import get_logger
from utils.cache import get_shared_cache

if TYPE_CHECKING:
    from ingestion.chunking import TextChunk

logger = get_logger("vectorstore.chroma")


class ChromaManager:
    """Manages persistent ChromaDB storage, semantic search, and deletion."""

    def __init__(
        self,
        persist_directory: Path | str,
        collection_name: str = "medisage_chunks",
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(path=str(self.persist_directory))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._cache = get_shared_cache()

    def health_check(self) -> bool:
        """Verify ChromaDB collection is accessible."""
        try:
            _ = self._collection.count()
            return True
        except Exception:
            return False

    def count(self) -> int:
        """Return total stored vector chunks."""
        return int(self._collection.count())

    def index_chunks(
        self,
        document_id: str,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> int:
        """Idempotently replace any existing vectors for document_id and upsert new chunks."""
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings lengths must match.")

        # Remove prior chunks for this document_id before indexing
        self.delete_document_vectors(document_id, invalidate_cache=False)

        if not chunks:
            self._cache.invalidate_on_index_change()
            return 0

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [c.to_metadata() for c in chunks]

        # Batch upserts to stay well within SQLite/Chroma parameter limits
        batch_size = 128
        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            self._collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

        self._cache.invalidate_on_index_change()
        logger.info("Indexed %d chunks for document_id=%s", len(chunks), document_id)
        return len(chunks)

    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 4,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Query ChromaDB and return normalized chunk matches with cosine similarity."""
        total = self.count()
        if total == 0:
            return []

        n_results = min(max(1, int(top_k)), total)
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        raw = self._collection.query(**kwargs)
        ids_list = (raw.get("ids") or [[]])[0]
        docs_list = (raw.get("documents") or [[]])[0]
        metas_list = (raw.get("metadatas") or [[]])[0]
        dist_list = (raw.get("distances") or [[]])[0]

        results: list[dict[str, Any]] = []
        for cid, text, meta, dist in zip(ids_list, docs_list, metas_list, dist_list):
            distance = float(dist) if dist is not None else 1.0
            similarity = max(0.0, min(1.0, 1.0 - distance))
            meta_dict = dict(meta or {})
            results.append(
                {
                    "chunk_id": cid,
                    "text": text or "",
                    "document_id": str(meta_dict.get("document_id", "")),
                    "filename": str(meta_dict.get("filename", "")),
                    "page_number": int(meta_dict.get("page_number", 1)),
                    "section_title": str(meta_dict.get("section_title", "")),
                    "chunk_index": int(meta_dict.get("chunk_index", 0)),
                    "token_count": int(meta_dict.get("token_count", 0)),
                    "ingestion_timestamp": str(meta_dict.get("ingestion_timestamp", "")),
                    "distance": round(distance, 4),
                    "similarity": round(similarity, 4),
                }
            )
        return results

    def get_document_chunks(self, document_id: str) -> list[dict[str, Any]]:
        """Retrieve all stored chunks for a document ordered by chunk_index."""
        raw = self._collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"],
        )
        ids = raw.get("ids") or []
        docs = raw.get("documents") or []
        metas = raw.get("metadatas") or []

        items: list[dict[str, Any]] = []
        for cid, text, meta in zip(ids, docs, metas):
            m = dict(meta or {})
            items.append(
                {
                    "chunk_id": cid,
                    "text": text or "",
                    "page_number": int(m.get("page_number", 1)),
                    "section_title": str(m.get("section_title", "")),
                    "chunk_index": int(m.get("chunk_index", 0)),
                    "token_count": int(m.get("token_count", 0)),
                }
            )
        items.sort(key=lambda x: x["chunk_index"])
        return items

    def delete_document_vectors(
        self, document_id: str, *, invalidate_cache: bool = True
    ) -> None:
        """Delete all vector chunks belonging to a document_id."""
        try:
            existing = self._collection.get(where={"document_id": document_id})
            ids = existing.get("ids") or []
            if ids:
                self._collection.delete(ids=ids)
                logger.info("Deleted %d vector chunks for document_id=%s", len(ids), document_id)
        except Exception as exc:
            logger.warning("Error deleting vectors for %s: %s", document_id, exc)
        if invalidate_cache:
            self._cache.invalidate_on_index_change()

    def get_all_vector_document_ids(self) -> set[str]:
        """Return the set of unique document_id values currently stored in ChromaDB."""
        if self.count() == 0:
            return set()
        raw = self._collection.get(include=["metadatas"])
        metas = raw.get("metadatas") or []
        return {
            str(m.get("document_id", ""))
            for m in metas
            if m and m.get("document_id")
        }

    def delete_orphaned_vectors(self, valid_document_ids: set[str]) -> int:
        """Remove any ChromaDB vector chunks whose document_id is absent from SQLite."""
        stored_doc_ids = self.get_all_vector_document_ids()
        orphans = [did for did in stored_doc_ids if did and did not in valid_document_ids]
        removed = 0
        for did in orphans:
            existing = self._collection.get(where={"document_id": did})
            ids = existing.get("ids") or []
            if ids:
                self._collection.delete(ids=ids)
                removed += len(ids)
        if removed:
            self._cache.invalidate_on_index_change()
            logger.info("Removed %d orphaned vector chunks across %d doc IDs", removed, len(orphans))
        return removed

    def clear_collection(self) -> None:
        """Delete and recreate the ChromaDB collection."""
        try:
            self._client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._cache.invalidate_on_index_change()

    def close(self) -> None:
        """Release internal ChromaDB client references (helps Windows temp-dir cleanup)."""
        import gc

        try:
            system = getattr(self._client, "_system", None)
            if system is not None and hasattr(system, "stop"):
                system.stop()
        except Exception:
            pass
        self._collection = None  # type: ignore[assignment]
        self._client = None  # type: ignore[assignment]
        gc.collect()
