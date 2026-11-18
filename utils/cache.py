"""Deterministic cache for query embeddings and retrieval results with index invalidation."""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any


class DeterministicRAGCache:
    """Thread-safe LRU cache invalidated whenever indexed documents change."""

    def __init__(self, max_entries: int = 256) -> None:
        self._lock = threading.Lock()
        self._max_entries = max_entries
        self._index_version = 1
        self._embedding_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._retrieval_cache: OrderedDict[str, Any] = OrderedDict()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _hash_key(parts: dict[str, Any]) -> str:
        payload = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def index_version(self) -> int:
        with self._lock:
            return self._index_version

    def get_embedding(self, model_name: str, text: str) -> list[float] | None:
        key = self._hash_key({"model": model_name, "text": text})
        with self._lock:
            val = self._embedding_cache.get(key)
            if val is not None:
                self._embedding_cache.move_to_end(key)
                self._hits += 1
                return list(val)
            self._misses += 1
            return None

    def set_embedding(self, model_name: str, text: str, vector: list[float]) -> None:
        key = self._hash_key({"model": model_name, "text": text})
        with self._lock:
            self._embedding_cache[key] = list(vector)
            self._embedding_cache.move_to_end(key)
            while len(self._embedding_cache) > self._max_entries:
                self._embedding_cache.popitem(last=False)

    def get_retrieval(
        self,
        query: str,
        top_k: int,
        threshold: float,
        filters: dict[str, Any] | None = None,
        collection_name: str = "",
    ) -> Any | None:
        key = self._hash_key(
            {
                "v": self._index_version,
                "collection": collection_name,
                "query": query.strip().lower(),
                "top_k": top_k,
                "threshold": round(threshold, 4),
                "filters": filters or {},
            }
        )
        with self._lock:
            val = self._retrieval_cache.get(key)
            if val is not None:
                self._retrieval_cache.move_to_end(key)
                self._hits += 1
                return val
            self._misses += 1
            return None

    def set_retrieval(
        self,
        query: str,
        top_k: int,
        threshold: float,
        filters: dict[str, Any] | None,
        collection_name: str,
        value: Any,
    ) -> None:
        key = self._hash_key(
            {
                "v": self._index_version,
                "collection": collection_name,
                "query": query.strip().lower(),
                "top_k": top_k,
                "threshold": round(threshold, 4),
                "filters": filters or {},
            }
        )
        with self._lock:
            self._retrieval_cache[key] = value
            self._retrieval_cache.move_to_end(key)
            while len(self._retrieval_cache) > self._max_entries:
                self._retrieval_cache.popitem(last=False)

    def invalidate_on_index_change(self) -> int:
        """Invalidate retrieval cache entries after document upload, re-index, or deletion."""
        with self._lock:
            self._index_version += 1
            self._retrieval_cache.clear()
            return self._index_version

    def clear_all(self) -> None:
        """Clear both retrieval and embedding caches."""
        with self._lock:
            self._index_version += 1
            self._retrieval_cache.clear()
            self._embedding_cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict[str, int]:
        """Return cache telemetry."""
        with self._lock:
            return {
                "index_version": self._index_version,
                "embedding_entries": len(self._embedding_cache),
                "retrieval_entries": len(self._retrieval_cache),
                "hits": self._hits,
                "misses": self._misses,
            }


_SHARED_CACHE = DeterministicRAGCache()


def get_shared_cache() -> DeterministicRAGCache:
    """Return the process-wide deterministic cache instance."""
    return _SHARED_CACHE
