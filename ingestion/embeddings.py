"""Cached local CPU embedding service using sentence-transformers all-MiniLM-L6-v2."""

from __future__ import annotations

import threading
import time
from typing import Sequence

from core.logging_utils import get_logger
from utils.cache import get_shared_cache

logger = get_logger("ingestion.embeddings")

_MODEL_LOCK = threading.Lock()
_LOADED_MODELS: dict[tuple[str, str], object] = {}
_COLD_START_MS: dict[tuple[str, str], float] = {}


def get_cached_sentence_transformer(
    model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"
):
    """Load and cache the SentenceTransformer instance once per process."""
    key = (model_name, device)
    with _MODEL_LOCK:
        if key in _LOADED_MODELS:
            return _LOADED_MODELS[key], _COLD_START_MS[key], False

        from sentence_transformers import SentenceTransformer

        t0 = time.perf_counter()
        logger.info("Loading local CPU embedding model: %s (%s)", model_name, device)
        try:
            model = SentenceTransformer(
                model_name, device=device, local_files_only=True
            )
        except Exception:
            model = SentenceTransformer(model_name, device=device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _LOADED_MODELS[key] = model
        _COLD_START_MS[key] = elapsed_ms
        logger.info(
            "Embedding model %s ready in %.1f ms (device=%s)",
            model_name,
            elapsed_ms,
            device,
        )
        return model, elapsed_ms, True


class EmbeddingService:
    """CPU-friendly embedding generator with singleton model reuse and query caching."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.last_encode_ms: float = 0.0
        self.cache = get_shared_cache()

    def ensure_loaded(self) -> float:
        """Eagerly load the cached embedding model and return cold-start latency in ms."""
        _, cold_ms, _ = get_cached_sentence_transformer(self.model_name, self.device)
        return cold_ms

    def is_loaded(self) -> bool:
        """Return True if the embedding model is already warm in memory."""
        return (self.model_name, self.device) in _LOADED_MODELS

    @property
    def cold_start_ms(self) -> float:
        return _COLD_START_MS.get((self.model_name, self.device), 0.0)

    def embed_texts(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        """Generate normalized dense CPU embeddings for document chunks."""
        items = [t if t is not None else "" for t in texts]
        if not items:
            return []

        model, _, _ = get_cached_sentence_transformer(self.model_name, self.device)
        t0 = time.perf_counter()
        vectors = model.encode(
            items,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        self.last_encode_ms = (time.perf_counter() - t0) * 1000.0
        return [vec.tolist() for vec in vectors]

    def embed_query(self, query: str) -> list[float]:
        """Generate or return a cached embedding vector for a single query string."""
        clean_query = (query or "").strip()
        if not clean_query:
            raise ValueError("Cannot embed an empty query.")

        cached = self.cache.get_embedding(self.model_name, clean_query)
        if cached is not None:
            self.last_encode_ms = 0.0
            return cached

        vec = self.embed_texts([clean_query])[0]
        self.cache.set_embedding(self.model_name, clean_query, vec)
        return vec
