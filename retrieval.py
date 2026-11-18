"""Top-level retrieval module re-exporting SemanticRetriever (Task 8 compatibility)."""

from rag.retriever import RetrievalResult, RetrievedChunk, SemanticRetriever

__all__ = ["RetrievalResult", "RetrievedChunk", "SemanticRetriever"]
