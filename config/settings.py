"""Centralized configuration management for MediSage."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
_PLACEHOLDER_KEYS = {"", "your_gemini_api_key_here", "REPLACE_ME", "changeme"}

# Load .env once if present without overwriting existing environment variables
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _resolve_path(raw_path: str | Path, default_rel: str) -> Path:
    val = str(raw_path).strip() if raw_path else default_rel
    p = Path(val)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


class Settings(BaseModel):
    """Application settings loaded from environment variables with safe defaults."""

    # LLM Provider
    llm_provider: Literal["GEMINI"] = Field(default="GEMINI")
    llm_model_id: str = Field(default="gemini-2.5-flash")
    llm_api_version: str = Field(default="v1beta")

    # Embeddings & Vector Store
    embedding_model_name: str = Field(default="all-MiniLM-L6-v2")
    embedding_device: str = Field(default="cpu")
    chroma_persist_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "chroma_db")
    chroma_collection_name: str = Field(default="medisage_chunks")

    # Storage & Limits
    sqlite_db_path: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "medisage.db")
    upload_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "uploads")
    samples_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "samples")
    max_upload_size_mb: int = Field(default=20, ge=1, le=100)

    # Chunking & Retrieval (Milestone 3 tuned defaults)
    chunk_size_tokens: int = Field(default=600, ge=150, le=1500)
    chunk_overlap_tokens: int = Field(default=100, ge=0, le=400)
    retrieval_top_k: int = Field(default=4, ge=1, le=12)
    similarity_threshold: float = Field(default=0.32, ge=0.0, le=0.95)
    max_history_turns: int = Field(default=4, ge=1, le=10)

    # Generation
    llm_temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    llm_max_output_tokens: int = Field(default=1024, ge=128, le=4096)
    llm_timeout_seconds: int = Field(default=30, ge=5, le=120)
    response_style: Literal["Concise", "Detailed", "Plain Language"] = Field(
        default="Concise"
    )

    # Logging
    log_level: str = Field(default="INFO")
    log_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "logs")

    @field_validator("chunk_overlap_tokens")
    @classmethod
    def validate_overlap(cls, overlap: int, info) -> int:
        chunk_size = info.data.get("chunk_size_tokens", 600)
        if overlap >= chunk_size:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
        return overlap

    def get_api_key(self) -> str:
        """Read the configured cloud LLM API key securely from the environment."""
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        raw = os.getenv("GEMINI_API_KEY", "").strip()
        if raw in _PLACEHOLDER_KEYS:
            return ""
        return raw

    def has_api_key(self) -> bool:
        """Return True if a non-placeholder API key is configured."""
        return bool(self.get_api_key())

    def masked_api_key(self) -> str:
        """Return a masked representation safe for UI display."""
        key = self.get_api_key()
        if not key:
            return "Not configured"
        if len(key) <= 8:
            return "****"
        return f"{key[:4]}...{key[-4:]}"


def build_settings_from_env() -> Settings:
    """Construct a Settings instance from current environment variables."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    provider = os.getenv("LLM_PROVIDER", "GEMINI").strip().upper() or "GEMINI"
    if provider not in {"GEMINI"}:
        provider = "GEMINI"

    style = os.getenv("RESPONSE_STYLE", "Concise").strip()
    if style not in {"Concise", "Detailed", "Plain Language"}:
        style = "Concise"

    return Settings(
        llm_provider=provider,  # type: ignore[arg-type]
        llm_model_id=os.getenv("LLM_MODEL_ID", "gemini-2.5-flash").strip()
        or "gemini-2.5-flash",
        llm_api_version=os.getenv("GEMINI_API_VERSION", "v1beta").strip() or "v1beta",
        embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2").strip()
        or "all-MiniLM-L6-v2",
        embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu").strip() or "cpu",
        chroma_persist_dir=_resolve_path(
            os.getenv("CHROMA_PERSIST_DIR", ""), "data/chroma_db"
        ),
        chroma_collection_name=os.getenv(
            "CHROMA_COLLECTION_NAME", "medisage_chunks"
        ).strip()
        or "medisage_chunks",
        sqlite_db_path=_resolve_path(
            os.getenv("SQLITE_DB_PATH", ""), "data/medisage.db"
        ),
        upload_dir=_resolve_path(os.getenv("UPLOAD_DIR", ""), "data/uploads"),
        samples_dir=PROJECT_ROOT / "data" / "samples",
        max_upload_size_mb=int(os.getenv("MAX_UPLOAD_SIZE_MB", "20")),
        chunk_size_tokens=int(os.getenv("CHUNK_SIZE_TOKENS", "600")),
        chunk_overlap_tokens=int(os.getenv("CHUNK_OVERLAP_TOKENS", "100")),
        retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "4")),
        similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.32")),
        max_history_turns=int(os.getenv("MAX_HISTORY_TURNS", "4")),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        llm_max_output_tokens=int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1024")),
        llm_timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
        response_style=style,  # type: ignore[arg-type]
        log_level=os.getenv("LOG_LEVEL", "INFO").strip() or "INFO",
        log_dir=_resolve_path(os.getenv("LOG_DIR", ""), "logs"),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return build_settings_from_env()


def reload_settings() -> Settings:
    """Clear settings cache and re-read environment variables."""
    get_settings.cache_clear()
    return get_settings()


def ensure_directories(settings: Settings | None = None) -> Settings:
    """Ensure required runtime data and log directories exist."""
    cfg = settings or get_settings()
    for folder in (
        cfg.upload_dir,
        cfg.chroma_persist_dir,
        cfg.samples_dir,
        cfg.sqlite_db_path.parent,
        cfg.log_dir,
    ):
        folder.mkdir(parents=True, exist_ok=True)
    return cfg
