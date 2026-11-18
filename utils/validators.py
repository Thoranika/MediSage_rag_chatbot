"""Security-focused input, file upload, path traversal, and settings validation."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

ALLOWED_EXTENSIONS: set[str] = {".pdf", ".docx", ".txt", ".md"}

ALLOWED_MIME_MAP: dict[str, set[str]] = {
    ".pdf": {"application/pdf", "application/x-pdf", "application/octet-stream"},
    ".docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/zip",
        "application/octet-stream",
    },
    ".txt": {"text/plain", "application/octet-stream"},
    ".md": {"text/markdown", "text/x-markdown", "text/plain", "application/octet-stream"},
}

SUPPORTED_MODELS: list[str] = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
]

SUPPORTED_RESPONSE_STYLES: list[str] = ["Concise", "Detailed", "Plain Language"]


@dataclass
class FileValidationResult:
    """Result of validating an uploaded document."""

    is_valid: bool
    original_filename: str
    extension: str
    size_bytes: int
    error_message: str = ""


def sanitize_original_filename(filename: str) -> str:
    """Extract a clean basename without path traversal characters."""
    if not filename:
        return ""
    cleaned = filename.replace("\\", "/").split("/")[-1].strip()
    cleaned = cleaned.replace("\x00", "")
    return cleaned


def validate_uploaded_file(
    filename: str,
    content: bytes,
    mime_type: str | None = None,
    max_size_mb: int = 20,
) -> FileValidationResult:
    """Validate uploaded file extension, MIME type, size, magic bytes, and integrity."""
    safe_name = sanitize_original_filename(filename)
    if not safe_name or safe_name in {".", ".."} or ".." in filename.replace("\\", "/").split("/"):
        return FileValidationResult(
            is_valid=False,
            original_filename=safe_name or "unknown",
            extension="",
            size_bytes=len(content) if content else 0,
            error_message="Invalid filename or unsafe path characters detected.",
        )

    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return FileValidationResult(
            is_valid=False,
            original_filename=safe_name,
            extension=ext,
            size_bytes=len(content) if content else 0,
            error_message=(
                f"Unsupported file type '{ext or 'none'}'. "
                f"Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
            ),
        )

    if not content or len(content) == 0:
        return FileValidationResult(
            is_valid=False,
            original_filename=safe_name,
            extension=ext,
            size_bytes=0,
            error_message="The uploaded file is empty (0 bytes).",
        )

    max_bytes = max_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        return FileValidationResult(
            is_valid=False,
            original_filename=safe_name,
            extension=ext,
            size_bytes=len(content),
            error_message=f"File exceeds the maximum allowed size of {max_size_mb} MB.",
        )

    if mime_type:
        norm_mime = mime_type.split(";")[0].strip().lower()
        if norm_mime and norm_mime not in ALLOWED_MIME_MAP[ext]:
            return FileValidationResult(
                is_valid=False,
                original_filename=safe_name,
                extension=ext,
                size_bytes=len(content),
                error_message=f"MIME type '{norm_mime}' does not match '{ext}' files.",
            )

    # Magic-byte & container validation
    if ext == ".pdf":
        head = content[:1024].lstrip()
        if not head.startswith(b"%PDF-"):
            return FileValidationResult(
                is_valid=False,
                original_filename=safe_name,
                extension=ext,
                size_bytes=len(content),
                error_message="Invalid or corrupt PDF header (missing %PDF signature).",
            )
    elif ext == ".docx":
        if not content.startswith(b"PK\x03\x04"):
            return FileValidationResult(
                is_valid=False,
                original_filename=safe_name,
                extension=ext,
                size_bytes=len(content),
                error_message="Invalid or corrupt DOCX archive signature.",
            )
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                names = set(zf.namelist())
                if "[Content_Types].xml" not in names and "word/document.xml" not in names:
                    return FileValidationResult(
                        is_valid=False,
                        original_filename=safe_name,
                        extension=ext,
                        size_bytes=len(content),
                        error_message="File is a ZIP archive but not a valid DOCX document.",
                    )
        except zipfile.BadZipFile:
            return FileValidationResult(
                is_valid=False,
                original_filename=safe_name,
                extension=ext,
                size_bytes=len(content),
                error_message="Corrupt DOCX file: unable to read ZIP structure.",
            )
    else:
        # TXT / MD text validation
        if b"\x00" in content[:4096]:
            return FileValidationResult(
                is_valid=False,
                original_filename=safe_name,
                extension=ext,
                size_bytes=len(content),
                error_message="Binary null bytes detected in text/markdown upload.",
            )
        try:
            text_sample = content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text_sample = content.decode("latin-1")
            except UnicodeDecodeError:
                return FileValidationResult(
                    is_valid=False,
                    original_filename=safe_name,
                    extension=ext,
                    size_bytes=len(content),
                    error_message="Unable to decode text file using UTF-8 or Latin-1.",
                )
        if not text_sample.strip():
            return FileValidationResult(
                is_valid=False,
                original_filename=safe_name,
                extension=ext,
                size_bytes=len(content),
                error_message="The text/markdown document contains only whitespace.",
            )

    return FileValidationResult(
        is_valid=True,
        original_filename=safe_name,
        extension=ext,
        size_bytes=len(content),
    )


def build_safe_storage_path(
    upload_dir: Path, original_filename: str, file_hash: str
) -> tuple[str, Path]:
    """Generate a traversal-safe UUID-prefixed storage filename inside upload_dir."""
    upload_dir_resolved = upload_dir.resolve()
    upload_dir_resolved.mkdir(parents=True, exist_ok=True)
    ext = Path(sanitize_original_filename(original_filename)).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported extension for storage: {ext}")

    safe_stored_name = f"doc_{uuid4().hex[:12]}_{file_hash[:8]}{ext}"
    candidate = (upload_dir_resolved / safe_stored_name).resolve()
    if not candidate.is_relative_to(upload_dir_resolved):
        raise ValueError("Unsafe storage path outside upload directory.")
    return safe_stored_name, candidate


def resolve_safe_stored_file(upload_dir: Path, stored_filename: str) -> Path:
    """Resolve an existing stored filename safely inside upload_dir."""
    if not stored_filename or "/" in stored_filename or "\\" in stored_filename or ".." in stored_filename:
        raise ValueError("Invalid stored filename.")
    upload_dir_resolved = upload_dir.resolve()
    target = (upload_dir_resolved / stored_filename).resolve()
    if not target.is_relative_to(upload_dir_resolved):
        raise ValueError("Path traversal attempt blocked.")
    return target


def validate_settings_payload(payload: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    """Validate user-configurable settings from the Settings UI."""
    errors: list[str] = []
    cleaned: dict[str, Any] = {}

    provider = str(payload.get("llm_provider", "GEMINI")).strip().upper()
    if provider != "GEMINI":
        errors.append("Only GEMINI provider is supported in this build.")
    cleaned["llm_provider"] = provider

    model_id = str(payload.get("llm_model_id", "gemini-2.5-flash")).strip()
    if not model_id or len(model_id) > 80 or any(ch.isspace() for ch in model_id):
        errors.append("Model ID must be a non-empty identifier without spaces.")
    cleaned["llm_model_id"] = model_id

    try:
        top_k = int(payload.get("retrieval_top_k", 4))
        if not 1 <= top_k <= 12:
            errors.append("Top-K retrieval must be between 1 and 12.")
        cleaned["retrieval_top_k"] = top_k
    except (TypeError, ValueError):
        errors.append("Top-K retrieval must be an integer.")

    try:
        threshold = float(payload.get("similarity_threshold", 0.32))
        if not 0.0 <= threshold <= 0.95:
            errors.append("Similarity threshold must be between 0.0 and 0.95.")
        cleaned["similarity_threshold"] = round(threshold, 3)
    except (TypeError, ValueError):
        errors.append("Similarity threshold must be a decimal number.")

    try:
        chunk_size = int(payload.get("chunk_size_tokens", 600))
        if not 150 <= chunk_size <= 1500:
            errors.append("Chunk size must be between 150 and 1500 tokens.")
        cleaned["chunk_size_tokens"] = chunk_size
    except (TypeError, ValueError):
        errors.append("Chunk size must be an integer.")
        chunk_size = 600

    try:
        chunk_overlap = int(payload.get("chunk_overlap_tokens", 100))
        if not 0 <= chunk_overlap <= 400:
            errors.append("Chunk overlap must be between 0 and 400 tokens.")
        elif chunk_overlap >= chunk_size:
            errors.append("Chunk overlap must be strictly less than chunk size.")
        cleaned["chunk_overlap_tokens"] = chunk_overlap
    except (TypeError, ValueError):
        errors.append("Chunk overlap must be an integer.")

    try:
        temperature = float(payload.get("llm_temperature", 0.2))
        if not 0.0 <= temperature <= 1.0:
            errors.append("Temperature must be between 0.0 and 1.0.")
        cleaned["llm_temperature"] = round(temperature, 2)
    except (TypeError, ValueError):
        errors.append("Temperature must be a decimal number.")

    try:
        max_tokens = int(payload.get("llm_max_output_tokens", 1024))
        if not 128 <= max_tokens <= 4096:
            errors.append("Max output tokens must be between 128 and 4096.")
        cleaned["llm_max_output_tokens"] = max_tokens
    except (TypeError, ValueError):
        errors.append("Max output tokens must be an integer.")

    style = str(payload.get("response_style", "Concise")).strip()
    if style not in SUPPORTED_RESPONSE_STYLES:
        errors.append(
            f"Response style must be one of: {', '.join(SUPPORTED_RESPONSE_STYLES)}."
        )
    cleaned["response_style"] = style

    return len(errors) == 0, errors, cleaned
