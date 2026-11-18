"""General-purpose helper functions for MediSage."""

from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from core.logging_utils import redact_sensitive_text

_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:\\[^\s\"']+")
_UNIX_PATH_RE = re.compile(r"(?:/(?:home|Users|var|tmp|opt|etc)/[^\s\"']+)")


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def compute_sha256_bytes(data: bytes) -> str:
    """Return SHA-256 hex digest of an in-memory byte buffer."""
    return hashlib.sha256(data).hexdigest()


def compute_sha256_file(path: Path) -> str:
    """Return SHA-256 hex digest of a file streamed in 64KB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def estimate_tokens(text: str) -> int:
    """Estimate token count on CPU without requiring an external tokenizer."""
    if not text or not text.strip():
        return 0
    word_est = int(math.ceil(len(text.split()) * 1.33))
    char_est = int(math.ceil(len(text) / 4.0))
    return max(1, max(word_est, char_est))


def format_file_size(num_bytes: int) -> str:
    """Format byte count into a human-readable string."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.2f} MB"


def sanitize_error_message(error: Exception | str) -> str:
    """Strip local filesystem paths and sensitive credentials from error messages."""
    raw = str(error) if isinstance(error, Exception) else error
    cleaned = redact_sensitive_text(raw)
    cleaned = _WINDOWS_PATH_RE.sub("[local-path]", cleaned)
    cleaned = _UNIX_PATH_RE.sub("[local-path]", cleaned)
    return cleaned.strip() or "An unexpected error occurred."


def normalize_filename_for_display(filename: str) -> str:
    """Return basename only without directory components."""
    return Path(filename).name
