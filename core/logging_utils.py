"""Structured logging utilities with secret and sensitive-content redaction."""

from __future__ import annotations

import logging
import re
from pathlib import Path

_API_KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z\-_]{20,}")
_BEARER_PATTERN = re.compile(r"(Bearer\s+)[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE)
_ENV_KEY_PATTERN = re.compile(
    r"(GEMINI_API_KEY|GOOGLE_API_KEY|API_KEY)\s*[:=]\s*([^\s,;\"']+)",
    re.IGNORECASE,
)


def redact_sensitive_text(text: str) -> str:
    """Mask API keys, bearer tokens, and long payloads in log strings."""
    if not text:
        return text
    masked = _API_KEY_PATTERN.sub("[REDACTED_API_KEY]", text)
    masked = _BEARER_PATTERN.sub(r"\1[REDACTED_TOKEN]", masked)
    masked = _ENV_KEY_PATTERN.sub(r"\1=[REDACTED]", masked)
    return masked


class SensitiveDataFilter(logging.Filter):
    """Logging filter that strips secrets from formatted log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            rendered = record.getMessage()
            redacted = redact_sensitive_text(rendered)
            if redacted != rendered:
                record.msg = redacted
                record.args = ()
        except Exception:
            pass
        return True


def configure_logging(log_level: str = "INFO", log_dir: Path | None = None) -> None:
    """Configure application-wide structured logging."""
    root_logger = logging.getLogger("medisage")
    if root_logger.handlers:
        root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        return

    level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(level)
    root_logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    redactor = SensitiveDataFilter()

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(redactor)
    root_logger.addHandler(stream_handler)

    if log_dir is not None:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(
                log_dir / "medisage.log", encoding="utf-8"
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            file_handler.addFilter(redactor)
            root_logger.addHandler(file_handler)
        except OSError:
            # Continue with stream logging if file log cannot be opened
            pass


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the `medisage` namespace."""
    configure_logging()
    clean_name = name if name.startswith("medisage") else f"medisage.{name}"
    return logging.getLogger(clean_name)
