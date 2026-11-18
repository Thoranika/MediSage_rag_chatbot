"""Cloud LLM provider adapter using the official Google GenAI SDK."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from config.settings import Settings, get_settings
from core.logging_utils import get_logger
from utils.helpers import sanitize_error_message

logger = get_logger("core.llm_client")

_TRANSIENT_KEYWORDS = (
    "429",
    "rate limit",
    "resource_exhausted",
    "quota",
    "timeout",
    "timed out",
    "deadline",
    "503",
    "unavailable",
    "connection reset",
    "connection aborted",
    "connecterror",
    "getaddrinfo",
    "temporarily unavailable",
)


@dataclass
class LLMResponse:
    """Normalized response from a cloud LLM provider."""

    success: bool
    text: str
    provider: str
    model_id: str
    error_type: str = ""
    error_message: str = ""
    latency_ms: float = 0.0
    attempts: int = 1


class BaseLLMClient:
    """Common provider-agnostic LLM client interface."""

    def generate(
        self, prompt: str, settings: Settings | dict[str, Any] | None = None
    ) -> LLMResponse:
        raise NotImplementedError


def _extract_setting(
    settings: Settings | dict[str, Any] | None,
    attr: str,
    default: Any,
) -> Any:
    if settings is None:
        return default
    if isinstance(settings, dict):
        return settings.get(attr, default)
    return getattr(settings, attr, default)


def _classify_error(exc: Exception) -> tuple[str, bool]:
    """Return (error_type, is_retryable) without exposing sensitive internals."""
    msg = str(exc).lower()
    if "429" in msg or "rate limit" in msg or "resource_exhausted" in msg or "quota" in msg:
        return "rate_limit", True
    if "timeout" in msg or "timed out" in msg or "deadline" in msg:
        return "timeout", True
    if (
        "connect" in msg
        or "getaddrinfo" in msg
        or "dns" in msg
        or "network" in msg
        or "ssl" in msg
        or "eof occurred" in msg
    ):
        return "network_error", True
    if (
        "404" in msg
        or "not_found" in msg
        or "model is not found" in msg
        or "is not supported for generatecontent" in msg
    ):
        return "unsupported_model", False
    if (
        "api key" in msg
        or "api_key" in msg
        or "401" in msg
        or "403" in msg
        or "permission" in msg
        or "unauthenticated" in msg
    ):
        return "auth_error", False
    if "developer instruction" in msg or "system instruction" in msg:
        return "developer_instruction_unsupported", False
    if "400" in msg or "invalid_argument" in msg:
        return "invalid_argument", False
    if any(k in msg for k in _TRANSIENT_KEYWORDS):
        return "api_error", True
    return "api_error", False


def _extract_text_from_genai_response(response: Any) -> str:
    """Safely extract text from a google-genai response object or handle malformed structures."""
    if response is None:
        raise ValueError("Received empty response object from Gemini API.")

    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    # Fallback inspection of candidates if .text helper is unavailable
    candidates = getattr(response, "candidates", None)
    if isinstance(candidates, list) and candidates:
        first = candidates[0]
        finish_reason = str(getattr(first, "finish_reason", "") or "").upper()
        if "SAFETY" in finish_reason:
            raise PermissionError("Response blocked by provider safety filters.")
        content = getattr(first, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        if isinstance(parts, list):
            collected = [
                str(getattr(p, "text", "")).strip()
                for p in parts
                if isinstance(getattr(p, "text", None), str) and getattr(p, "text", "").strip()
            ]
            if collected:
                return "\n".join(collected)
        if "MAX_TOKENS" in finish_reason:
            raise ValueError(
                "Gemini output hit max_output_tokens before producing visible text."
            )

    raise ValueError("Malformed or empty text content in Gemini API response.")


class GeminiLLMClient(BaseLLMClient):
    """Google Gemini API adapter using the official `google-genai` SDK (`v1beta`)."""

    def __init__(
        self,
        *,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.6,
        client_factory: Any = None,
    ) -> None:
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self._client_factory = client_factory

    def _build_client(
        self, api_key: str, timeout_seconds: int, api_version: str = "v1beta"
    ):
        if self._client_factory is not None:
            return self._client_factory(api_key=api_key, timeout_seconds=timeout_seconds)

        from google import genai
        from google.genai import types

        # `google-genai` defaults to `v1`, which rejects `system_instruction` ("Developer instruction").
        # Using `v1beta` enables documented system instructions and Gemini 2.5 Flash configuration.
        http_options = types.HttpOptions(
            api_version=api_version or "v1beta",
            timeout=int(timeout_seconds * 1000),
        )
        return genai.Client(api_key=api_key, http_options=http_options)

    @staticmethod
    def _build_gen_config(
        *,
        model_id: str,
        temperature: float,
        max_tokens: int,
        system_instruction: str | None,
    ):
        from google.genai import types

        kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system_instruction:
            kwargs["system_instruction"] = system_instruction

        # Disable hidden thinking-token overhead on Gemini 2.5 Flash so tokens go to the grounded reply
        if "2.5-flash" in model_id.lower() and hasattr(types, "ThinkingConfig"):
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

        return types.GenerateContentConfig(**kwargs)

    def generate(
        self, prompt: str, settings: Settings | dict[str, Any] | None = None
    ) -> LLMResponse:
        cfg = get_settings()
        provider = str(
            _extract_setting(settings, "llm_provider", cfg.llm_provider)
        ).upper()
        model_id = str(
            _extract_setting(settings, "llm_model_id", cfg.llm_model_id)
        ).strip()
        api_version = str(
            _extract_setting(settings, "llm_api_version", cfg.llm_api_version)
        ).strip() or "v1beta"
        temperature = float(
            _extract_setting(settings, "llm_temperature", cfg.llm_temperature)
        )
        max_tokens = int(
            _extract_setting(
                settings, "llm_max_output_tokens", cfg.llm_max_output_tokens
            )
        )
        timeout_seconds = int(
            _extract_setting(settings, "llm_timeout_seconds", cfg.llm_timeout_seconds)
        )
        system_instruction = _extract_setting(settings, "system_instruction", None)

        api_key = ""
        if isinstance(settings, dict) and settings.get("api_key"):
            api_key = str(settings["api_key"]).strip()
        elif isinstance(settings, Settings):
            api_key = settings.get_api_key()
        else:
            api_key = cfg.get_api_key()

        if provider != "GEMINI":
            return LLMResponse(
                success=False,
                text="",
                provider=provider,
                model_id=model_id,
                error_type="unsupported_provider",
                error_message=f"Unsupported LLM provider '{provider}'. Configure LLM_PROVIDER=GEMINI.",
            )

        if not api_key:
            return LLMResponse(
                success=False,
                text="",
                provider=provider,
                model_id=model_id,
                error_type="missing_api_key",
                error_message=(
                    "GEMINI_API_KEY is not configured. Add it to your .env file to enable "
                    "cloud Gemini generation."
                ),
            )

        if not prompt or not prompt.strip():
            return LLMResponse(
                success=False,
                text="",
                provider=provider,
                model_id=model_id,
                error_type="invalid_prompt",
                error_message="Cannot generate a response for an empty prompt.",
            )

        t0 = time.perf_counter()
        last_error_type = "api_error"
        last_error_msg = "Unknown API error."
        attempts = 0
        active_prompt = prompt
        active_system_instruction = system_instruction
        folded_instruction = False

        for attempt in range(1, self.max_retries + 2):
            attempts = attempt
            try:
                client = self._build_client(api_key, timeout_seconds, api_version)
                gen_config = self._build_gen_config(
                    model_id=model_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_instruction=active_system_instruction,
                )
                raw_response = client.models.generate_content(
                    model=model_id,
                    contents=active_prompt,
                    config=gen_config,
                )
                text = _extract_text_from_genai_response(raw_response)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                logger.info(
                    "LLM generation succeeded (provider=%s, model=%s, api_version=%s, attempts=%d, latency=%.1fms)",
                    provider,
                    model_id,
                    api_version,
                    attempts,
                    elapsed_ms,
                )
                return LLMResponse(
                    success=True,
                    text=text,
                    provider=provider,
                    model_id=model_id,
                    latency_ms=elapsed_ms,
                    attempts=attempts,
                )
            except PermissionError as exc:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return LLMResponse(
                    success=False,
                    text="",
                    provider=provider,
                    model_id=model_id,
                    error_type="safety_blocked",
                    error_message=sanitize_error_message(exc),
                    latency_ms=elapsed_ms,
                    attempts=attempts,
                )
            except ValueError as exc:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return LLMResponse(
                    success=False,
                    text="",
                    provider=provider,
                    model_id=model_id,
                    error_type="malformed_response",
                    error_message=sanitize_error_message(exc),
                    latency_ms=elapsed_ms,
                    attempts=attempts,
                )
            except Exception as exc:
                err_type, retryable = _classify_error(exc)
                last_error_type = err_type
                last_error_msg = sanitize_error_message(exc)
                logger.warning(
                    "LLM call attempt %d failed (%s): %s",
                    attempt,
                    err_type,
                    last_error_msg,
                )
                # Safety fallback if an endpoint/model rejects separate system_instruction
                if (
                    err_type == "developer_instruction_unsupported"
                    and active_system_instruction
                    and not folded_instruction
                ):
                    folded_instruction = True
                    active_prompt = (
                        f"### System Instructions\n{active_system_instruction}\n\n"
                        f"{prompt}"
                    )
                    active_system_instruction = None
                    continue

                if retryable and attempt <= self.max_retries:
                    if self.retry_backoff_seconds > 0:
                        time.sleep(self.retry_backoff_seconds * attempt)
                    continue
                break

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        friendly_messages = {
            "rate_limit": "Cloud LLM rate limit or quota exceeded. Please wait a moment and try again.",
            "timeout": "Cloud LLM request timed out before completing. Please retry.",
            "network_error": "Network connection to the Gemini API failed. Check internet connectivity and retry.",
            "unsupported_model": f"Configured Gemini model '{model_id}' is unsupported or unavailable. Select a valid model in Settings.",
            "auth_error": "Cloud LLM authentication failed. Verify your GEMINI_API_KEY in .env.",
        }
        return LLMResponse(
            success=False,
            text="",
            provider=provider,
            model_id=model_id,
            error_type=last_error_type,
            error_message=friendly_messages.get(last_error_type, last_error_msg),
            latency_ms=elapsed_ms,
            attempts=attempts,
        )


def generate(
    prompt: str, settings: Settings | dict[str, Any] | None = None
) -> LLMResponse:
    """Module-level convenience `generate(prompt, settings)` interface required by Task 7."""
    client = GeminiLLMClient()
    return client.generate(prompt, settings)


def test_llm_connection(settings: Settings | dict[str, Any] | None = None) -> LLMResponse:
    """Execute a minimal health-check prompt against the configured cloud LLM."""
    cfg_settings: dict[str, Any] = {
        "llm_max_output_tokens": 128,
        "llm_temperature": 0.0,
        "system_instruction": "You are a connectivity check endpoint. Reply concisely.",
    }
    if isinstance(settings, Settings):
        cfg_settings.update(
            {
                "llm_provider": settings.llm_provider,
                "llm_model_id": settings.llm_model_id,
                "llm_api_version": settings.llm_api_version,
                "llm_timeout_seconds": settings.llm_timeout_seconds,
                "api_key": settings.get_api_key(),
            }
        )
    elif isinstance(settings, dict):
        cfg_settings.update(settings)
    return generate("Reply with exactly: OK", cfg_settings)
