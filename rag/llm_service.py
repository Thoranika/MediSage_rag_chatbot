"""LLM service wrapper integrating core.llm_client into the RAG pipeline."""

from __future__ import annotations

from typing import Any

from config.settings import Settings, get_settings
from core.llm_client import BaseLLMClient, GeminiLLMClient, LLMResponse


class LLMService:
    """Reusable RAG-layer LLM service wrapping a configurable BaseLLMClient."""

    def __init__(
        self,
        client: BaseLLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or GeminiLLMClient()

    def generate_answer(
        self,
        *,
        prompt: str,
        system_instruction: str,
        overrides: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Invoke the configured cloud LLM with controlled medical-grounding settings."""
        payload: dict[str, Any] = {
            "llm_provider": self.settings.llm_provider,
            "llm_model_id": self.settings.llm_model_id,
            "llm_api_version": self.settings.llm_api_version,
            "llm_temperature": self.settings.llm_temperature,
            "llm_max_output_tokens": self.settings.llm_max_output_tokens,
            "llm_timeout_seconds": self.settings.llm_timeout_seconds,
            "system_instruction": system_instruction,
            "api_key": self.settings.get_api_key(),
        }
        if overrides:
            payload.update(overrides)
        return self.client.generate(prompt, payload)
