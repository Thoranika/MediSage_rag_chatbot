"""Tests for Cloud LLM adapter (mocked), citation validation, medical safety, and E2E RAG workflow."""

from __future__ import annotations

import logging
from types import SimpleNamespace

from chatbot.chat_manager import ChatManager
from core.llm_client import GeminiLLMClient
from core.logging_utils import SensitiveDataFilter, redact_sensitive_text
from ingestion.ingestion_service import IngestionService
from rag.citations import validate_and_extract_citations
from rag.llm_service import LLMService
from rag.query_rewriter import rewrite_followup_query
from rag.rag_pipeline import RAGPipeline
from rag.retriever import RetrievedChunk, SemanticRetriever
from rag.safety import assess_query_safety
from tests.conftest import MockGroundedLLM


def test_llm_client_missing_key_error(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = GeminiLLMClient(max_retries=0, retry_backoff_seconds=0.0)
    resp = client.generate("Hello", {"llm_provider": "GEMINI", "api_key": ""})
    assert resp.success is False
    assert resp.error_type == "missing_api_key"


def test_llm_client_rate_limit_retry_and_timeout_handling() -> None:
    attempts = {"count": 0}

    class _RateLimitThenSuccessModels:
        def generate_content(self, **kwargs):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("429 RESOURCE_EXHAUSTED rate limit exceeded")
            return SimpleNamespace(text="Grounded reply [Source: guide.pdf, page 1]")

    client = GeminiLLMClient(
        max_retries=2,
        retry_backoff_seconds=0.0,
        client_factory=lambda api_key, timeout_seconds: SimpleNamespace(
            models=_RateLimitThenSuccessModels()
        ),
    )
    resp = client.generate("Test prompt", {"api_key": "fake-test-key"})
    assert resp.success is True
    assert resp.attempts == 2

    # Persistent timeout returns clean failure
    class _TimeoutModels:
        def generate_content(self, **kwargs):
            raise TimeoutError("Request timed out after 30s")

    timeout_client = GeminiLLMClient(
        max_retries=1,
        retry_backoff_seconds=0.0,
        client_factory=lambda api_key, timeout_seconds: SimpleNamespace(
            models=_TimeoutModels()
        ),
    )
    t_resp = timeout_client.generate("Test prompt", {"api_key": "fake-test-key"})
    assert t_resp.success is False
    assert t_resp.error_type == "timeout"

    # Malformed / empty response handling
    class _EmptyModels:
        def generate_content(self, **kwargs):
            return SimpleNamespace(text="", candidates=[])

    empty_client = GeminiLLMClient(
        max_retries=0,
        retry_backoff_seconds=0.0,
        client_factory=lambda api_key, timeout_seconds: SimpleNamespace(
            models=_EmptyModels()
        ),
    )
    e_resp = empty_client.generate("Test prompt", {"api_key": "fake-test-key"})
    assert e_resp.success is False
    assert e_resp.error_type == "malformed_response"


def test_sensitive_key_redaction_in_logs() -> None:
    raw = "Error calling GEMINI_API_KEY=AIzaSyDummyKey1234567890abcdefg"
    redacted = redact_sensitive_text(raw)
    assert "AIzaSy" not in redacted
    assert "REDACTED" in redacted

    record = logging.LogRecord(
        "medisage.test", logging.INFO, __file__, 1, "Key is AIzaSySecretToken1234567890xyz", (), None
    )
    SensitiveDataFilter().filter(record)
    assert "AIzaSy" not in record.getMessage()


def test_citation_validation_strips_fabricated_sources() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            document_id="d1",
            filename="WHO_Hypertension_Clinical_Guideline.pdf",
            page_number=3,
            section_title="Pregnancy",
            text="ACE inhibitors and ARBs are contraindicated in pregnancy.",
            similarity=0.82,
            distance=0.18,
        )
    ]
    model_answer = (
        "ACE inhibitors are contraindicated in pregnancy "
        "[Source: WHO_Hypertension_Clinical_Guideline.pdf, page 3]. "
        "Another fake claim [Source: Hallucinated_Manual.pdf, page 99]."
    )
    report = validate_and_extract_citations(model_answer, chunks)
    assert report.all_citations_valid is False
    assert len(report.invalid_citations_removed) == 1
    assert "Hallucinated_Manual.pdf" in report.invalid_citations_removed[0]
    assert "Hallucinated_Manual.pdf" not in report.cleaned_answer
    assert len(report.citations) == 1
    assert report.citations[0].page_number == 3


def test_safety_guardrails_emergency_diagnosis_and_dosage() -> None:
    emerg = assess_query_safety("I have crushing chest pain and difficulty breathing right now.")
    assert emerg.is_emergency is True
    assert "emergency" in emerg.override_response.lower()

    diag = assess_query_safety("Can you diagnose me with Stage 2 hypertension?")
    assert diag.requires_safety_boundary is True
    assert "personal_diagnosis_request" in diag.flags

    dose = assess_query_safety("What dose should I take of lisinopril or should I stop taking it?")
    assert dose.requires_safety_boundary is True
    assert "personalized_dosage_or_cessation_request" in dose.flags


def test_followup_query_rewriter_preserves_original_question() -> None:
    history = [
        {"role": "user", "content": "When is metformin used in Type 2 diabetes?"},
        {"role": "assistant", "content": "Metformin is first-line unless eGFR < 30."},
    ]
    rewritten, changed = rewrite_followup_query(
        "What should be monitored with it long term?", history
    )
    assert changed is True
    assert "metformin" in rewritten.lower()


def test_end_to_end_rag_and_chat_persistence(
    isolated_env, shared_embedding_service, sample_pdf_bytes: bytes
) -> None:
    cfg, db, chroma = isolated_env
    ing = IngestionService(
        db=db, chroma=chroma, embeddings=shared_embedding_service, settings=cfg
    )
    ing.ingest_bytes(filename="hypertension.pdf", content=sample_pdf_bytes)

    retriever = SemanticRetriever(chroma=chroma, embeddings=shared_embedding_service)
    mock_llm = MockGroundedLLM(
        "ACE inhibitors and ARBs are strictly contraindicated in pregnancy "
        "[Source: hypertension.pdf, page 2]."
    )
    pipeline = RAGPipeline(
        retriever=retriever,
        llm_service=LLMService(client=mock_llm, settings=cfg),
        db=db,
        settings=cfg,
    )
    chat = ChatManager(db=db, rag_pipeline=pipeline)
    conv = chat.create_conversation("Pregnancy Meds")

    _, a_msg, resp = chat.ask(
        conversation_id=conv["conversation_id"],
        user_question="Which blood pressure drugs are contraindicated during pregnancy?",
    )
    assert resp.llm_used is True
    assert resp.retrieval_status in {"ok", "low_confidence"}
    assert len(resp.citations) == 1
    assert resp.citations[0].filename == "hypertension.pdf"
    assert resp.citations[0].page_number == 2
    assert "Educational reference only" in a_msg["content"]

    # Verify persistence across manager re-instantiation
    reloaded_msgs = db.get_messages(conv["conversation_id"])
    assert len(reloaded_msgs) == 2
    assert reloaded_msgs[1]["citations"][0]["page_number"] == 2

    # Emergency prompt never calls LLM and blocks safely
    calls_before = len(mock_llm.calls)
    _, _, emerg_resp = chat.ask(
        conversation_id=conv["conversation_id"],
        user_question="I have chest pain and cannot breathe",
    )
    assert emerg_resp.retrieval_status == "safety_blocked"
    assert len(mock_llm.calls) == calls_before


def test_llm_client_v1beta_default_and_developer_instruction_fallback() -> None:
    real_client = GeminiLLMClient()
    built = real_client._build_client("dummy-key", 15, "v1beta")
    assert built._api_client._http_options.api_version == "v1beta"

    seen_configs = []

    class _DevInstructionFallbackModels:
        def generate_content(self, **kwargs):
            seen_configs.append(kwargs)
            if getattr(kwargs.get("config"), "system_instruction", None):
                raise RuntimeError(
                    "400 INVALID_ARGUMENT: Developer instruction is not enabled for api version v1."
                )
            return SimpleNamespace(text="Fallback reply preserving instructions.")

    client = GeminiLLMClient(
        max_retries=1,
        retry_backoff_seconds=0.0,
        client_factory=lambda api_key, timeout_seconds: SimpleNamespace(
            models=_DevInstructionFallbackModels()
        ),
    )
    resp = client.generate(
        "What is Stage 1 BP?",
        {"api_key": "fake-key", "system_instruction": "Answer only from context."},
    )
    assert resp.success is True
    assert resp.attempts == 2
    assert "System Instructions" in seen_configs[1]["contents"]


def test_abstention_response_suppresses_unrelated_citations() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            document_id="d1",
            filename="WHO_Hypertension_Clinical_Guideline.pdf",
            page_number=1,
            section_title="BP",
            text="Stage 1 hypertension is 130-139/80-89 mmHg.",
            similarity=0.35,
            distance=0.65,
        )
    ]
    abstention = (
        "I could not find sufficient supporting information in your uploaded medical "
        "reference documents. Please consult an authoritative medical source or qualified healthcare professional."
    )
    report = validate_and_extract_citations(abstention, chunks)
    assert report.citations == []


def test_grouped_semicolon_citations_are_validated_and_split() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            document_id="d1",
            filename="DocA.pdf",
            page_number=1,
            section_title="BP",
            text="Stage 1 hypertension is 130-139/80-89 mmHg.",
            similarity=0.80,
            distance=0.20,
        ),
        RetrievedChunk(
            chunk_id="c2",
            document_id="d2",
            filename="DocB.docx",
            page_number=2,
            section_title="BP",
            text="Stage 2 hypertension is >= 140/90 mmHg.",
            similarity=0.75,
            distance=0.25,
        ),
    ]
    ans = "Hypertension stages [Source: DocA.pdf, page 1; Source: DocB.docx, page 2; Source: Fake.pdf, page 9]."
    report = validate_and_extract_citations(ans, chunks)
    assert report.valid_citation_count == 2
    assert len(report.citations) == 2
    assert len(report.invalid_citations_removed) == 1
    assert "Fake.pdf" not in report.cleaned_answer
    assert "[Source: DocA.pdf, page 1] [Source: DocB.docx, page 2]" in report.cleaned_answer
