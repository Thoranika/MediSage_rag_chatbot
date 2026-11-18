"""Complete RAG pipeline: safety check -> query rewrite -> retrieval -> prompt -> LLM -> citation validation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from config.settings import Settings, get_settings
from core.logging_utils import get_logger
from database.db import DatabaseManager
from rag.citations import (
    VerifiedCitation,
    is_abstention_response,
    validate_and_extract_citations,
)
from rag.llm_service import LLMService
from rag.prompt_builder import build_grounded_prompt
from rag.query_rewriter import rewrite_followup_query
from rag.retriever import RetrievalResult, RetrievedChunk, SemanticRetriever
from rag.safety import (
    EDUCATIONAL_DISCLAIMER,
    SafetyAssessment,
    assess_query_safety,
    build_safety_footer,
)

logger = get_logger("rag.pipeline")


@dataclass
class RAGResponse:
    """End-to-end result produced by RAGPipeline.answer_question."""

    question: str
    rewritten_query: str
    was_rewritten: bool
    answer: str
    retrieval_status: str  # "ok", "low_confidence", "no_evidence", "safety_blocked", "llm_error"
    citations: list[VerifiedCitation] = field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    safety_flags: list[str] = field(default_factory=list)
    invalid_citations_removed: list[str] = field(default_factory=list)
    llm_used: bool = False
    llm_provider: str = ""
    llm_model_id: str = ""
    llm_error_type: str = ""
    latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    disclaimer: str = EDUCATIONAL_DISCLAIMER


class RAGPipeline:
    """Orchestrates evidence-grounded medical question answering."""

    def __init__(
        self,
        *,
        retriever: SemanticRetriever,
        llm_service: LLMService,
        db: DatabaseManager | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm_service = llm_service
        self.db = db
        self.settings = settings or get_settings()

    def _effective_runtime_params(
        self, overrides: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        params = {
            "llm_provider": self.settings.llm_provider,
            "llm_model_id": self.settings.llm_model_id,
            "llm_api_version": self.settings.llm_api_version,
            "retrieval_top_k": self.settings.retrieval_top_k,
            "similarity_threshold": self.settings.similarity_threshold,
            "llm_temperature": self.settings.llm_temperature,
            "llm_max_output_tokens": self.settings.llm_max_output_tokens,
            "response_style": self.settings.response_style,
            "max_history_turns": self.settings.max_history_turns,
        }
        if self.db is not None:
            persisted = self.db.get_all_settings()
            for k in params:
                if k in persisted:
                    params[k] = persisted[k]
        if overrides:
            params.update(overrides)
        return params

    def answer_question(
        self,
        question: str,
        *,
        recent_messages: list[dict[str, Any]] | None = None,
        metadata_filters: dict[str, Any] | None = None,
        setting_overrides: dict[str, Any] | None = None,
    ) -> RAGResponse:
        """Execute the full RAG workflow for a user question."""
        t0 = time.perf_counter()
        clean_q = (question or "").strip()
        params = self._effective_runtime_params(setting_overrides)

        if not clean_q:
            return RAGResponse(
                question="",
                rewritten_query="",
                was_rewritten=False,
                answer="Please enter a medical reference question.",
                retrieval_status="no_evidence",
                latency_ms=0.0,
            )

        # 1. Medical safety & emergency triage gate
        safety: SafetyAssessment = assess_query_safety(clean_q)
        if safety.is_emergency:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return RAGResponse(
                question=clean_q,
                rewritten_query=clean_q,
                was_rewritten=False,
                answer=f"{safety.override_response}\n\n{build_safety_footer()}",
                retrieval_status="safety_blocked",
                safety_flags=safety.flags,
                latency_ms=round(elapsed, 2),
            )

        # 2. Bounded follow-up query rewriting (for retrieval only; displayed question stays unchanged)
        max_msgs = int(params["max_history_turns"]) * 2
        bounded_history = (recent_messages or [])[-max_msgs:]
        standalone_query, was_rewritten = rewrite_followup_query(
            clean_q, bounded_history
        )

        # 3. Always execute fresh semantic retrieval for factual queries
        retrieval: RetrievalResult = self.retriever.retrieve(
            standalone_query,
            top_k=int(params["retrieval_top_k"]),
            similarity_threshold=float(params["similarity_threshold"]),
            metadata_filters=metadata_filters,
        )

        # 4. Handle empty / insufficient retrieval honestly without guessing
        if retrieval.status == "no_evidence" or not retrieval.chunks:
            elapsed = (time.perf_counter() - t0) * 1000.0
            no_ev_lines = [
                "I could not find sufficient supporting information in your uploaded medical reference documents. "
                "Please consult an authoritative medical source or qualified healthcare professional.",
                "",
                "Suggestions:",
                "- Upload an authoritative medical guideline PDF or DOCX covering this topic in **Knowledge Library**.",
                "- Rephrase using specific clinical terminology or check document filters if enabled.",
            ]
            if safety.requires_safety_boundary:
                no_ev_lines.insert(
                    0,
                    "**Clinical Safety Notice:** MediSage cannot diagnose conditions, prescribe medications, "
                    "or recommend personalized dosages/treatment changes. Please consult a qualified healthcare professional.",
                )
                no_ev_lines.insert(1, "")

            return RAGResponse(
                question=clean_q,
                rewritten_query=standalone_query,
                was_rewritten=was_rewritten,
                answer="\n".join(no_ev_lines) + f"\n\n{build_safety_footer()}",
                retrieval_status="no_evidence",
                retrieved_chunks=[],
                safety_flags=safety.flags,
                latency_ms=round(elapsed, 2),
                retrieval_latency_ms=retrieval.latency_ms,
            )

        # 5. Construct grounded prompt and call Cloud LLM
        system_instruction, user_prompt = build_grounded_prompt(
            user_question=clean_q,
            retrieved_chunks=retrieval.chunks,
            recent_messages=bounded_history,
            safety=safety,
            response_style=str(params["response_style"]),
        )

        llm_resp = self.llm_service.generate_answer(
            prompt=user_prompt,
            system_instruction=system_instruction,
            overrides={
                "llm_provider": params["llm_provider"],
                "llm_model_id": params["llm_model_id"],
                "llm_api_version": params["llm_api_version"],
                "llm_temperature": params["llm_temperature"],
                "llm_max_output_tokens": params["llm_max_output_tokens"],
            },
        )

        if not llm_resp.success:
            elapsed = (time.perf_counter() - t0) * 1000.0
            # Present transparent retrieval excerpts while explicitly reporting LLM error/config gap
            citations = [
                VerifiedCitation(
                    document_id=ch.document_id,
                    filename=ch.filename,
                    page_number=ch.page_number,
                    section_title=ch.section_title,
                    similarity_score=ch.similarity,
                    excerpt=" ".join(ch.text.split())[:360],
                    verified=True,
                )
                for ch in retrieval.chunks
            ]
            msg = (
                f"**Cloud LLM Unavailable ({llm_resp.error_type}):** {llm_resp.error_message}\n\n"
                f"However, semantic retrieval succeeded and found **{len(retrieval.chunks)} relevant passage(s)** "
                "below in the Sources panel for your direct inspection."
            )
            return RAGResponse(
                question=clean_q,
                rewritten_query=standalone_query,
                was_rewritten=was_rewritten,
                answer=f"{msg}\n\n{build_safety_footer()}",
                retrieval_status="llm_error",
                citations=citations,
                retrieved_chunks=retrieval.chunks,
                safety_flags=safety.flags,
                llm_used=False,
                llm_provider=llm_resp.provider,
                llm_model_id=llm_resp.model_id,
                llm_error_type=llm_resp.error_type,
                latency_ms=round(elapsed, 2),
                retrieval_latency_ms=retrieval.latency_ms,
            )

        # 6. Validate every citation against actual retrieved chunks
        cit_report = validate_and_extract_citations(llm_resp.text, retrieval.chunks)
        answer_body = cit_report.cleaned_answer
        effective_status = retrieval.status
        citations_out = cit_report.citations
        if is_abstention_response(answer_body):
            effective_status = "no_evidence"
            citations_out = []
        elif retrieval.status == "low_confidence":
            answer_body = (
                "> *Low-confidence retrieval notice: Matched passages had moderate similarity; "
                "please verify the cited excerpts below.*\n\n" + answer_body
            )

        final_answer = f"{answer_body}\n\n{build_safety_footer()}"
        elapsed = (time.perf_counter() - t0) * 1000.0

        return RAGResponse(
            question=clean_q,
            rewritten_query=standalone_query,
            was_rewritten=was_rewritten,
            answer=final_answer,
            retrieval_status=effective_status,
            citations=citations_out,
            retrieved_chunks=retrieval.chunks,
            safety_flags=safety.flags,
            invalid_citations_removed=cit_report.invalid_citations_removed,
            llm_used=True,
            llm_provider=llm_resp.provider,
            llm_model_id=llm_resp.model_id,
            latency_ms=round(elapsed, 2),
            retrieval_latency_ms=retrieval.latency_ms,
        )
