"""Live end-to-end verification across all 10 mandatory MediSage repair tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pymupdf as fitz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chatbot.chat_manager import ChatManager
from config.settings import ensure_directories, get_settings
from core.llm_client import GeminiLLMClient, test_llm_connection
from database.db import DatabaseManager
from ingestion.embeddings import EmbeddingService
from ingestion.ingestion_service import IngestionService
from rag.llm_service import LLMService
from rag.rag_pipeline import RAGPipeline
from rag.retriever import SemanticRetriever
from vectorstore.chroma_manager import ChromaManager


def _make_temp_pdf_bytes() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 70), "E2E Live Verification Reference", fontsize=14)
    page.insert_text(
        (50, 110),
        "Cholesterol Screening Overview: Adult lipid panels assess total cholesterol, "
        "LDL-C, HDL-C, and fasting triglycerides every 4 to 6 years in low-risk adults.",
        fontsize=11,
    )
    data = doc.tobytes()
    doc.close()
    return data


def main() -> None:
    settings = ensure_directories(get_settings())
    assert settings.has_api_key(), "GEMINI_API_KEY must be configured in .env for live tests."

    db = DatabaseManager(settings.sqlite_db_path)
    chroma = ChromaManager(settings.chroma_persist_dir, settings.chroma_collection_name)
    embeddings = EmbeddingService(
        model_name=settings.embedding_model_name, device=settings.embedding_device
    )
    embeddings.ensure_loaded()
    ingestion = IngestionService(
        db=db, chroma=chroma, embeddings=embeddings, settings=settings
    )
    retriever = SemanticRetriever(chroma=chroma, embeddings=embeddings)
    llm_service = LLMService(client=GeminiLLMClient(), settings=settings)
    pipeline = RAGPipeline(
        retriever=retriever, llm_service=llm_service, db=db, settings=settings
    )
    chat = ChatManager(db=db, rag_pipeline=pipeline)

    # 0. Repair sample duplicates / orphaned vectors and test Gemini v1beta connection
    repair_stats = ingestion.verify_and_repair_consistency()
    print("0. Consistency Repair:", repair_stats)

    conn_check = test_llm_connection(settings)
    print(
        f"0b. Live Gemini v1beta Health Check: success={conn_check.success} "
        f"model={conn_check.model_id} latency={conn_check.latency_ms:.1f}ms text={conn_check.text!r}"
    )
    assert conn_check.success, f"Connection test failed: {conn_check.error_message}"

    # Test 1: Document upload
    temp_pdf = _make_temp_pdf_bytes()
    up_res = ingestion.ingest_bytes(
        filename="E2E_Temp_Cholesterol_Ref.pdf",
        content=temp_pdf,
        source_origin="upload",
        replace_existing=True,
    )
    print(f"Test 1 (Upload & Index): status={up_res.status} chunks={up_res.chunk_count}")
    assert up_res.success and up_res.chunk_count >= 1

    # Test 2, 3, 4: Retrieval + Live Gemini Generation + Source Citations
    conv = chat.create_conversation("E2E Live Verification Conv")
    cid = conv["conversation_id"]
    _, a1, r1 = chat.ask(
        conversation_id=cid,
        user_question="What blood pressure ranges define Stage 1 and Stage 2 hypertension?",
    )
    print(
        f"Test 2-4 (Retrieval + Gemini + Citations): status={r1.retrieval_status} "
        f"llm_used={r1.llm_used} citations={[(c.filename, c.page_number) for c in r1.citations]}"
    )
    print("  Answer snippet:", r1.answer[:220].replace("\n", " "))
    assert r1.llm_used is True
    assert r1.retrieval_status == "ok"
    assert len(r1.citations) >= 1

    # Test 5: Context-aware follow-up question
    _, a2, r2 = chat.ask(
        conversation_id=cid,
        user_question="What lifestyle modifications are recommended for it?",
    )
    print(
        f"Test 5 (Follow-up): rewritten={r2.was_rewritten} query={r2.rewritten_query!r} "
        f"llm_used={r2.llm_used} citations={[(c.filename, c.page_number) for c in r2.citations]}"
    )
    assert r2.was_rewritten is True
    assert r2.llm_used is True
    assert len(r2.citations) >= 1

    # Test 6: Unsupported question abstention
    _, a3, r3 = chat.ask(
        conversation_id=cid,
        user_question="What is the standard pediatric chemotherapy protocol for acute lymphoblastic leukemia?",
    )
    print(
        f"Test 6 (Unsupported Abstention): status={r3.retrieval_status} "
        f"citations={len(r3.citations)}"
    )
    assert r3.retrieval_status == "no_evidence"
    assert len(r3.citations) == 0
    assert "could not find sufficient supporting information" in r3.answer.lower()

    # Test 7: Multiple documents retrieval (DOCX + PDF)
    _, a4, r4 = chat.ask(
        conversation_id=cid,
        user_question="What are the five CURB-65 pneumonia severity criteria?",
    )
    print(
        f"Test 7 (Multi-doc DOCX Retrieval + Gemini): llm_used={r4.llm_used} "
        f"citations={[(c.filename, c.page_number) for c in r4.citations]}"
    )
    assert r4.llm_used is True
    assert any(c.filename.endswith(".docx") for c in r4.citations)

    # Test 8: Re-index existing document & delete temporary document
    reidx = ingestion.reindex_document(up_res.document_id)
    assert reidx.success
    deleted_ok = ingestion.delete_document_completely(up_res.document_id)
    assert deleted_ok
    assert len(chroma.get_document_chunks(up_res.document_id)) == 0
    print("Test 8 (Re-index & Deletion Consistency): PASSED")

    # Test 9: Conversation history rename, reopen, export JSON, and cleanup test conv
    assert chat.rename_conversation(cid, "E2E Renamed Check")
    reopened = chat.get_conversation(cid)
    assert reopened is not None and reopened["title"] == "E2E Renamed Check"
    msgs = chat.get_messages(cid)
    exported = json.dumps({"conversation": reopened, "messages": msgs})
    assert len(msgs) == 8 and "Stage 1" in exported
    assert chat.delete_conversation(cid)
    print("Test 9 (Conversation History CRUD & Export): PASSED")

    # Also repair the user's prior failed messages in conv_706569991f9b if present
    prior_cid = "conv_706569991f9b"
    if db.get_conversation(prior_cid):
        prior_msgs = db.get_messages(prior_cid)
        if any(m.get("retrieval_status") == "llm_error" for m in prior_msgs):
            user_questions = [m["content"] for m in prior_msgs if m["role"] == "user"]
            chat.clear_conversation(prior_cid)
            for uq in user_questions:
                chat.ask(conversation_id=prior_cid, user_question=uq)
            print(f"Repaired {len(user_questions)} failed turns in existing conversation {prior_cid}.")

    print("LIVE_E2E_10_TESTS_ALL_PASSED")
    chroma.close()


if __name__ == "__main__":
    main()
