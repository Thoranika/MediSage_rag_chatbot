"""SQLite schema, CRUD, duplicate hash detection, and cascade deletion tests."""

from __future__ import annotations


def test_schema_initializes_and_health_check_passes(isolated_env) -> None:
    _, db, _ = isolated_env
    assert db.health_check() is True
    stats = db.get_knowledge_stats()
    assert stats["total_documents"] == 0
    assert stats["conversation_count"] == 0


def test_document_crud_and_duplicate_hash_lookup(isolated_env) -> None:
    _, db, _ = isolated_env
    rec = db.add_document(
        stored_filename="doc_abc_1234.pdf",
        original_filename="guideline.pdf",
        file_hash="deadbeef" * 8,
        file_type="pdf",
        file_size_bytes=2048,
        processing_status="processing",
    )
    assert rec["original_filename"] == "guideline.pdf"
    found = db.get_document_by_hash("deadbeef" * 8)
    assert found is not None
    assert found["document_id"] == rec["document_id"]

    updated = db.update_document_status(
        rec["document_id"],
        processing_status="indexed",
        page_count=4,
        chunk_count=10,
    )
    assert updated is not None
    assert updated["processing_status"] == "indexed"
    assert updated["chunk_count"] == 10

    assert db.delete_document(rec["document_id"]) is True
    assert db.get_document(rec["document_id"]) is None


def test_conversation_message_and_feedback_cascade_delete(isolated_env) -> None:
    _, db, _ = isolated_env
    conv = db.create_conversation("Hypertension Follow-up")
    cid = conv["conversation_id"]

    db.add_message(conversation_id=cid, role="user", content="What is Stage 1 BP?")
    a_msg = db.add_message(
        conversation_id=cid,
        role="assistant",
        content="130-139/80-89 mmHg [Source: guide.pdf, page 1]",
        retrieval_status="ok",
        citations=[{"filename": "guide.pdf", "page_number": 1, "excerpt": "Stage 1..."}],
    )
    db.add_feedback(
        message_id=a_msg["message_id"],
        conversation_id=cid,
        rating="helpful",
        comment="Clear citation.",
    )

    assert len(db.get_messages(cid)) == 2
    assert db.get_feedback_stats()["helpful"] == 1

    # Rename and bounded history
    assert db.rename_conversation(cid, "Renamed Query") is True
    assert db.get_conversation(cid)["title"] == "Renamed Query"
    assert len(db.get_recent_messages(cid, limit=1)) == 1

    # Deleting conversation must cascade delete messages and feedback
    assert db.delete_conversation(cid) is True
    assert db.get_messages(cid) == []
    assert db.get_feedback_stats()["total"] == 0


def test_persistent_settings_and_evaluation_runs(isolated_env) -> None:
    _, db, _ = isolated_env
    db.set_settings_bulk({"retrieval_top_k": 6, "similarity_threshold": 0.3})
    loaded = db.get_all_settings()
    assert loaded["retrieval_top_k"] == 6
    assert loaded["similarity_threshold"] == 0.3

    run_id = db.save_evaluation_run(
        "Unit Eval", {"top_k": 4}, {"recall_at_k": 0.95}
    )
    runs = db.list_evaluation_runs()
    assert any(r["run_id"] == run_id for r in runs)
