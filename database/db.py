"""SQLite database manager with parameterized queries and CRUD operations."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from core.logging_utils import get_logger
from utils.helpers import utc_now_iso

logger = get_logger("database")

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


class DatabaseManager:
    """Thread-safe parameterized SQLite persistence manager for MediSage."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON;")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        """Create tables and indexes automatically from schema.sql."""
        schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")
        with self._connection() as conn:
            conn.executescript(schema_sql)

    def health_check(self) -> bool:
        """Return True if SQLite database responds to a lightweight check."""
        try:
            with self._connection() as conn:
                row = conn.execute("SELECT 1 AS ok").fetchone()
                return bool(row and row["ok"] == 1)
        except sqlite3.Error:
            return False

    # ------------------------------------------------------------------
    # Document CRUD
    # ------------------------------------------------------------------
    def add_document(
        self,
        *,
        stored_filename: str,
        original_filename: str,
        file_hash: str,
        file_type: str,
        file_size_bytes: int,
        processing_status: str = "processing",
        page_count: int = 0,
        chunk_count: int = 0,
        error_details: str = "",
        source_origin: str = "upload",
        document_id: str | None = None,
    ) -> dict[str, Any]:
        doc_id = document_id or f"doc_{uuid4().hex[:12]}"
        now = utc_now_iso()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    document_id, stored_filename, original_filename, file_hash,
                    file_type, file_size_bytes, page_count, chunk_count,
                    processing_status, error_details, source_origin,
                    upload_timestamp, updated_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    stored_filename,
                    original_filename,
                    file_hash,
                    file_type,
                    int(file_size_bytes),
                    int(page_count),
                    int(chunk_count),
                    processing_status,
                    error_details,
                    source_origin,
                    now,
                    now,
                ),
            )
        record = self.get_document(doc_id)
        assert record is not None
        return record

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
            return dict(row) if row else None

    def get_document_by_hash(self, file_hash: str) -> dict[str, Any] | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE file_hash = ?",
                (file_hash,),
            ).fetchone()
            return dict(row) if row else None

    def list_documents(self, status: str | None = None) -> list[dict[str, Any]]:
        with self._connection() as conn:
            if status:
                rows = conn.execute(
                    """
                    SELECT * FROM documents
                    WHERE processing_status = ?
                    ORDER BY upload_timestamp DESC
                    """,
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM documents ORDER BY upload_timestamp DESC"
                ).fetchall()
            return [dict(r) for r in rows]

    def update_document_status(
        self,
        document_id: str,
        *,
        processing_status: str,
        page_count: int | None = None,
        chunk_count: int | None = None,
        error_details: str | None = None,
        stored_filename: str | None = None,
        file_size_bytes: int | None = None,
    ) -> dict[str, Any] | None:
        existing = self.get_document(document_id)
        if not existing:
            return None
        now = utc_now_iso()
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE documents
                SET processing_status = ?,
                    page_count = ?,
                    chunk_count = ?,
                    error_details = ?,
                    stored_filename = ?,
                    file_size_bytes = ?,
                    updated_timestamp = ?
                WHERE document_id = ?
                """,
                (
                    processing_status,
                    int(page_count) if page_count is not None else existing["page_count"],
                    int(chunk_count) if chunk_count is not None else existing["chunk_count"],
                    error_details if error_details is not None else existing["error_details"],
                    stored_filename if stored_filename is not None else existing["stored_filename"],
                    int(file_size_bytes) if file_size_bytes is not None else existing["file_size_bytes"],
                    now,
                    document_id,
                ),
            )
        return self.get_document(document_id)

    def delete_document(self, document_id: str) -> bool:
        with self._connection() as conn:
            cur = conn.execute(
                "DELETE FROM documents WHERE document_id = ?",
                (document_id,),
            )
            return cur.rowcount > 0

    def get_knowledge_stats(self) -> dict[str, Any]:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_documents,
                    SUM(CASE WHEN processing_status = 'indexed' THEN 1 ELSE 0 END) AS indexed_documents,
                    SUM(CASE WHEN processing_status = 'failed' THEN 1 ELSE 0 END) AS failed_documents,
                    COALESCE(SUM(CASE WHEN processing_status = 'indexed' THEN chunk_count ELSE 0 END), 0) AS total_chunks,
                    COALESCE(SUM(CASE WHEN processing_status = 'indexed' THEN page_count ELSE 0 END), 0) AS total_pages,
                    MAX(updated_timestamp) AS last_updated
                FROM documents
                """
            ).fetchone()
            conv_row = conn.execute(
                "SELECT COUNT(*) AS conversation_count FROM conversations"
            ).fetchone()
            msg_row = conn.execute(
                "SELECT COUNT(*) AS message_count FROM messages"
            ).fetchone()
            return {
                "total_documents": int(row["total_documents"] or 0),
                "indexed_documents": int(row["indexed_documents"] or 0),
                "failed_documents": int(row["failed_documents"] or 0),
                "total_chunks": int(row["total_chunks"] or 0),
                "total_pages": int(row["total_pages"] or 0),
                "last_updated": row["last_updated"] or "",
                "conversation_count": int(conv_row["conversation_count"] or 0),
                "message_count": int(msg_row["message_count"] or 0),
            }

    # ------------------------------------------------------------------
    # Conversation & Message CRUD
    # ------------------------------------------------------------------
    def create_conversation(
        self, title: str = "New Medical Query", conversation_id: str | None = None
    ) -> dict[str, Any]:
        cid = conversation_id or f"conv_{uuid4().hex[:12]}"
        clean_title = (title or "New Medical Query").strip()[:120]
        now = utc_now_iso()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO conversations (conversation_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (cid, clean_title, now, now),
            )
        rec = self.get_conversation(cid)
        assert rec is not None
        return rec

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            return dict(row) if row else None

    def list_conversations(self, search_query: str = "") -> list[dict[str, Any]]:
        with self._connection() as conn:
            if search_query and search_query.strip():
                like_expr = f"%{search_query.strip()}%"
                rows = conn.execute(
                    """
                    SELECT c.*,
                           (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.conversation_id) AS message_count
                    FROM conversations c
                    WHERE c.title LIKE ?
                    ORDER BY c.updated_at DESC
                    """,
                    (like_expr,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT c.*,
                           (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.conversation_id) AS message_count
                    FROM conversations c
                    ORDER BY c.updated_at DESC
                    """
                ).fetchall()
            return [dict(r) for r in rows]

    def rename_conversation(self, conversation_id: str, new_title: str) -> bool:
        clean_title = new_title.strip()[:120]
        if not clean_title:
            return False
        now = utc_now_iso()
        with self._connection() as conn:
            cur = conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE conversation_id = ?",
                (clean_title, now, conversation_id),
            )
            return cur.rowcount > 0

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._connection() as conn:
            cur = conn.execute(
                "DELETE FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )
            return cur.rowcount > 0

    def clear_conversation_messages(self, conversation_id: str) -> int:
        now = utc_now_iso()
        with self._connection() as conn:
            cur = conn.execute(
                "DELETE FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                (now, conversation_id),
            )
            return cur.rowcount

    def add_message(
        self,
        *,
        conversation_id: str,
        role: str,
        content: str,
        rewritten_query: str = "",
        retrieval_status: str = "",
        citations: list[dict[str, Any]] | None = None,
        safety_flags: list[str] | None = None,
        latency_ms: float = 0.0,
        message_id: str | None = None,
    ) -> dict[str, Any]:
        if role not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'")
        mid = message_id or f"msg_{uuid4().hex[:12]}"
        now = utc_now_iso()

        # Store only compact citation metadata & brief excerpts (never full source documents)
        compact_citations: list[dict[str, Any]] = []
        for item in citations or []:
            compact_citations.append(
                {
                    "document_id": item.get("document_id", ""),
                    "filename": item.get("filename", ""),
                    "page_number": item.get("page_number", 1),
                    "section_title": item.get("section_title", ""),
                    "similarity_score": round(float(item.get("similarity_score", 0.0)), 4),
                    "excerpt": str(item.get("excerpt", ""))[:450],
                    "verified": bool(item.get("verified", True)),
                }
            )

        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO messages (
                    message_id, conversation_id, role, content,
                    rewritten_query, retrieval_status, citations_json,
                    safety_flags_json, latency_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mid,
                    conversation_id,
                    role,
                    content,
                    rewritten_query,
                    retrieval_status,
                    json.dumps(compact_citations),
                    json.dumps(safety_flags or []),
                    float(latency_ms),
                    now,
                ),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                (now, conversation_id),
            )
            row = conn.execute(
                "SELECT * FROM messages WHERE message_id = ?",
                (mid,),
            ).fetchone()
            return self._deserialize_message(dict(row))

    @staticmethod
    def _deserialize_message(row: dict[str, Any]) -> dict[str, Any]:
        try:
            row["citations"] = json.loads(row.get("citations_json") or "[]")
        except json.JSONDecodeError:
            row["citations"] = []
        try:
            row["safety_flags"] = json.loads(row.get("safety_flags_json") or "[]")
        except json.JSONDecodeError:
            row["safety_flags"] = []
        return row

    def get_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC, rowid ASC
                """,
                (conversation_id,),
            ).fetchall()
            return [self._deserialize_message(dict(r)) for r in rows]

    def get_recent_messages(
        self, conversation_id: str, limit: int = 6
    ) -> list[dict[str, Any]]:
        """Return the most recent bounded messages in chronological order."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM (
                    SELECT rowid AS _rid, * FROM messages
                    WHERE conversation_id = ?
                    ORDER BY created_at DESC, rowid DESC
                    LIMIT ?
                ) sub
                ORDER BY created_at ASC, _rid ASC
                """,
                (conversation_id, max(1, int(limit))),
            ).fetchall()
            out: list[dict[str, Any]] = []
            for r in rows:
                d = dict(r)
                d.pop("_rid", None)
                out.append(self._deserialize_message(d))
            return out

    # ------------------------------------------------------------------
    # Feedback CRUD
    # ------------------------------------------------------------------
    def add_feedback(
        self,
        *,
        message_id: str,
        conversation_id: str,
        rating: str,
        comment: str = "",
    ) -> dict[str, Any]:
        if rating not in {"helpful", "not_helpful"}:
            raise ValueError("rating must be 'helpful' or 'not_helpful'")
        now = utc_now_iso()
        with self._connection() as conn:
            existing = conn.execute(
                "SELECT feedback_id FROM feedback WHERE message_id = ?",
                (message_id,),
            ).fetchone()
            if existing:
                fid = existing["feedback_id"]
                conn.execute(
                    """
                    UPDATE feedback
                    SET rating = ?, comment = ?, created_at = ?
                    WHERE feedback_id = ?
                    """,
                    (rating, comment.strip()[:500], now, fid),
                )
            else:
                fid = f"fb_{uuid4().hex[:12]}"
                conn.execute(
                    """
                    INSERT INTO feedback (
                        feedback_id, message_id, conversation_id, rating, comment, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (fid, message_id, conversation_id, rating, comment.strip()[:500], now),
                )
            row = conn.execute(
                "SELECT * FROM feedback WHERE feedback_id = ?", (fid,)
            ).fetchone()
            return dict(row)

    def get_feedback_for_conversation(self, conversation_id: str) -> dict[str, str]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT message_id, rating FROM feedback WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchall()
            return {r["message_id"]: r["rating"] for r in rows}

    def list_feedback(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT f.*, m.content AS message_preview, c.title AS conversation_title
                FROM feedback f
                LEFT JOIN messages m ON f.message_id = m.message_id
                LEFT JOIN conversations c ON f.conversation_id = c.conversation_id
                ORDER BY f.created_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_feedback_stats(self) -> dict[str, int]:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN rating = 'helpful' THEN 1 ELSE 0 END) AS helpful,
                    SUM(CASE WHEN rating = 'not_helpful' THEN 1 ELSE 0 END) AS not_helpful
                FROM feedback
                """
            ).fetchone()
            return {
                "total": int(row["total"] or 0),
                "helpful": int(row["helpful"] or 0),
                "not_helpful": int(row["not_helpful"] or 0),
            }

    def delete_feedback(self, feedback_id: str) -> bool:
        with self._connection() as conn:
            cur = conn.execute(
                "DELETE FROM feedback WHERE feedback_id = ?",
                (feedback_id,),
            )
            return cur.rowcount > 0

    def delete_all_conversations_and_feedback(self) -> int:
        with self._connection() as conn:
            conn.execute("DELETE FROM feedback")
            conn.execute("DELETE FROM messages")
            cur = conn.execute("DELETE FROM conversations")
            return cur.rowcount

    # ------------------------------------------------------------------
    # Persistent Settings & Evaluation Runs
    # ------------------------------------------------------------------
    def set_setting(self, key: str, value: Any) -> None:
        now = utc_now_iso()
        serialized = json.dumps(value)
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO app_settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (key, serialized, now),
            )

    def set_settings_bulk(self, items: dict[str, Any]) -> None:
        now = utc_now_iso()
        with self._connection() as conn:
            for key, value in items.items():
                conn.execute(
                    """
                    INSERT INTO app_settings (key, value, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                    """,
                    (key, json.dumps(value), now),
                )

    def get_setting(self, key: str, default: Any = None) -> Any:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT value FROM app_settings WHERE key = ?", (key,)
            ).fetchone()
            if not row:
                return default
            try:
                return json.loads(row["value"])
            except json.JSONDecodeError:
                return row["value"]

    def get_all_settings(self) -> dict[str, Any]:
        with self._connection() as conn:
            rows = conn.execute("SELECT key, value FROM app_settings").fetchall()
            out: dict[str, Any] = {}
            for r in rows:
                try:
                    out[r["key"]] = json.loads(r["value"])
                except json.JSONDecodeError:
                    out[r["key"]] = r["value"]
            return out

    def save_evaluation_run(
        self, run_label: str, config: dict[str, Any], metrics: dict[str, Any]
    ) -> str:
        run_id = f"eval_{uuid4().hex[:10]}"
        now = utc_now_iso()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO evaluation_runs (run_id, run_label, config_json, metrics_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, run_label, json.dumps(config), json.dumps(metrics), now),
            )
        return run_id

    def list_evaluation_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM evaluation_runs ORDER BY created_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["config"] = json.loads(item.pop("config_json", "{}"))
                item["metrics"] = json.loads(item.pop("metrics_json", "{}"))
                results.append(item)
            return results
