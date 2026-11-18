"""Conversation persistence, bounded context loading, and chat orchestration."""

from __future__ import annotations

from typing import Any

from database.db import DatabaseManager
from rag.rag_pipeline import RAGPipeline, RAGResponse


class ChatManager:
    """Manages multi-turn medical conversations persisted in SQLite."""

    def __init__(
        self,
        *,
        db: DatabaseManager,
        rag_pipeline: RAGPipeline,
        max_history_messages: int = 6,
    ) -> None:
        self.db = db
        self.rag_pipeline = rag_pipeline
        self.max_history_messages = max(2, int(max_history_messages))

    def create_conversation(self, title: str = "New Medical Query") -> dict[str, Any]:
        return self.db.create_conversation(title=title)

    def list_conversations(self, search_query: str = "") -> list[dict[str, Any]]:
        return self.db.list_conversations(search_query=search_query)

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        return self.db.get_conversation(conversation_id)

    def rename_conversation(self, conversation_id: str, new_title: str) -> bool:
        return self.db.rename_conversation(conversation_id, new_title)

    def delete_conversation(self, conversation_id: str) -> bool:
        return self.db.delete_conversation(conversation_id)

    def clear_conversation(self, conversation_id: str) -> int:
        return self.db.clear_conversation_messages(conversation_id)

    def get_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        return self.db.get_messages(conversation_id)

    def submit_feedback(
        self,
        *,
        message_id: str,
        conversation_id: str,
        rating: str,
        comment: str = "",
    ) -> dict[str, Any]:
        return self.db.add_feedback(
            message_id=message_id,
            conversation_id=conversation_id,
            rating=rating,
            comment=comment,
        )

    def ask(
        self,
        *,
        conversation_id: str,
        user_question: str,
        metadata_filters: dict[str, Any] | None = None,
        setting_overrides: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], RAGResponse]:
        """Persist user message, run RAG pipeline with bounded history, and save assistant response."""
        conv = self.db.get_conversation(conversation_id)
        if conv is None:
            conv = self.db.create_conversation(
                title=user_question[:60], conversation_id=conversation_id
            )

        # Auto-title new conversations from the first user query
        existing_msgs = self.db.get_messages(conversation_id)
        if not existing_msgs and conv.get("title") in {
            "New Medical Query",
            "New Conversation",
        }:
            clean_title = " ".join(user_question.split())[:58]
            if len(user_question.strip()) > 58:
                clean_title += "..."
            self.db.rename_conversation(conversation_id, clean_title or "Medical Query")

        # Load bounded recent history BEFORE adding current message
        recent = self.db.get_recent_messages(
            conversation_id, limit=self.max_history_messages
        )

        user_msg = self.db.add_message(
            conversation_id=conversation_id,
            role="user",
            content=user_question.strip(),
        )

        rag_resp = self.rag_pipeline.answer_question(
            user_question,
            recent_messages=recent,
            metadata_filters=metadata_filters,
            setting_overrides=setting_overrides,
        )

        assistant_msg = self.db.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=rag_resp.answer,
            rewritten_query=rag_resp.rewritten_query if rag_resp.was_rewritten else "",
            retrieval_status=rag_resp.retrieval_status,
            citations=[c.to_dict() for c in rag_resp.citations],
            safety_flags=rag_resp.safety_flags,
            latency_ms=rag_resp.latency_ms,
        )

        return user_msg, assistant_msg, rag_resp
