"""Standalone follow-up query rewriter using bounded recent chat context."""

from __future__ import annotations

import re
from typing import Any

_PRONOUN_OR_FOLLOWUP_RE = re.compile(
    r"\b(it|its|they|them|their|this|that|these|those|such|same|"
    r"the condition|the disease|the drug|the medication|the treatment|"
    r"above|previous|mentioned)\b",
    re.IGNORECASE,
)
_SHORT_FOLLOWUP_PREFIX_RE = re.compile(
    r"^(?:what about|how about|and what|what are|any|side effects|contraindications|"
    r"symptoms|targets|criteria|in pregnancy|in elderly|first[- ]line)\b",
    re.IGNORECASE,
)


def _extract_anchor_topic(recent_messages: list[dict[str, Any]]) -> str:
    """Extract a concise topic phrase from the latest prior user questions."""
    prior_user_msgs = [
        str(m.get("content", "")).strip()
        for m in recent_messages
        if m.get("role") == "user" and str(m.get("content", "")).strip()
    ]
    if not prior_user_msgs:
        return ""

    last_q = prior_user_msgs[-1]
    # Remove trailing punctuation and question words to form a crisp context anchor
    anchor = re.sub(r"[?!.]+$", "", last_q).strip()
    if len(anchor) > 120:
        anchor = anchor[:120].rsplit(" ", 1)[0]
    return anchor


def rewrite_followup_query(
    user_query: str,
    recent_messages: list[dict[str, Any]] | None = None,
) -> tuple[str, bool]:
    """
    Rewrite context-dependent follow-up queries into standalone retrieval queries.

    Returns (standalone_query, was_rewritten). Never alters the user's displayed text.
    """
    original = (user_query or "").strip()
    if not original or not recent_messages:
        return original, False

    needs_rewrite = bool(_PRONOUN_OR_FOLLOWUP_RE.search(original)) or (
        len(original.split()) <= 7 and bool(_SHORT_FOLLOWUP_PREFIX_RE.search(original))
    )
    if not needs_rewrite:
        return original, False

    anchor = _extract_anchor_topic(recent_messages)
    if not anchor:
        return original, False

    # Avoid duplicating anchor if already present
    anchor_words = {w.lower() for w in anchor.split() if len(w) > 4}
    query_words = {w.lower().strip("?.,") for w in original.split()}
    if anchor_words and len(anchor_words & query_words) >= max(2, len(anchor_words) // 2):
        return original, False

    standalone = f"{original} (Context: {anchor})"
    return standalone, True
