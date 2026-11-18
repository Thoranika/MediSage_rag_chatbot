"""Grounded medical prompt construction enforcing strict source fidelity and citations."""

from __future__ import annotations

from typing import Any

from rag.retriever import RetrievedChunk
from rag.safety import SafetyAssessment

BASE_SYSTEM_INSTRUCTION = (
    "You are MediSage, an evidence-grounded medical knowledge assistant designed strictly "
    "for educational reference from user-uploaded medical documents.\n"
    "Rules:\n"
    "1. Answer ONLY using facts explicitly present in the supplied Source Passages.\n"
    "2. Do NOT diagnose conditions, prescribe medications, recommend personalized dosages, "
    "or instruct a user to start or stop treatment.\n"
    "3. If the supplied passages do not contain enough evidence to answer the question, "
    "respond with: 'I could not find sufficient supporting information in your uploaded medical "
    "reference documents. Please consult an authoritative medical source or qualified healthcare professional.'\n"
    "4. Cite supporting facts inline using the exact format [Source: <filename>, page <page_number>]. "
    "Never invent filenames, page numbers, numbers, or clinical claims.\n"
    "5. If retrieved sources state conflicting facts, explain the conflict transparently and cite both sources.\n"
    "6. Do not infer unstated patient details."
)

_STYLE_GUIDANCE = {
    "Concise": "Keep the explanation concise, structured with short bullet points where helpful, and in plain language.",
    "Detailed": "Provide a thorough, well-structured explanation covering all relevant clinical details found in the passages.",
    "Plain Language": "Explain medical terms in clear, accessible everyday language suitable for general educational reading.",
}


def build_grounded_prompt(
    *,
    user_question: str,
    retrieved_chunks: list[RetrievedChunk],
    recent_messages: list[dict[str, Any]] | None = None,
    safety: SafetyAssessment | None = None,
    response_style: str = "Concise",
    max_history_messages: int = 4,
) -> tuple[str, str]:
    """Construct (system_instruction, user_prompt) for cloud LLM generation."""
    style_note = _STYLE_GUIDANCE.get(response_style, _STYLE_GUIDANCE["Concise"])
    system_parts = [BASE_SYSTEM_INSTRUCTION, f"Style guidance: {style_note}"]
    if safety and safety.system_guidance:
        system_parts.append(safety.system_guidance)
    system_instruction = "\n\n".join(system_parts)

    sections: list[str] = []

    # Bounded conversation history (excluding retrieved passages to avoid context bloat)
    if recent_messages:
        trimmed = recent_messages[-max_history_messages:]
        history_lines: list[str] = []
        for msg in trimmed:
            role_label = "User" if msg.get("role") == "user" else "Assistant"
            text = str(msg.get("content", "")).strip()
            if len(text) > 350:
                text = text[:350].rstrip() + "..."
            if text:
                history_lines.append(f"{role_label}: {text}")
        if history_lines:
            sections.append(
                "### Recent Conversation Context (for follow-up continuity only)\n"
                + "\n".join(history_lines)
            )

    # Retrieved source passages with explicit source/page headers
    passage_blocks: list[str] = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        section_tag = f" | Section: {chunk.section_title}" if chunk.section_title else ""
        passage_blocks.append(
            f"[Passage {idx} | Source: {chunk.filename}, page {chunk.page_number}{section_tag} | Similarity: {chunk.similarity:.2f}]\n"
            f"{chunk.text}"
        )

    sections.append("### Retrieved Source Passages\n" + "\n\n".join(passage_blocks))
    sections.append(f"### User Question\n{user_question.strip()}")
    sections.append(
        "### Instructions\n"
        "Answer the User Question using ONLY the Retrieved Source Passages above. "
        "Include exact citations formatted as [Source: filename, page N] for every factual claim. "
        "If the passages do not answer the question, reply: "
        "'I could not find sufficient supporting information in your uploaded medical reference documents. "
        "Please consult an authoritative medical source or qualified healthcare professional.'"
    )

    return system_instruction, "\n\n".join(sections)
